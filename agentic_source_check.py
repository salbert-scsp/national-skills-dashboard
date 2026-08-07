"""
Credibility audit layer.

Runs AFTER the cross-encoder has already established that a candidate page is about
the right subject. Its job is narrower than it looks: judge whether that correct page
is authoritative and non-promotional, and distill it into a clean 1-2 sentence
definition for embedding. It scores nothing and scrapes nothing.

Uses gemini-flash-lite-latest with a structured Pydantic response schema, one call
per audited skill, sized to run inside free-tier quota.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, Field

from gemini_keys import DailyQuotaExhausted, GeminiKeyPool

logger = logging.getLogger(__name__)

load_dotenv()

# Retry policy, mirroring scraping.request_with_backoff so both network legs behave
# the same way. Free-tier RPM is the binding constraint, so the initial wait is
# longer than the scraper's.
MAX_ATTEMPTS = 5
INITIAL_BACKOFF = 3.0
MAX_BACKOFF = 60.0

# How long to wait before deciding an unclassifiable 429 is the daily wall. A per-minute
# window is 60s, so a limit that survives the wait is not a per-minute limit.
DAILY_PROBE_WAIT = 60.0

# Retried: quota and transient server failures. Anything else (400 bad request,
# 403 bad key) will not improve on retry and is raised immediately.
RETRYABLE_CODES = {429, 500, 502, 503, 504}
RETRYABLE_STATUSES = {"RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED"}

# 429 classifications.
QUOTA_PER_MINUTE = "per_minute"
QUOTA_PER_DAY = "per_day"
QUOTA_UNKNOWN = "unknown"


def _quota_violations(error: genai_errors.APIError) -> list:
    """Pulls the QuotaFailure violations out of a Google API error, if present."""
    details = error.details
    if not isinstance(details, dict):
        return []

    error_block = details.get("error", details)
    if not isinstance(error_block, dict):
        return []

    violations = []
    for entry in error_block.get("details", []) or []:
        if not isinstance(entry, dict):
            continue
        if "QuotaFailure" in str(entry.get("@type", "")):
            for violation in entry.get("violations", []) or []:
                if isinstance(violation, dict):
                    violations.append(violation)
    return violations


def classify_429(error: genai_errors.APIError) -> str:
    """
    Decides whether a 429 is a per-minute throttle or the daily wall.

    Gemini returns 429 for both, and the difference matters enormously: a per-minute
    limit clears by waiting, while a daily one can only be answered with a different
    key or tomorrow. Treating a minute burst as the daily wall would retire every key
    in the pool over a few seconds of throttling.

    Google usually names the limit in a QuotaFailure violation, e.g.
    "GenerateRequestsPerDayPerProjectPerModel". When it does, believe it. When it does
    not, return UNKNOWN and let the caller fall back to the timing probe.
    """
    haystacks = []
    for violation in _quota_violations(error):
        haystacks.append(str(violation.get("quotaId", "")))
        haystacks.append(str(violation.get("quotaMetric", "")))

    joined = " ".join(haystacks).lower()
    if "perday" in joined or "per_day" in joined:
        return QUOTA_PER_DAY
    if "perminute" in joined or "per_minute" in joined:
        return QUOTA_PER_MINUTE

    # Nothing structured. The message sometimes says it in prose.
    message = str(getattr(error, "message", "") or "").lower()
    if "per day" in message or "daily" in message:
        return QUOTA_PER_DAY
    if "per minute" in message:
        return QUOTA_PER_MINUTE

    return QUOTA_UNKNOWN


def _retry_after_seconds(error: genai_errors.APIError) -> Optional[float]:
    """Extracts Google's suggested RetryInfo delay, e.g. 'retryDelay': '27s'."""
    details = error.details
    if not isinstance(details, dict):
        return None

    error_block = details.get("error", details)
    if not isinstance(error_block, dict):
        return None

    for entry in error_block.get("details", []) or []:
        if not isinstance(entry, dict):
            continue
        delay = entry.get("retryDelay")
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                return float(delay[:-1])
            except ValueError:
                continue
    return None


def _is_retryable(error: genai_errors.APIError) -> bool:
    if getattr(error, "code", None) in RETRYABLE_CODES:
        return True
    return str(getattr(error, "status", "")).upper() in RETRYABLE_STATUSES


class MultiSourceEvaluation(BaseModel):
    is_credible: bool = Field(
        description="True if at least one candidate source is highly authoritative, accurate, and relevant."
    )
    best_source_name: Optional[str] = Field(
        default="None",
        description="The exact name of the best candidate source. Set to 'None' if no source is credible.",
    )
    confidence_score: float = Field(
        description="Confidence level in this evaluation from 0.0 to 1.0."
    )
    clean_summary: Optional[str] = Field(
        default=None,
        description="A concise, 1-2 sentence definition or summary distilled from the best source.",
    )


class ReferenceProposal(BaseModel):
    """
    An answer to "which encyclopedia article is this skill actually about?".

    Deliberately separate from MultiSourceEvaluation. The audit grades text it is
    handed; this picks which text to hand it. Keeping them apart is what lets the
    second pass treat agreement between the two as two signals rather than one.
    """

    article_exists: bool = Field(
        description="True only if an English Wikipedia article covers this specific product or technology."
    )
    proposed_title: Optional[str] = Field(
        default=None,
        description="The exact English Wikipedia article title. Null when article_exists is false.",
    )
    same_as_current: bool = Field(
        default=False,
        description="True if the article already on file is the correct one.",
    )
    confidence: float = Field(
        default=0.0, description="Confidence in this proposal from 0.0 to 1.0."
    )
    rationale: Optional[str] = Field(
        default=None, description="One short sentence, at most 160 characters."
    )


class DefinitionDraft(BaseModel):
    """
    A definition written from the taxonomy alone, for products with no article.

    The output of this has NO external source behind it, which is why callers must
    mark it as such and must never let it auto-approve.
    """

    definition: Optional[str] = Field(
        default=None,
        description="A factual 1-2 sentence definition, or null if the product is unknown to you.",
    )
    is_software_product: bool = Field(
        default=True,
        description="False if the name does not appear to denote a real software product or technology.",
    )


class AuditUnavailable(RuntimeError):
    """
    The credibility audit could not be performed at all.

    Distinct from a verdict of "not credible": this means quota exhaustion or a
    hard API failure, so the skill is UNAUDITED. Callers must record is_credible
    as NULL rather than 0, or a rate-limit outage gets permanently filed as a
    source-quality judgement.
    """

    def __init__(self, skill_name: str):
        super().__init__(f"Gemini credibility audit unavailable for {skill_name!r}")
        self.skill_name = skill_name


UNAUDITED_RESULT: Dict[str, Any] = {
    "is_credible": False,
    "best_source_name": "None",
    "confidence_score": 0.0,
    "clean_summary": None,
}


class AgenticSourceChecker:
    def __init__(self, pool: Optional[GeminiKeyPool] = None) -> None:
        self.pool = pool or GeminiKeyPool()
        self.model_id = "gemini-flash-lite-latest"
        self.client = None
        self._build_client()

    def _build_client(self) -> None:
        """
        Constructs the client on the pool's current key.

        genai.Client binds its key at construction, so rotating means building a new
        client rather than reassigning an attribute on the old one.
        """
        key = self.pool.current_key()
        if key is None:
            raise DailyQuotaExhausted(self.pool.total_keys())
        self.client = genai.Client(api_key=key)

    def _rotate_or_give_up(self) -> None:
        """
        Retires the current key and rebuilds on the next one.

        Raises DailyQuotaExhausted when the pool is spent, which is the signal the
        ingestion loop uses to stop and write its backlog.
        """
        if not self.pool.mark_exhausted_and_advance():
            raise DailyQuotaExhausted(self.pool.total_keys())
        self._build_client()

    def _generate_with_backoff(
        self,
        prompt: str,
        system_instruction: str,
        skill_name: str,
        *,
        response_schema: type = MultiSourceEvaluation,
        max_output_tokens: int = 256,
    ):
        """
        Issues one structured call, retrying quota and transient failures with doubling backoff.

        Without this, a single 429 was swallowed by the caller's except clause and the
        skill silently fell back to O*NET boilerplate that then got embedded and scored
        as though it were a real definition. A quota wall must never be mistaken for a
        credibility verdict, so exhaustion raises rather than returning a verdict.

        The schema and token budget are parameters solely so the second-pass calls can
        share this engine. Every key rotation, 429 classification and backoff decision
        below must apply identically to every call the process makes, or two callers
        would keep separate and disagreeing views of which keys are still alive.
        The defaults are the credibility audit's original hardcoded values.
        """
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=0.0,
            max_output_tokens=max_output_tokens,
        )

        backoff = INITIAL_BACKOFF
        # Tracks whether the 60-second probe has already run for the CURRENT key. It
        # resets on rotation, because a fresh key deserves its own chance to prove
        # whether its 429 is a minute limit or a day limit.
        probed_this_key = False

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return self.client.models.generate_content(
                    model=self.model_id,
                    contents=prompt,
                    config=config,
                )
            except genai_errors.APIError as err:
                if not _is_retryable(err):
                    logger.error(
                        "Gemini returned non-retryable %s (%s) for %r. Not retrying.",
                        getattr(err, "code", "?"), getattr(err, "status", "?"), skill_name,
                    )
                    raise

                if getattr(err, "code", None) == 429:
                    kind = classify_429(err)

                    if kind == QUOTA_PER_DAY:
                        # Google has said the day is gone. Waiting cannot help, so
                        # rotate immediately rather than burning 60 seconds first.
                        logger.warning(
                            "Gemini key %s reports its DAILY quota is spent (while "
                            "auditing %r). Rotating without waiting.",
                            self.pool.current_label(), skill_name,
                        )
                        self._rotate_or_give_up()
                        probed_this_key = False
                        continue

                    if kind == QUOTA_UNKNOWN:
                        if not probed_this_key:
                            # The rule as specified: a 429 that survives a full minute
                            # is not a per-minute limit.
                            logger.warning(
                                "Gemini 429 for %r on key %s with no quota detail. "
                                "Waiting %.0fs to tell a minute limit from the daily wall.",
                                skill_name, self.pool.current_label(), DAILY_PROBE_WAIT,
                            )
                            time.sleep(DAILY_PROBE_WAIT)
                            probed_this_key = True
                            continue

                        logger.warning(
                            "Gemini still 429 for %r on key %s after the %.0fs wait. "
                            "Treating this as the daily limit.",
                            skill_name, self.pool.current_label(), DAILY_PROBE_WAIT,
                        )
                        self._rotate_or_give_up()
                        probed_this_key = False
                        continue

                    # QUOTA_PER_MINUTE: a throttle, not a wall. Honour Google's
                    # suggested delay and retry the SAME key. Retiring it here would
                    # burn the whole pool on a few seconds of burst traffic.
                    wait = min(_retry_after_seconds(err) or backoff, MAX_BACKOFF)
                    logger.info(
                        "Gemini per-minute throttle for %r (attempt %d/%d). Waiting %.1fs.",
                        skill_name, attempt, MAX_ATTEMPTS, wait,
                    )
                    time.sleep(wait)
                    backoff = min(backoff * 2, MAX_BACKOFF)
                    continue

                # Non-429 retryable: 500/503 and friends. Unchanged doubling backoff.
                if attempt == MAX_ATTEMPTS:
                    logger.error(
                        "Gemini still failing for %r after %d attempts. Giving up; "
                        "this skill is unaudited, not judged uncredible.",
                        skill_name, MAX_ATTEMPTS,
                    )
                    raise

                wait = min(_retry_after_seconds(err) or backoff, MAX_BACKOFF)
                logger.warning(
                    "Gemini %s for %r (attempt %d/%d). Waiting %.1fs.",
                    getattr(err, "status", "error"), skill_name, attempt, MAX_ATTEMPTS, wait,
                )
                time.sleep(wait)
                backoff = min(backoff * 2, MAX_BACKOFF)

        # Every attempt was consumed by throttling. Unaudited, not a verdict.
        logger.error(
            "Gemini exhausted %d attempts for %r without a response.",
            MAX_ATTEMPTS, skill_name,
        )
        raise AuditUnavailable(skill_name)

    def evaluate_candidates(
        self,
        skill_name: str,
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Audits candidate sources and returns a credibility verdict plus clean summary."""
        if not candidates:
            return dict(UNAUDITED_RESULT)

        formatted_sources = ""
        for idx, candidate in enumerate(candidates, 1):
            name = candidate.get("source_name", f"Source_{idx}")
            text = str(candidate.get("raw_text", "")).strip()
            formatted_sources += f"\n--- Source {idx}: {name} ---\n{text}\n"

        system_instruction = (
            "You are an expert technical source auditor. "
            "Review provided candidate sources and identify the single most authoritative, "
            "accurate, and non-promotional source for the target skill."
        )

        prompt = f"""
Target Skill: {skill_name}

Candidate Sources to Audit:
{formatted_sources}

Instructions:
1. Review all provided candidate sources above.
2. Identify the single most authoritative, accurate, and non-promotional source for "{skill_name}".
3. If a credible source exists, set `is_credible` to True, set `best_source_name` to its name, assign a confidence score, and write a 1-2 sentence `clean_summary`.
4. If all sources are irrelevant, promotional junk, or spam, set `is_credible` to False, set `best_source_name` to "None", and `clean_summary` to null.
"""

        try:
            response = self._generate_with_backoff(prompt, system_instruction, skill_name)

            # A truncated response yields invalid JSON, which would otherwise look
            # identical to a genuine "not credible" verdict. Surface it explicitly.
            for candidate_response in (response.candidates or []):
                if str(getattr(candidate_response, "finish_reason", "")).endswith("MAX_TOKENS"):
                    logger.error(
                        "Gemini response for %r hit max_output_tokens and is truncated. "
                        "Treating as unaudited; raise max_output_tokens.",
                        skill_name,
                    )
                    return dict(UNAUDITED_RESULT)

            if response.text:
                return json.loads(response.text)

            logger.warning("Gemini returned an empty response for %r.", skill_name)

        except DailyQuotaExhausted:
            # MUST propagate untouched, and must be caught before the bare Exception
            # clause below. This is not a per-skill failure: every key in the pool is
            # spent, so the whole run has to stop and record its backlog. Collapsing it
            # into an unaudited result would queue this skill for human review and then
            # do the same for every remaining skill in the run -- which is exactly the
            # hundreds-of-useless-cards outcome this work exists to prevent.
            raise
        except genai_errors.APIError:
            # Deliberately NOT swallowed into an unaudited verdict. The caller must be
            # able to tell "the audit could not run" from "the audit ran and rejected
            # the source" -- collapsing them would file an outage as a source
            # credibility failure. AuditUnavailable is raised for the caller to handle.
            raise AuditUnavailable(skill_name) from None
        except AuditUnavailable:
            raise
        except json.JSONDecodeError:
            logger.exception("Gemini returned unparseable JSON for %r.", skill_name)
        except Exception:
            logger.exception("Gemini audit call failed for %r.", skill_name)

        return dict(UNAUDITED_RESULT)

    # ----------------------------------------------------------------------
    # Second-pass calls
    #
    # These live on the checker, rather than in second_pass.py with their own
    # genai.Client, so that every Gemini request the process makes shares one
    # GeminiKeyPool. Two pools would keep two views of gemini_key_state.json and
    # one would keep hammering a key the other had already retired.
    # ----------------------------------------------------------------------

    def _call_or_none(
        self,
        *,
        prompt: str,
        system_instruction: str,
        skill_name: str,
        response_schema: type,
        max_output_tokens: int,
        what: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Runs one second-pass call and returns the parsed object, or None.

        None means "no usable answer" and nothing more. That is why this does NOT
        reuse evaluate_candidates' truncation handling, which returns UNAUDITED_RESULT:
        a half-parsed proposal must never become a title the pipeline then fetches.

        DailyQuotaExhausted propagates so the run stops cleanly; APIError becomes
        AuditUnavailable so the caller can record a retryable attempt rather than a
        verdict. Both contracts match evaluate_candidates exactly.
        """
        try:
            response = self._generate_with_backoff(
                prompt,
                system_instruction,
                skill_name,
                response_schema=response_schema,
                max_output_tokens=max_output_tokens,
            )

            for candidate_response in (response.candidates or []):
                if str(getattr(candidate_response, "finish_reason", "")).endswith("MAX_TOKENS"):
                    logger.error(
                        "Gemini %s for %r hit max_output_tokens. Discarding it; a "
                        "truncated answer is not a partial answer.",
                        what, skill_name,
                    )
                    return None

            if response.text:
                return json.loads(response.text)

            logger.warning("Gemini returned an empty %s for %r.", what, skill_name)

        except DailyQuotaExhausted:
            raise
        except genai_errors.APIError:
            raise AuditUnavailable(skill_name) from None
        except AuditUnavailable:
            raise
        except json.JSONDecodeError:
            logger.exception("Gemini returned unparseable JSON for %s of %r.", what, skill_name)
        except Exception:
            logger.exception("Gemini %s call failed for %r.", what, skill_name)

        return None

    def propose_reference_page(
        self,
        *,
        skill_name: str,
        category: str,
        occupation_titles: Optional[List[str]] = None,
        current_title: Optional[str] = None,
        current_extract_head: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Asks which Wikipedia article a skill name actually denotes.

        This is the question the pipeline never asks. Its resolver guesses lexically
        and its auditor grades whatever the guess produced, so an O*NET name like
        "Oracle Essbase" lands on the Oracle Database article, scores 0.98 because the
        page really is about Oracle, and is then failed by an auditor that can only
        say "not credible" without saying what would have been.

        skill_name is passed VERBATIM, vendor prefix and all. The prefix is the signal:
        "Balsamiq Studios Balsamiq Mockups" is only resolvable because it names its
        vendor. Cleaning it is what lost the entity in the first place.

        Returns the parsed ReferenceProposal, or None if no usable answer came back.
        """
        titles = [t for t in (occupation_titles or []) if t][:3]
        occupation_line = ", ".join(titles) if titles else "not recorded"

        if current_title:
            head = (current_extract_head or "").strip()[:300]
            current_block = f"""
The pipeline currently has this skill pointed at the article "{current_title}", WHICH MAY BE WRONG.
Its opening text reads: {head or "(no text on file)"}

If "{current_title}" is in fact the correct article for this skill, set same_as_current to true.
"""
        else:
            current_block = "\nThe pipeline found no article at all for this skill.\n"

        system_instruction = (
            "You are an expert at mapping vendor product names from occupational "
            "taxonomies onto English Wikipedia articles. You are precise about "
            "disambiguation and you readily admit when no article exists."
        )

        prompt = f"""
Skill name, exactly as it appears in the O*NET taxonomy: {skill_name}
O*NET category: {category or "unspecified"}
Occupations that list this skill: {occupation_line}
{current_block}
Instructions:
1. Decide which English Wikipedia article, if any, is about this specific product or technology.
2. Many of these names are discontinued or niche vendor products with no article of their own. If that is the case here, set article_exists to false and proposed_title to null. Do NOT substitute an article about the vendor, a competitor, or the general category. A wrong article is worse than none.
3. If an article does exist, set article_exists to true and give its EXACT title, including any disambiguation parenthetical, for example "Pascal (programming language)".
4. Give a confidence score and a rationale of at most one short sentence.
"""

        return self._call_or_none(
            prompt=prompt,
            system_instruction=system_instruction,
            skill_name=skill_name,
            response_schema=ReferenceProposal,
            max_output_tokens=192,
            what="reference proposal",
        )

    def draft_definition(
        self,
        *,
        skill_name: str,
        category: str,
        occupation_titles: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Writes a definition for a product that has no encyclopedia article.

        The last resort, and the only place in the pipeline where definition text has
        no external source behind it. Callers MUST record that provenance and must
        never let the result auto-approve: the whole point of the audit layer is that
        model assertions get corroborated, and there is nothing here to corroborate.

        Returns the parsed DefinitionDraft, or None.
        """
        titles = [t for t in (occupation_titles or []) if t][:3]
        occupation_line = ", ".join(titles) if titles else "not recorded"

        system_instruction = (
            "You are a technical writer documenting enterprise software products. "
            "You write plainly and you say so when you do not know a product."
        )

        prompt = f"""
Product name, exactly as it appears in the O*NET taxonomy: {skill_name}
O*NET category: {category or "unspecified"}
Occupations that list it: {occupation_line}

No Wikipedia article covers this product, so there is no source to quote.

Instructions:
1. If you genuinely know this product, write a factual 1-2 sentence definition: what it is, who makes it, what it is used for.
2. If you do not know it, or you would have to guess at what it does, set definition to null. A guess here is worse than nothing, because a human will read it as researched.
3. Do not speculate about features, versions, or market position. Do not write marketing language.
4. Set is_software_product to false if the name does not appear to denote a real software product or technology at all.
"""

        return self._call_or_none(
            prompt=prompt,
            system_instruction=system_instruction,
            skill_name=skill_name,
            response_schema=DefinitionDraft,
            max_output_tokens=192,
            what="definition draft",
        )


_source_checker: Optional[AgenticSourceChecker] = None


def get_source_checker() -> AgenticSourceChecker:
    """
    Lazily constructs the Gemini client and its key pool.

    Deliberately not instantiated at import time: a missing GEMINI_API_KEY used to
    break the scraping.py import chain and take down the whole web app, even though
    the audit layer is only reached for candidates that already passed the gate.

    Cached for the process, so key rotation persists across skills within one run --
    rebuilding the pool per call would restart at the .env key and re-probe a key
    already known to be spent, wasting a request and up to 60 seconds each time.
    """
    global _source_checker
    if _source_checker is None:
        _source_checker = AgenticSourceChecker()
    return _source_checker


def reset_source_checker() -> None:
    """Drops the cached checker so the next call rebuilds the pool. For tests."""
    global _source_checker
    _source_checker = None
