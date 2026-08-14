"""
Credibility audit layer.

Runs AFTER the cross-encoder has already established that a candidate page is about
the right subject. Its job is narrower than it looks: judge whether that correct page
is authoritative and non-promotional, and distill it into a clean 1-2 sentence
definition for embedding. It scores nothing and scrapes nothing.

Uses gemini-flash-lite-latest with a structured Pydantic response schema, sized to run
inside free-tier quota.

Calls are BATCHED where the caller can supply several skills at once. The free tier is
metered per request, not per token, so packing BATCH_SIZE skills into one call is a
direct multiplier on how much of the taxonomy fits in a day. Three rules keep that from
costing accuracy:

  1. Every item carries its own source text and its own item_id, and results are mapped
     back by that id, never by position. A response that drops, reorders or invents an
     item is detected rather than silently misattributed.
  2. The prompt tells the model to judge each item only against its own text. Batching is
     a transport decision; it must not turn N independent verdicts into one consensus.
  3. A batch that comes back short is SALVAGED, not discarded: whatever parsed is kept,
     and the missing ids are re-asked in a smaller batch, down to one call each. Under
     truncation the old all-or-nothing handling would have thrown away N verdicts where
     it used to throw away one.

What is NOT batched: the propose and audit calls for the SAME skill. Those are two
independent opinions by construction, and merging them would make the auto-approval bar
one opinion agreeing with itself. Batching across different skills does not touch that.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, Field

from embedding_probe import format_results
from gemini_keys import (
    DEFAULT_ROLE,
    GEMINI_ROLES,
    ROLE_AUDIT,
    ROLE_DRAFT,
    ROLE_EMBEDDING,
    ROLE_FLAGSHIP,
    ROLE_PROPOSAL,
    ROLE_VERIFY,
    DailyQuotaExhausted,
    GeminiKeyPool,
    describe_role,
)

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

# How many skills go into one batched call. Six is a deliberate middle: it is a 6x cut in
# requests, each item still carries its whole intro extract, and the output budget below
# leaves room to spare so a normal batch never truncates. Larger batches would need the
# inputs trimmed harder and split the model's attention further per item.
#
# The audit and the proposal are tuned separately because they are not the same shape:
# an audit carries a page of source text per item and writes a summary back, a proposal
# carries a product name and writes a title. Both are env-overridable so backing off
# after a bad measurement is a restart, not an edit. Set them to 1 to unbatch entirely.
#
# Measured on 2026-08-10 at 6: audit-failure rate 11.8% against 10.3% unbatched, zero
# misattributed summaries in 368, zero truncations. Re-measure with audit_ab.py before
# moving these.
def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, "").strip() or default))
    except ValueError:
        logger.warning("%s is not an integer; using %d.", name, default)
        return default


AUDIT_BATCH_SIZE = _int_env("GEMINI_AUDIT_BATCH_SIZE", 6)
PROPOSAL_BATCH_SIZE = _int_env("GEMINI_PROPOSAL_BATCH_SIZE", 6)
FLAGSHIP_BATCH_SIZE = _int_env("GEMINI_FLAGSHIP_BATCH_SIZE", 6)
# Six here is bounded by INPUT rather than output: each item carries up to eight search
# results plus a definition, so a batch of six is already the largest prompt any of these
# calls sends. Gemini requests are not the constraint on this pass anyway -- the 25-second
# DuckDuckGo cadence is, by a factor of about a hundred.
EMBEDDING_BATCH_SIZE = _int_env("GEMINI_EMBEDDING_BATCH_SIZE", 6)

# The second pass runs both legs over the SAME chunk of skills, so its chunk has to be
# the smaller of the two budgets. Ingestion only audits, and uses AUDIT_BATCH_SIZE.
BATCH_SIZE = min(AUDIT_BATCH_SIZE, PROPOSAL_BATCH_SIZE)

# Output budget for a batched call, per item plus fixed overhead. The single-call budgets
# (256 for an audit, 192 for a proposal) were flat; a batch has to scale or it truncates
# by construction.
AUDIT_TOKENS_PER_ITEM = _int_env("GEMINI_AUDIT_TOKENS_PER_ITEM", 130)
# Larger than the audit's, and larger than a title needs, because a proposal may come
# back carrying a written definition instead of a title. Only the unsure items spend it.
#
# Raised 200 -> 280 with DEFINITION_SPEC. A 400-character definition is roughly 100
# tokens against the ~38 the old "1-2 sentences" produced, and a budget that does not
# move with the spec would truncate exactly the items that took the instruction
# seriously. _salvage_objects would recover the rest of the batch, but the item whose
# definition was cut is the one that needed it.
PROPOSAL_TOKENS_PER_ITEM = _int_env("GEMINI_PROPOSAL_TOKENS_PER_ITEM", 280)
# The single-skill drafting call, which returns a definition and nothing else.
DRAFT_MAX_TOKENS = _int_env("GEMINI_DRAFT_MAX_TOKENS", 256)
# A verdict is three booleans, a one-sentence problem and a confidence, so it is the
# cheapest of the batched calls. Sized above the audit's anyway: an item that fails wants
# room to say why, and a truncated explanation is the one part of the answer a reviewer
# actually reads.
VERIFY_TOKENS_PER_ITEM = _int_env("GEMINI_VERIFY_TOKENS_PER_ITEM", 110)
# A flagship answer is a version name plus one or two sentences, so it sits between the
# two: larger than an audit verdict, smaller than a proposal that may carry a definition.
FLAGSHIP_TOKENS_PER_ITEM = _int_env("GEMINI_FLAGSHIP_TOKENS_PER_ITEM", 160)
# A boolean, a one-sentence justification, a copied URL and a confidence. The URL is why
# this sits above the verify budget rather than below it: a truncated evidence_url is a
# broken link on the dashboard, and the sentence explaining a FALSE is the part a reviewer
# reads when they disagree with the verdict.
EMBEDDING_TOKENS_PER_ITEM = _int_env("GEMINI_EMBEDDING_TOKENS_PER_ITEM", 140)
BATCH_TOKENS_OVERHEAD = 64

# Wikipedia intro extracts were previously sent whole and unbounded, which was affordable
# at one skill per call and is not at six. 1500 characters is past the end of a normal
# intro paragraph, so the audit still sees the definition it is grading. Raise it if a
# measurement ever shows the audit missing something that was past the cut.
MAX_SOURCE_CHARS = _int_env("GEMINI_MAX_SOURCE_CHARS", 1500)

# --- What a written definition has to look like ---------------------------------
#
# Measured on the live store, comparing the two kinds of definition it holds:
#
#     Wikipedia-derived   n=481   median 191 chars   p75 238
#     Model-authored      n=62    median 150 chars   p75 171
#
# The model's are consistently thinner, and the gap is not only length. A Wikipedia
# lead names the technical category, the vendor and its lineage, and what the thing is
# actually used for:
#
#     "SAP IQ is a column-based, petabyte-scale relational database software system
#      developed by Sybase and later SAP, designed for business intelligence, data
#      warehousing, and ad-hoc data analysis."
#
# while the model states what it is, who makes it and who it is for, and then stops:
#
#     "DynaSCAPE Design is a computer-aided design software developed by DynaSCAPE
#      Software specifically for landscape architects and professional designers."
#
# This matters beyond readability: the definition is the text the bi-encoder scores, so
# a thin one measures thin. "Column-based relational database, business intelligence,
# data warehousing" is signal against the tech-base and ML-pipeline anchors that
# "computer-aided design software for landscape architects" simply does not carry.
#
# Stated once and shared by all three prompts that ask for a definition, so they cannot
# drift apart.
DEFINITION_SPEC = (
    "START WITH THE PRODUCT'S NAME AS THE SUBJECT, exactly as an encyclopedia lead does: "
    "'GeoPak Bridge is a civil engineering add-on...', never 'Civil engineering add-on "
    "developed by...'. A definition that opens with its category reads as a fragment on "
    "the card and loses the one word a reader is scanning for. "
    "Write 2 to 3 full sentences, about 200 to 400 characters. Match the depth of an "
    "encyclopedia opening paragraph, and cover in this order: (1) what the product IS, "
    "naming its specific technical category rather than a generic one -- 'column-oriented "
    "relational database', not 'database software'; (2) who develops it, including the "
    "original vendor if it has changed hands; (3) what it is actually USED FOR, naming "
    "the concrete tasks, workflows or industries, not just the audience. Name the "
    "platforms, file formats, protocols, languages or standards it works with when you "
    "know them. Write plain declarative prose with no marketing adjectives, no hedging "
    "and no 'is a software tool that allows users to'. State only what you are confident "
    "is true: a shorter definition is better than an invented detail. "
    "IF THE NAME IS A CATEGORY rather than one product -- 'Tariff databases', "
    "'Voice-activated perio charting software', 'Plant information data entry software' "
    "-- define the CLASS: what this kind of software does, the tasks and workflows it "
    "supports, and the kind of organisation that uses it. Do not attach a vendor to a "
    "category, and do not silently describe one market-leading product as though it were "
    "the whole class. Naming a few representative products at the end is useful; "
    "presenting one of them as the definition is not."
)


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

    It ALSO carries a definition, which is not a second question but the same one
    answered a different way: "what is this product?". A title the model is unsure of
    costs a fetch, a rescore and an audit call to end up arguing for an article nobody
    stood behind, so below the caller's confidence bar the definition is the answer that
    gets used and the page is never chased. Asking for it here rather than in a second
    request is the whole saving.
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
    definition: Optional[str] = Field(
        default=None,
        description=(
            "A factual definition of the product, written ONLY when article_exists is "
            "false or confidence is below the stated bar. Null otherwise, and null when "
            "the product is genuinely unknown to you. " + DEFINITION_SPEC
        ),
    )
    is_software_product: bool = Field(
        default=True,
        description="False if the name does not appear to denote a real software product or technology.",
    )


class DefinitionDraft(BaseModel):
    """
    A definition written from the taxonomy alone, for products with no article.

    The output of this has NO external source behind it, which is why callers must
    mark it as such and must never let it auto-approve.
    """

    definition: Optional[str] = Field(
        default=None,
        description=(
            "A factual definition, or null if the product is unknown to you. "
            + DEFINITION_SPEC
        ),
    )
    is_software_product: bool = Field(
        default=True,
        description="False if the name does not appear to denote a real software product or technology.",
    )


class BatchedEvaluation(MultiSourceEvaluation):
    """One audit verdict inside a batch, tagged with the id it answers."""

    item_id: str = Field(description="The exact item_id of the task this evaluation answers.")


class BatchedEvaluations(BaseModel):
    evaluations: List[BatchedEvaluation] = Field(
        description="Exactly one evaluation per task, in any order."
    )


class BatchedProposal(ReferenceProposal):
    """One reference proposal inside a batch, tagged with the id it answers."""

    item_id: str = Field(description="The exact item_id of the task this proposal answers.")


class BatchedProposals(BaseModel):
    proposals: List[BatchedProposal] = Field(
        description="Exactly one proposal per task, in any order."
    )


class FlagshipDescription(BaseModel):
    """
    What the flagship deployment of a product embeds, for scoring purposes only.

    O*NET names generic categories ("Word processing software") and enterprise suites
    whose encyclopedia definitions predate the AI features now shipped in them. Scoring
    that text measures the product as it was, not as it is deployed.

    Two distinct answers come back here, and they are used differently:

      - `is_generic_category` decides WHICH product is measured. A category name is
        scored as its flagship outright, because a category article outscores every
        product in its category: "Word processing software" reads ai 0.321 against
        Microsoft Word's 0.192, and was classified an AI Skill on the strength of
        abstract taxonomy prose. Substituting the flagship makes the category equal to
        its flagship by construction, so it can never sit above it.
      - `embedded_ai_summary` decides HOW MUCH embedded AI that product carries, and
        feeds embedded_ai_sim only.

    `is_flagship_version_evaluated` is the honest-negative escape: most software has no
    dominant AI-bearing version, and inflating a legacy tool would be worse than leaving
    it where it is.
    """

    is_generic_category: bool = Field(
        description=(
            "True if this name denotes a CATEGORY of software rather than one specific "
            "product, e.g. 'Word processing software', 'Spreadsheet software', "
            "'Statistical software'. False for a named product like 'Adobe Photoshop', "
            "'Slack' or 'Microsoft Teams', even when that product is very widely used."
        )
    )
    is_flagship_version_evaluated: bool = Field(
        description=(
            "True only if a dominant, widely deployed version of this product ships "
            "native AI features. False for legacy, niche or discontinued software, and "
            "false when you are not sure."
        )
    )
    flagship_version: Optional[str] = Field(
        default=None,
        description=(
            "The single market-leading product this term denotes in practice, named as "
            "plainly as possible so it can be matched against a product list: "
            "'Microsoft Word' for 'Word processing software', not 'Microsoft Word in "
            "Microsoft 365'. Null when you cannot name one."
        ),
    )
    flagship_definition: Optional[str] = Field(
        default=None,
        description=(
            "One or two factual sentences defining the flagship product itself: what it "
            "is, who makes it, what it is used for. Required whenever is_generic_category "
            "is true, because this text may be what the category is scored on. Do not "
            "describe its AI features here; that belongs in embedded_ai_summary."
        ),
    )
    embedded_ai_summary: Optional[str] = Field(
        default=None,
        description=(
            "One or two sentences naming the AI features that version embeds and what "
            "they do, e.g. 'Microsoft 365 Copilot in Word drafts and rewrites documents "
            "from prompts and summarizes long files.' Null when "
            "is_flagship_version_evaluated is false. State only features that actually "
            "shipped; do not describe announcements or roadmaps."
        ),
    )
    confidence_score: float = Field(
        description="0.0 to 1.0, how sure you are that these features shipped in this product."
    )


class DefinitionVerdict(BaseModel):
    """
    A SECOND opinion on a definition the model wrote from its own knowledge.

    Until now a written definition had no second opinion at all, which is why accepting
    one required a human click. That click was granted 139 times out of 139, so the gate
    was a rubber stamp on the largest bucket in the queue.

    This is the same bar a Wikipedia page already clears: one call chooses the answer, a
    DIFFERENT call grades it, and agreement between them is what permits auto-approval.
    The verifier is deliberately shown the definition and nothing else about how it was
    produced -- no rationale, no confidence score -- so it cannot agree with the writer's
    reasoning instead of checking the claim.
    """

    describes_this_skill: bool = Field(
        description=(
            "True if the definition is about THIS product or category, not a "
            "similarly-named one. A definition that is accurate and well written but "
            "describes a different product is false: sharing a word with the name is "
            "not enough."
        )
    )
    contains_invented_specifics: bool = Field(
        description=(
            "True if the definition asserts specifics that read as fabricated: version "
            "numbers, release dates, customer names, market-share claims, or feature "
            "lists too precise to be general knowledge. This is the characteristic "
            "failure of a definition written without a source."
        )
    )
    is_generic_category: bool = Field(
        description=(
            "True if the name denotes a CLASS of software rather than one product, e.g. "
            "'Tariff databases'. Such a definition is expected to describe the class and "
            "must not be marked wrong for naming no vendor."
        )
    )
    problem: Optional[str] = Field(
        default=None,
        description="One short sentence naming the problem, or null if there is none.",
    )
    confidence: float = Field(
        description="0.0 to 1.0, how sure you are of this judgement."
    )


class BatchedDefinitionVerdict(DefinitionVerdict):
    """One verdict inside a batch, tagged with the id it answers."""

    item_id: str = Field(description="The exact item_id of the task this verdict answers.")


class BatchedDefinitionVerdicts(BaseModel):
    verdicts: List[BatchedDefinitionVerdict] = Field(
        description="Exactly one verdict per task, in any order."
    )


class BatchedFlagship(FlagshipDescription):
    """One flagship description inside a batch, tagged with the id it answers."""

    item_id: str = Field(description="The exact item_id of the task this description answers.")


class BatchedFlagshipDescriptions(BaseModel):
    descriptions: List[BatchedFlagship] = Field(
        description="Exactly one description per task, in any order."
    )


class EmbeddingVerdict(BaseModel):
    """
    Whether a product itself ships AI features, judged from search results.

    This replaces a cosine similarity that could not tell the difference. embedded_ai_sim
    put LaTeX, Thomson EndNote, SofTech CADRA and Transoft AutoTURN in a class called
    "Non-Technical Embedded AI" purely on the vocabulary of their definitions.

    THE FAILURE THIS SCHEMA EXISTS TO CATCH is not a wrong search, it is a right search
    read wrongly. Searching "does LaTeX embed AI" returns Overleaf, Prism and Underleaf --
    third-party editors built AROUND LaTeX, every one of them arguing yes. The correct
    answer is no. So the judgement being asked for is narrow and stated as such: does the
    PRODUCT NAMED ship the feature, in the product, today.
    """

    embeds_ai: bool = Field(
        description=(
            "True only if the named product ITSELF ships AI features that are generally "
            "available today. False for a product whose AI comes from a separate tool, "
            "an add-on by another vendor, or an integration; false for legacy, "
            "discontinued and purely technical software; and false when the search "
            "results do not actually establish it."
        )
    )
    evidence: str = Field(
        description=(
            "One sentence naming the specific AI feature and what it does, e.g. "
            "'Lightroom ships Generative Upscale and AI Sharpen for image enhancement.' "
            "When embeds_ai is false, state in one sentence why the results do not "
            "establish it, e.g. 'The results describe third-party editors built around "
            "LaTeX, not features of LaTeX itself.'"
        )
    )
    evidence_url: Optional[str] = Field(
        default=None,
        description=(
            "The single result URL that best supports the verdict, copied exactly from "
            "the results given. Null if none of them do. Never invent a URL."
        ),
    )
    confidence: float = Field(
        description=(
            "0.0 to 1.0, how sure you are. A low score costs nothing and is the right "
            "answer when the results are thin or off-topic."
        )
    )


class BatchedEmbeddingVerdict(EmbeddingVerdict):
    """One embedding verdict inside a batch, tagged with the id it answers."""

    item_id: str = Field(description="The exact item_id of the task this verdict answers.")


class BatchedEmbeddingVerdicts(BaseModel):
    verdicts: List[BatchedEmbeddingVerdict] = Field(
        description="Exactly one verdict per task, in any order."
    )


def _salvage_objects(text: str) -> List[Dict[str, Any]]:
    """
    Pulls whole JSON objects out of a response that does not parse as a whole.

    A truncated batch is a valid list of results with the tail cut off mid-object. The
    complete objects before the cut are perfectly good answers, and the old handling --
    written when a response held one item -- discarded them all. This scans for balanced
    top-level braces and parses each span on its own, so what survived the cut is used
    and only the genuinely missing ids get re-asked.

    Every balanced object is collected, at whatever nesting depth, and then filtered to
    the ones carrying an item_id. The wrapper object is the thing that got truncated, so
    scanning only at the top level would find nothing at all -- the results live one
    level inside it.

    Deliberately dumb about strings containing braces: a span that does not parse is
    skipped rather than repaired. This runs on rationale and summary text scraped from
    the open web, and guessing at malformed JSON is how bad data gets in.
    """
    found = []
    starts = []
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            starts.append(index)
        elif char == "}" and starts:
            start = starts.pop()
            try:
                parsed = json.loads(text[start:index + 1])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and parsed.get("item_id"):
                found.append(parsed)

    return found


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
    """
    One Gemini client bound to one ROLE's key pool.

    A checker is per-role, not per-process. Each distinct task -- audit, proposal, draft,
    verify, flagship, embedding -- owns a .env key slot and falls back to the shared
    spares only once its own key is spent, so one task burning its daily quota cannot
    starve the others. See gemini_keys.GEMINI_ROLES.
    """

    def __init__(
        self, pool: Optional[GeminiKeyPool] = None, role: str = DEFAULT_ROLE
    ) -> None:
        self.role = role
        self.pool = pool or GeminiKeyPool(role)
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
            raise DailyQuotaExhausted(self.pool.total_keys(), self.role)
        self.client = genai.Client(api_key=key)

    def _rotate_or_give_up(self) -> None:
        """
        Retires the current key and rebuilds on the next one.

        Raises DailyQuotaExhausted when the pool is spent, which is the signal the
        ingestion loop uses to stop and write its backlog.
        """
        if not self.pool.mark_exhausted_and_advance():
            raise DailyQuotaExhausted(self.pool.total_keys(), self.role)
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
1. FIRST decide whether the source is about "{skill_name}" ITSELF. A source that is authoritative, accurate and well written, but is about a different product, company, or subject, is NOT credible for this skill. Sharing a word with the name is not enough: "Cosmo's Cosmic Adventure" is not a source for "Cosmo Software Cosmo World". Judge the subject before you judge the quality.
2. If the source is about the right subject, judge whether it is authoritative, accurate and non-promotional.
3. If it passes both, set `is_credible` to True, set `best_source_name` to its name, assign a confidence score, and write a 1-2 sentence `clean_summary`.
4. If the source is about something else, or is promotional junk or spam, set `is_credible` to False, set `best_source_name` to "None", and `clean_summary` to null. An article about the VENDOR is not a source for one of its individual products, and an article about the general category is not a source for a specific product in it.
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
    # Batched calls
    #
    # Same engine, same pool, same 429 handling as everything else here. What differs
    # is only how many skills ride in one request, and the mapping back by item_id.
    # ----------------------------------------------------------------------

    def _batch_call(
        self,
        *,
        prompt: str,
        system_instruction: str,
        label: str,
        response_schema: type,
        container: str,
        max_output_tokens: int,
    ) -> List[Dict[str, Any]]:
        """
        Runs one batched call and returns whatever items parsed. Never raises on shape.

        Returns a possibly SHORT list. Callers must reconcile against the ids they sent
        rather than assuming a full answer -- that reconciliation is what turns a
        truncated or malformed response into a small re-ask instead of N lost verdicts.

        DailyQuotaExhausted and AuditUnavailable propagate exactly as they do for a
        single call: the first stops the run, the second is per-batch and retryable.
        """
        response = self._generate_with_backoff(
            prompt,
            system_instruction,
            label,
            response_schema=response_schema,
            max_output_tokens=max_output_tokens,
        )

        truncated = any(
            str(getattr(candidate, "finish_reason", "")).endswith("MAX_TOKENS")
            for candidate in (response.candidates or [])
        )

        text = response.text or ""
        if not text:
            logger.warning("Gemini returned an empty batched %s for %s.", container, label)
            return []

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            # Expected when truncated, and the whole reason _salvage_objects exists.
            items = _salvage_objects(text)
            logger.warning(
                "Batched %s for %s did not parse whole%s. Salvaged %d complete item(s); "
                "the rest will be re-asked.",
                container, label, " (truncated)" if truncated else "", len(items),
            )
            return items

        if isinstance(payload, dict):
            items = payload.get(container) or []
        elif isinstance(payload, list):
            items = payload
        else:
            items = []

        parsed = [item for item in items if isinstance(item, dict)]
        if truncated:
            logger.warning(
                "Batched %s for %s hit max_output_tokens but still parsed %d item(s). "
                "Any missing ones will be re-asked.",
                container, label, len(parsed),
            )
        return parsed

    def evaluate_candidates_batch(
        self, items: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Audits several skills in one call. Returns {item_id: evaluation}.

        Each item is {"item_id", "skill_name", "source_name", "raw_text"}. The returned
        evaluations have the same shape evaluate_candidates returns, so callers map them
        onto entries with the code they already have.

        An id missing from the answer is re-asked in a smaller batch, halving down to
        single calls. An id that never comes back is simply absent from the result, which
        the caller must treat as UNAUDITED -- never as "not credible". A batch that fails
        to produce anything at all is not a verdict on any of its skills.
        """
        results: Dict[str, Dict[str, Any]] = {}
        wanted = [item for item in items if item.get("item_id")]
        if not wanted:
            return results

        if len(wanted) == 1:
            # One item is not a batch. Route it through the single-call path so there is
            # exactly one implementation of the one-skill audit.
            item = wanted[0]
            results[item["item_id"]] = self.evaluate_candidates(
                skill_name=item.get("skill_name") or item["item_id"],
                candidates=[{
                    "source_name": item.get("source_name") or "Wikipedia",
                    "raw_text": item.get("raw_text") or "",
                }],
            )
            return results

        label = f"{len(wanted)} skills ({wanted[0].get('skill_name')} and others)"

        blocks = ""
        for item in wanted:
            text = str(item.get("raw_text") or "").strip()[:MAX_SOURCE_CHARS]
            blocks += (
                f"\n--- TASK item_id={item['item_id']} ---\n"
                f"Target skill: {item.get('skill_name') or ''}\n"
                f"Source name: {item.get('source_name') or 'Wikipedia'}\n"
                f"Source text:\n{text}\n"
                f"--- END TASK item_id={item['item_id']} ---\n"
            )

        system_instruction = (
            "You are an expert technical source auditor. "
            "Review provided candidate sources and identify the single most authoritative, "
            "accurate, and non-promotional source for the target skill. "
            "You are given several unrelated audit tasks in one request and you judge each "
            "one strictly on its own source text."
        )

        prompt = f"""
You are given {len(wanted)} INDEPENDENT audit tasks.

{blocks}

Instructions:
1. Return exactly one evaluation object per task, echoing that task's item_id verbatim. Do not merge, skip, or invent tasks.
2. For each task, FIRST decide whether the source is about that task's target skill ITSELF. A source that is authoritative, accurate and well written, but is about a different product, company, or subject, is NOT credible. Sharing a word with the name is not enough: "Cosmo's Cosmic Adventure" is not a source for "Cosmo Software Cosmo World". Judge the subject before you judge the quality.
3. If the source is about the right subject AND is authoritative, accurate and non-promotional, set `is_credible` to True, set `best_source_name` to the source name, assign a confidence score, and write a 1-2 sentence `clean_summary` of that skill.
4. If a task's source is about something else, or is promotional junk or spam, set `is_credible` to False, `best_source_name` to "None", and `clean_summary` to null. An article about the VENDOR is not a source for one of its individual products, and an article about the general category is not a source for a specific product in it.
5. Judge each task ONLY against its own source text. One task's verdict must not influence another's; these products are unrelated to each other.
6. Keep every clean_summary to at most two sentences.
"""

        parsed = self._batch_call(
            prompt=prompt,
            system_instruction=system_instruction,
            label=label,
            response_schema=BatchedEvaluations,
            container="evaluations",
            max_output_tokens=AUDIT_TOKENS_PER_ITEM * len(wanted) + BATCH_TOKENS_OVERHEAD,
        )

        expected = {item["item_id"] for item in wanted}
        for evaluation in parsed:
            item_id = str(evaluation.get("item_id") or "")
            if item_id not in expected or item_id in results:
                # An id we did not send, or a second answer for one we did. Both are
                # signs of a confused response, and neither can be attributed safely.
                logger.warning("Batched audit returned an unusable item_id %r.", item_id)
                continue
            verdict = {key: value for key, value in evaluation.items() if key != "item_id"}
            results[item_id] = verdict

        missing = [item for item in wanted if item["item_id"] not in results]
        if not missing:
            return results

        if len(missing) == len(wanted):
            # Nothing came back at all, so re-asking the same set the same way is not a
            # salvage, it is a repeat. Split it up instead: single calls have the whole
            # token budget to themselves and cannot be truncated by a neighbour.
            logger.warning(
                "Batched audit of %d skill(s) returned nothing usable. Falling back to "
                "one call each.", len(wanted),
            )
            for item in wanted:
                try:
                    results.update(self.evaluate_candidates_batch([item]))
                except AuditUnavailable:
                    # Per-item outage. Leave the id absent -- unaudited, not judged --
                    # and let the remaining skills in this batch still be audited.
                    logger.warning(
                        "Audit unavailable for %r during batch fallback.",
                        item.get("skill_name") or item["item_id"],
                    )
            return results

        logger.info(
            "Batched audit answered %d of %d; re-asking the remaining %d.",
            len(results), len(wanted), len(missing),
        )
        results.update(self.evaluate_candidates_batch(missing))
        return results

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
        confidence_bar: float = 0.9,
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
            "disambiguation and you readily admit when no article exists. When you "
            "cannot name the right article confidently, you say what the product is "
            "instead of guessing at a page."
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
4. Give an honest confidence score and a rationale of at most one short sentence. Do not inflate confidence: a low score costs nothing and a wrong article costs a person's time.
5. WRITE A DEFINITION INSTEAD OF GUESSING. If article_exists is false, or your confidence is below {confidence_bar:.2f}, write the product's definition in `definition`. That definition is what will be used, and no page will be fetched, so a guessed title below the bar helps nobody. Still fill in proposed_title with your best guess when you have one, so a human can check it.\n   {DEFINITION_SPEC}
6. Leave `definition` null when your confidence is at or above {confidence_bar:.2f} and an article exists. Leave it null if the product is genuinely unknown to you rather than writing something you would have to invent.
7. Length is a target, not a quota. If you know only enough for one solid sentence, write one solid sentence rather than padding it to reach the range.
7. Set is_software_product to false if the name does not denote a real software product or technology at all.
"""

        return self._call_or_none(
            prompt=prompt,
            system_instruction=system_instruction,
            skill_name=skill_name,
            response_schema=ReferenceProposal,
            max_output_tokens=PROPOSAL_TOKENS_PER_ITEM + BATCH_TOKENS_OVERHEAD,
            what="reference proposal",
        )

    def propose_reference_pages_batch(
        self, items: List[Dict[str, Any]], *, confidence_bar: float = 0.9
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Asks which article each of several skills denotes, in one call.

        Each item is {"item_id", "skill_name", "category", "occupation_titles",
        "current_title", "current_extract_head"}. Returns {item_id: proposal}, with the
        same shape propose_reference_page returns for one skill, plus `definition` and
        `is_software_product`.

        `confidence_bar` is stated to the model, and is the caller's threshold: below it
        the caller will not chase the page, so a definition is the more useful answer and
        this is the only chance to get one without paying for another request.

        An id missing from the answer is absent from the result, which callers must read
        as "no usable answer" -- the same meaning propose_reference_page's None has. A
        half-parsed proposal must never become a title the pipeline then fetches, which
        is why nothing here reconstructs a partial object.
        """
        results: Dict[str, Optional[Dict[str, Any]]] = {}
        wanted = [item for item in items if item.get("item_id")]
        if not wanted:
            return results

        if len(wanted) == 1:
            item = wanted[0]
            results[item["item_id"]] = self.propose_reference_page(
                skill_name=item.get("skill_name") or item["item_id"],
                category=item.get("category") or "",
                occupation_titles=item.get("occupation_titles") or [],
                current_title=item.get("current_title"),
                current_extract_head=item.get("current_extract_head") or "",
                confidence_bar=confidence_bar,
            )
            return results

        label = f"{len(wanted)} skills ({wanted[0].get('skill_name')} and others)"

        blocks = ""
        for item in wanted:
            titles = [t for t in (item.get("occupation_titles") or []) if t][:3]
            current_title = item.get("current_title")
            if current_title:
                head = str(item.get("current_extract_head") or "").strip()[:300]
                current_line = (
                    f'Currently pointed at the article "{current_title}", WHICH MAY BE WRONG. '
                    f'Its opening text reads: {head or "(no text on file)"}'
                )
            else:
                current_line = "No article was found at all for this skill."

            blocks += (
                f"\n--- TASK item_id={item['item_id']} ---\n"
                f"Skill name, exactly as it appears in the O*NET taxonomy: {item.get('skill_name') or ''}\n"
                f"O*NET category: {item.get('category') or 'unspecified'}\n"
                f"Occupations that list this skill: {', '.join(titles) if titles else 'not recorded'}\n"
                f"{current_line}\n"
                f"--- END TASK item_id={item['item_id']} ---\n"
            )

        system_instruction = (
            "You are an expert at mapping vendor product names from occupational "
            "taxonomies onto English Wikipedia articles. You are precise about "
            "disambiguation and you readily admit when no article exists. You are given "
            "several unrelated products in one request and you resolve each on its own. "
            "When you cannot name the right article confidently, you say what the "
            "product is instead of guessing at a page."
        )

        prompt = f"""
You are given {len(wanted)} INDEPENDENT resolution tasks.

{blocks}

Instructions:
1. Return exactly one proposal object per task, echoing that task's item_id verbatim. Do not merge, skip, or invent tasks.
2. For each task, decide which English Wikipedia article, if any, is about that specific product or technology.
3. Many of these names are discontinued or niche vendor products with no article of their own. If that is the case, set article_exists to false and proposed_title to null. Do NOT substitute an article about the vendor, a competitor, or the general category. A wrong article is worse than none.
4. If an article does exist, set article_exists to true and give its EXACT title, including any disambiguation parenthetical, for example "Pascal (programming language)".
5. If the article a task is currently pointed at is in fact the correct one, set same_as_current to true for that task.
6. Give an honest confidence score and a rationale of at most one short sentence per task. Do not inflate confidence: a low score costs nothing and a wrong article costs a person's time.
7. WRITE A DEFINITION INSTEAD OF GUESSING. If article_exists is false, or your confidence is below {confidence_bar:.2f}, write the product's definition in `definition`. That definition is what will be used, and no page will be fetched, so a guessed title below the bar helps nobody. Still fill in proposed_title with your best guess when you have one, so a human can check it.\n   {DEFINITION_SPEC}
8. Leave `definition` null when your confidence is at or above {confidence_bar:.2f} and an article exists. Leave it null if the product is genuinely unknown to you rather than writing something you would have to invent.
9. Length is a target, not a quota. If you know only enough for one solid sentence, write one solid sentence rather than padding it to reach the range.
9. Set is_software_product to false if a name does not denote a real software product or technology at all.
10. Resolve each task independently. These products are unrelated to each other.
"""

        parsed = self._batch_call(
            prompt=prompt,
            system_instruction=system_instruction,
            label=label,
            response_schema=BatchedProposals,
            container="proposals",
            max_output_tokens=PROPOSAL_TOKENS_PER_ITEM * len(wanted) + BATCH_TOKENS_OVERHEAD,
        )

        expected = {item["item_id"] for item in wanted}
        for proposal in parsed:
            item_id = str(proposal.get("item_id") or "")
            if item_id not in expected or item_id in results:
                logger.warning("Batched proposal returned an unusable item_id %r.", item_id)
                continue
            results[item_id] = {
                key: value for key, value in proposal.items() if key != "item_id"
            }

        missing = [item for item in wanted if item["item_id"] not in results]
        if not missing:
            return results

        if len(missing) == len(wanted):
            logger.warning(
                "Batched proposal for %d skill(s) returned nothing usable. Falling back "
                "to one call each.", len(wanted),
            )
            for item in wanted:
                try:
                    results.update(self.propose_reference_pages_batch([item]))
                except AuditUnavailable:
                    logger.warning(
                        "Proposal unavailable for %r during batch fallback.",
                        item.get("skill_name") or item["item_id"],
                    )
            return results

        logger.info(
            "Batched proposal answered %d of %d; re-asking the remaining %d.",
            len(results), len(wanted), len(missing),
        )
        results.update(self.propose_reference_pages_batch(missing))
        return results

    def verify_definitions_batch(
        self, items: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Grades written definitions in one call. The second of the two opinions.

        Each item is {"item_id", "skill_name", "category", "occupation_titles",
        "definition"}. Returns {item_id: verdict}.

        An id missing from the answer is ABSENT from the result, and callers must read
        that as "not verified" and stage the item for a human. A missing verdict must
        never be read as a pass: the whole point of this call is that nothing sourceless
        reaches the dashboard on one opinion, and a quota wall or a truncated response is
        not agreement.

        The prompt is given the definition and NOT how it was produced. Showing the
        writer's rationale or confidence would invite the verifier to agree with the
        reasoning rather than check the claim, which is the same reason the propose and
        audit calls for a page are kept separate.
        """
        results: Dict[str, Dict[str, Any]] = {}
        wanted = [
            item for item in items
            if item.get("item_id") and (item.get("definition") or "").strip()
        ]
        if not wanted:
            return results

        label = f"{len(wanted)} definition(s) ({wanted[0].get('skill_name')} and others)"

        blocks = ""
        for item in wanted:
            titles = [t for t in (item.get("occupation_titles") or []) if t][:3]
            definition = str(item.get("definition") or "").strip()[:MAX_SOURCE_CHARS]
            blocks += (
                f"\n--- TASK item_id={item['item_id']} ---\n"
                f"Skill name, exactly as it appears in the O*NET taxonomy: {item.get('skill_name') or ''}\n"
                f"O*NET category: {item.get('category') or 'unspecified'}\n"
                f"Occupations that list this skill: {', '.join(titles) if titles else 'not recorded'}\n"
                f"Definition to grade: {definition}\n"
                f"--- END TASK item_id={item['item_id']} ---\n"
            )

        system_instruction = (
            "You are a fact-checker for a skills taxonomy. You are given definitions "
            "written from memory, with no source behind them, and your job is to catch "
            "two specific failures: a definition that describes the wrong product, and a "
            "definition that invents specifics it cannot know. You are strict and you "
            "prefer a false alarm to letting an invented claim through. You judge each "
            "definition on its own."
        )

        prompt = f"""
You are given {len(wanted)} INDEPENDENT definitions to grade.

{blocks}

Instructions:
1. Return exactly one verdict object per task, echoing that task's item_id verbatim. Do not merge, skip, or invent tasks.
2. FIRST decide whether the definition is about the named product ITSELF. A definition that is accurate, fluent and plausible but describes a DIFFERENT product is not acceptable. Sharing a word with the name is not enough: a definition of a video game is not a definition of "Cosmo Software Cosmo World" merely because both contain "Cosmo". Set describes_this_skill accordingly.
3. THEN look for invented specifics: version numbers, release years, named customers, market-share or ranking claims, pricing, or feature lists more precise than general knowledge supports. Set contains_invented_specifics true if you find any. A definition that stays general is CORRECT behaviour, not a weak answer.
4. Judge a CATEGORY name as a category. "Tariff databases" or "Voice-activated perio charting software" name a class of software, and a definition describing that class is right. Set is_generic_category true and do not mark it wrong for naming no vendor.
5. Do not penalise a definition for being short, for omitting the vendor when the vendor is genuinely unknown, or for describing a discontinued product in the past tense.
6. Do not reward fluency. A well-written definition of the wrong thing is the failure this check exists to catch.
7. When you set either flag against a definition, name the problem in one short sentence.
8. Give an honest confidence. If you do not know the product at all, say so with a low confidence rather than guessing that the definition is fine.
9. Grade each task independently. These products are unrelated to each other.
"""

        parsed = self._batch_call(
            prompt=prompt,
            system_instruction=system_instruction,
            label=label,
            response_schema=BatchedDefinitionVerdicts,
            container="verdicts",
            max_output_tokens=VERIFY_TOKENS_PER_ITEM * len(wanted) + BATCH_TOKENS_OVERHEAD,
        )

        expected = {item["item_id"] for item in wanted}
        for verdict in parsed:
            item_id = str(verdict.get("item_id") or "")
            if item_id not in expected or item_id in results:
                logger.warning("Definition verdict returned an unusable item_id %r.", item_id)
                continue
            results[item_id] = {
                key: value for key, value in verdict.items() if key != "item_id"
            }

        missing = [item for item in wanted if item["item_id"] not in results]
        if not missing:
            return results

        if len(missing) == len(wanted):
            logger.warning(
                "Verification of %d definition(s) returned nothing usable. They will be "
                "staged for a human rather than treated as verified.", len(wanted),
            )
            return results

        logger.info(
            "Verified %d of %d definition(s); re-asking the remaining %d.",
            len(results), len(wanted), len(missing),
        )
        results.update(self.verify_definitions_batch(missing))
        return results

    def describe_flagship_versions_batch(
        self, items: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Asks what the flagship deployment of each of several products embeds, in one call.

        Each item is {"item_id", "skill_name", "category", "current_definition"}. Returns
        {item_id: description}, each carrying is_flagship_version_evaluated,
        flagship_version, embedded_ai_summary and confidence_score.

        An id missing from the answer is absent from the result, which callers must read
        as "not evaluated" and leave the skill's scoring untouched. A skill this pass
        never reached must be indistinguishable from one it evaluated and declined, so
        that a quota wall cannot quietly suppress a product's embedded-AI signal.

        Reconciliation, re-asking and single-item fallback follow
        propose_reference_pages_batch exactly; the two differ only in what they ask.
        """
        results: Dict[str, Dict[str, Any]] = {}
        wanted = [item for item in items if item.get("item_id")]
        if not wanted:
            return results

        label = f"{len(wanted)} flagship lookup(s) ({wanted[0].get('skill_name')} and others)"

        blocks = ""
        for item in wanted:
            definition = str(item.get("current_definition") or "").strip()[:MAX_SOURCE_CHARS]
            blocks += (
                f"\n--- TASK item_id={item['item_id']} ---\n"
                f"Skill name, exactly as it appears in the O*NET taxonomy: {item.get('skill_name') or ''}\n"
                f"O*NET category: {item.get('category') or 'unspecified'}\n"
                f"Definition currently on file: {definition or '(none recorded)'}\n"
                f"--- END TASK item_id={item['item_id']} ---\n"
            )

        system_instruction = (
            "You are an expert on enterprise and consumer software product lines and "
            "which AI features have actually shipped in them. You distinguish sharply "
            "between features that are generally available today, features in preview, "
            "and features that were merely announced. You are given several unrelated "
            "products in one request and you assess each on its own. You are comfortable "
            "answering that a product has no notable AI features, because most software "
            "does not."
        )

        prompt = f"""
You are given {len(wanted)} INDEPENDENT product assessments.

{blocks}

Instructions:
1. Return exactly one description object per task, echoing that task's item_id verbatim. Do not merge, skip, or invent tasks.
2. FIRST decide whether the name is a CATEGORY of software or ONE SPECIFIC PRODUCT, and report it in is_generic_category. "Word processing software", "Spreadsheet software" and "Statistical software" are categories. "Adobe Photoshop", "Slack" and "Microsoft Teams" are products, however widely used they are.
3. For a CATEGORY, name in flagship_version the single market-leading product the term denotes in practice -- "Microsoft Word" for "Word processing software" -- and write flagship_definition, one or two factual sentences defining THAT product. Name it as plainly as you can, without edition or suite qualifiers, so it can be matched against a product list. Pick ONE, the most common deployment, not a list.
4. For a PRODUCT, set is_generic_category false and leave flagship_version and flagship_definition null. The product is already what it is.
5. Many of these are enterprise platforms whose definition on file predates the AI features since added to them. Judge the product as it ships TODAY, not as the definition describes it.
6. Set is_flagship_version_evaluated to true ONLY when a dominant, widely deployed version of the product ships native AI features that are generally available. Set it to false for legacy, niche, discontinued or purely technical software, and false whenever you are unsure.
7. When it is false, leave embedded_ai_summary null. Do not stretch to find an AI angle. Most of these products do not have one, and a false positive here misclassifies a skill. This is independent of is_generic_category: a category can have a clear flagship that carries no AI features at all, and you should say so.
8. When it is true, write one or two sentences in embedded_ai_summary naming the AI features and what they do for the user.
9. Describe only what the product itself embeds. A product that can be integrated with an external AI tool does not qualify; the feature must ship in the product.
10. Do not describe announcements, previews, or roadmap items as if they had shipped.
11. Give an honest confidence_score. A low score costs nothing.
12. Assess each task independently. These products are unrelated to each other.
"""

        parsed = self._batch_call(
            prompt=prompt,
            system_instruction=system_instruction,
            label=label,
            response_schema=BatchedFlagshipDescriptions,
            container="descriptions",
            max_output_tokens=FLAGSHIP_TOKENS_PER_ITEM * len(wanted) + BATCH_TOKENS_OVERHEAD,
        )

        expected = {item["item_id"] for item in wanted}
        for description in parsed:
            item_id = str(description.get("item_id") or "")
            if item_id not in expected or item_id in results:
                logger.warning("Batched flagship lookup returned an unusable item_id %r.", item_id)
                continue
            results[item_id] = {
                key: value for key, value in description.items() if key != "item_id"
            }

        missing = [item for item in wanted if item["item_id"] not in results]
        if not missing:
            return results

        if len(missing) == len(wanted):
            logger.warning(
                "Batched flagship lookup for %d skill(s) returned nothing usable. These "
                "skills keep their existing scoring.", len(wanted),
            )
            return results

        logger.info(
            "Batched flagship lookup answered %d of %d; re-asking the remaining %d.",
            len(results), len(wanted), len(missing),
        )
        results.update(self.describe_flagship_versions_batch(missing))
        return results

    def grade_embedding_batch(
        self, items: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Reads search results for several products and says which of them embed AI.

        Each item is {"item_id", "skill_name", "definition", "results"}, where `results`
        is the list embedding_probe.search_embedding_evidence returned. Returns
        {item_id: verdict}, each carrying embeds_ai, evidence, evidence_url and
        confidence.

        AN ID MISSING FROM THE ANSWER IS ABSENT FROM THE RESULT, and callers must read
        that as `unknown` rather than False. This is the same discipline
        verify_definitions_batch follows and it matters more here: a truncated response
        or a quota wall would otherwise mark a product as shipping no AI, strip its
        boost, and be indistinguishable from a real finding.

        THE DEFINITION IS SENT ALONGSIDE THE RESULTS, and it is the reason this works at
        all. Search results for "does LaTeX embed AI" are dominated by Overleaf, Prism
        and Underleaf. Knowing that LaTeX is a document preparation system from 1984 is
        what lets the model see those as third-party editors rather than as LaTeX
        shipping AI. It costs no extra request; the definition is already on file.

        Reconciliation, re-asking and single-item fallback follow
        describe_flagship_versions_batch exactly; the two differ only in what they ask.
        """
        results: Dict[str, Dict[str, Any]] = {}
        wanted = [
            item for item in items
            if item.get("item_id") and item.get("results")
        ]
        if not wanted:
            # Sending an item with no search results would ask the model to judge from
            # nothing, and it would answer. Absent is the correct outcome: no evidence
            # was gathered, so nothing was established.
            return results

        label = f"{len(wanted)} embedding verdict(s) ({wanted[0].get('skill_name')} and others)"

        blocks = ""
        for item in wanted:
            definition = str(item.get("definition") or "").strip()[:MAX_SOURCE_CHARS]
            blocks += (
                f"\n--- TASK item_id={item['item_id']} ---\n"
                f"Product name, exactly as it appears in the O*NET taxonomy: {item.get('skill_name') or ''}\n"
                f"What this product is, from our own records: {definition or '(none recorded)'}\n"
                f"Web search results for \"does {item.get('skill_name')} embed AI\":\n"
                f"{format_results(item.get('results') or [])}\n"
                f"--- END TASK item_id={item['item_id']} ---\n"
            )

        system_instruction = (
            "You are an expert on which AI features have actually shipped inside "
            "software products. You read web search results critically and you are not "
            "persuaded by a result merely mentioning a product name next to the word AI. "
            "You distinguish sharply between a feature that ships in a product, a "
            "separate product built around it by someone else, and an integration a user "
            "can wire up themselves. You are given several unrelated products in one "
            "request and you assess each on its own. You are comfortable answering that "
            "a product has no AI features, because most software does not."
        )

        prompt = f"""
You are given {len(wanted)} INDEPENDENT products. For each one, decide whether THAT PRODUCT ITSELF ships AI features.

{blocks}

Instructions:
1. Return exactly one verdict object per task, echoing that task's item_id verbatim. Do not merge, skip, or invent tasks.
2. Use the definition on file to establish WHAT THE PRODUCT IS before you read the results. The results are web pages and many of them are about something else that shares a word with the name.
3. Set embeds_ai true ONLY when the product itself ships AI features that are generally available today.
4. A separate product built around this one does NOT count. Searching for a document preparation system returns third-party editors that add AI to it; those are different products by different vendors, and the answer for the original is false.
5. An integration the user can connect does NOT count. A product that can send data to an external AI service has not embedded anything. The feature must ship in the product.
6. A plugin, extension or add-on by a THIRD PARTY does not count. One shipped by the product's own vendor as part of the product does.
7. Do not count announcements, previews, betas or roadmap items as shipped.
8. Do not count a vendor's other products. If the results describe AI in a different product from the same company, that is false for this one.
9. Set embeds_ai false when the results simply do not establish it. False is the correct and expected answer for most of these, especially for legacy, niche, discontinued and purely technical software.
10. In evidence, name the specific feature in one sentence when true, or say in one sentence why the results do not establish it when false. Do not restate the product name and stop.
11. Copy evidence_url exactly from one of the results given, or leave it null. Never invent a URL.
12. Give an honest confidence. Thin or off-topic results deserve a low score.
13. Assess each task independently. These products are unrelated to each other.
"""

        parsed = self._batch_call(
            prompt=prompt,
            system_instruction=system_instruction,
            label=label,
            response_schema=BatchedEmbeddingVerdicts,
            container="verdicts",
            max_output_tokens=EMBEDDING_TOKENS_PER_ITEM * len(wanted) + BATCH_TOKENS_OVERHEAD,
        )

        expected = {item["item_id"] for item in wanted}
        # The HOSTS actually surfaced by the search, per item, to check citations against.
        #
        # MEASURED, twice, and the two measurements pull in opposite directions:
        #
        #   - Grading LaTeX, the model returned the right verdict citing
        #     "openai.com/index/prism-a-free-latex-editor-...", a domain that was in none
        #     of the results. That is a fabricated source pointing at an unrelated
        #     company, and the drawer renders it as a link.
        #   - Grading six Adobe products, four cited helpx.adobe.com deep links that were
        #     also not verbatim in the results -- but helpx.adobe.com WAS among the result
        #     hosts. Those are the canonical vendor documentation pages, and they are the
        #     single most useful thing to link.
        #
        # So the check is on the HOST, not the exact URL. Exact matching dropped every
        # one of those four and left the column empty; host matching keeps them and still
        # rejects openai.com for LaTeX. The residual risk is a deep path that 404s on a
        # host the search did surface, which is a broken link on the right vendor's site
        # rather than evidence attributed to the wrong company.
        offered = {
            item["item_id"]: {
                host for host in (
                    urlparse(str(result.get("url") or "")).hostname
                    for result in (item.get("results") or [])
                ) if host
            }
            for item in wanted
        }

        for verdict in parsed:
            item_id = str(verdict.get("item_id") or "")
            if item_id not in expected or item_id in results:
                logger.warning("Batched embedding grading returned an unusable item_id %r.", item_id)
                continue

            cleaned = {key: value for key, value in verdict.items() if key != "item_id"}
            cited = str(cleaned.get("evidence_url") or "").strip()
            if cited and urlparse(cited).hostname not in offered.get(item_id, set()):
                logger.warning(
                    "Embedding verdict for %r cited %s, a host that was not in its search "
                    "results. Dropping the citation; the verdict stands.",
                    item_id, cited[:120],
                )
                cleaned["evidence_url"] = None
            results[item_id] = cleaned

        missing = [item for item in wanted if item["item_id"] not in results]
        if not missing:
            return results

        if len(missing) == len(wanted):
            logger.warning(
                "Batched embedding grading for %d skill(s) returned nothing usable. "
                "These skills stay unknown and will be retried on the next run.",
                len(wanted),
            )
            return results

        logger.info(
            "Batched embedding grading answered %d of %d; re-asking the remaining %d.",
            len(results), len(wanted), len(missing),
        )
        results.update(self.grade_embedding_batch(missing))
        return results

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
1. If you genuinely know this product, write its definition.
   {DEFINITION_SPEC}
2. If you do not know it, or you would have to guess at what it does, set definition to null. A guess here is worse than nothing, because a human will read it as researched.
3. Do not speculate about features, versions, or market position. Do not write marketing language.
4. Set is_software_product to false if the name does not appear to denote a real software product or technology at all.
5. Length is a target, not a quota. If you know only enough for one solid sentence, write one solid sentence rather than padding it to reach the range.
"""

        return self._call_or_none(
            prompt=prompt,
            system_instruction=system_instruction,
            skill_name=skill_name,
            response_schema=DefinitionDraft,
            max_output_tokens=DRAFT_MAX_TOKENS,
            what="definition draft",
        )


_source_checkers: Dict[str, AgenticSourceChecker] = {}


def get_source_checker(role: str = DEFAULT_ROLE) -> AgenticSourceChecker:
    """
    Lazily constructs the Gemini client and key pool for one ROLE.

    Deliberately not instantiated at import time: a missing GEMINI_API_KEY used to
    break the scraping.py import chain and take down the whole web app, even though
    the audit layer is only reached for candidates that already passed the gate.

    Cached PER ROLE for the process, so key rotation persists across skills within one
    run -- rebuilding the pool per call would restart at the role's .env key and
    re-probe a key already known to be spent, wasting a request and up to 60 seconds
    each time. Keying the cache by role is what actually delivers the isolation: a
    single shared instance would put every task on whichever key the first caller
    happened to open, and the .env slots would do nothing.

    An unknown role is not an error -- it falls back to the audit pool, matching
    gemini_keys.env_var_for -- so a typo degrades to the old shared behaviour rather
    than taking a pass down mid-run.
    """
    key = role if role in GEMINI_ROLES else DEFAULT_ROLE
    if key not in _source_checkers:
        logger.info("Opening a Gemini client for %s.", describe_role(key))
        _source_checkers[key] = AgenticSourceChecker(role=key)
    return _source_checkers[key]


def reset_source_checker(role: Optional[str] = None) -> None:
    """
    Drops cached checkers so the next call rebuilds the pool. For tests.

    Clears every role by default. A test that reset only its own role would leave
    another role's client alive holding a key the test has since changed.
    """
    if role is None:
        _source_checkers.clear()
    else:
        _source_checkers.pop(role, None)
