"""
Candidate acquisition and relevance gating.

Order of operations, which is deliberate:
  1. Resolve candidate reference pages for a skill (hard mappings, then exact-title
     probes, then a search pool only if needed).
  2. Cross-encoder scores each candidate against the skill to decide whether the
     page is ABOUT the skill. Highest scorer wins.
  3. Only if the winner clears CROSS_ENCODER_THRESHOLD does Gemini audit it for
     credibility and distill a summary.

Step 3 never runs on a page step 2 already rejected, so no API quota is spent
auditing a page known to be wrong.

Rate-limit discipline:
  - Persistent JSON cache (CACHE_FILE) short-circuits repeat scraping entirely.
  - Every outbound request retries HTTP 429 with exponential backoff, honoring
    Retry-After when the server sends it.
  - Candidate probing is tiered: a confident first hit skips the search pool, which
    is what keeps per-skill request counts in single digits.
"""

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, unquote, urlparse

import requests
from dotenv import load_dotenv

import storage
from agentic_source_check import AuditUnavailable, get_source_checker
from cross_encoder import score_pair

logger = logging.getLogger(__name__)

load_dotenv()

TOKEN = os.getenv("CAREERONESTOP_TOKEN", "").strip().replace(";", "")
USER_ID = os.getenv("CAREERONESTOP_USER_ID", "").strip()
USER_AGENT = os.getenv("USER_AGENT", "AISkillsAnalyticsDashboard/1.0")
CACHE_FILE = storage.CACHE_FILE

# A candidate page must reach this score for Gemini to audit it. Below it, the
# skill goes to human review with the raw extract attached and no API call spent.
# Calibrated against ms-marco sigmoid output: correct pages land near 0.99, a
# wrong-sense page near 0.10, and disambiguation pages near 0.83.
CROSS_ENCODER_THRESHOLD = 0.90

# Cache entries written by an older resolution strategy are ignored rather than
# trusted. Bump this whenever candidate resolution or scoring changes.
CACHE_SCHEMA_VERSION = 2

# Retry policy for all outbound scraping.
MAX_ATTEMPTS = 5
INITIAL_BACKOFF = 2.0
MAX_BACKOFF = 60.0
POLITENESS_DELAY = 0.05

session = requests.Session()
WIKI_HEADERS = {"User-Agent": USER_AGENT}

NOISE_CLEANER = re.compile(
    r'(?i)\b(software|systems?|analytics|big data software|tools?|database|language|platform|framework|library)\b'
)

DISAMBIGUATION_MARKERS = (
    "may refer to",
    "may also refer to",
    "commonly refers to",
    "can refer to",
)

MAJOR_LANGUAGES_LIST = {
    "c", "c++", "c#", "python", "r", "java", "javascript", "typescript", "ruby", "go",
    "golang", "scala", "julia", "perl", "php", "rust", "swift", "kotlin", "bash", "shell",
    "html", "css", "sql", "pl/sql", "matlab", "fortran", "cobol", "lisp", "haskell", "clojure",
}

# Canonical destinations for entities the search API resolves incorrectly.
EXACT_HARD_MAPPINGS = {
    "c#": "C Sharp (programming language)",
    "c++": "C++",
    "c": "C (programming language)",
    "r": "R (programming language)",
    "go": "Go (programming language)",
    "julia": "Julia (programming language)",
    "pandas": "Pandas (software)",
    "pyspark": "Apache Spark",
    "mlflow": "MLflow",
    "amazon simple storage service s3": "Amazon S3",
    "amazon web services aws sagemaker": "Amazon SageMaker",
    "sagemaker": "Amazon SageMaker",
}


# --------------------------------------------------------------------------
# Persistent scrape cache
# --------------------------------------------------------------------------

def load_local_cache() -> Dict[str, Any]:
    """
    Loads the scrape cache, discarding entries from an older schema version.

    The version filter stays HERE rather than moving to storage.py: it encodes which
    candidate-resolution strategy produced an entry, which is a scraping fact, not a
    storage one.
    """
    try:
        raw = storage.load_scrape_cache_raw()
    except storage.StorageUnreadable as err:
        logger.warning("Could not read scrape cache %s (%s). Starting empty.", CACHE_FILE, err)
        return {}

    if not isinstance(raw, dict):
        return {}

    kept = {
        key: value
        for key, value in raw.items()
        if isinstance(value, dict) and value.get("schema_version") == CACHE_SCHEMA_VERSION
    }
    dropped = len(raw) - len(kept)
    if dropped:
        logger.info("Discarded %d stale cache entries below schema version %d.", dropped, CACHE_SCHEMA_VERSION)
    return kept


def save_local_cache(cache_data: Dict[str, Any]) -> None:
    try:
        storage.save_scrape_cache_raw(cache_data)
    except OSError as err:
        logger.warning("Could not write scrape cache %s (%s).", CACHE_FILE, err)


# --------------------------------------------------------------------------
# HTTP with 429 backoff
# --------------------------------------------------------------------------

def _sleep_for_retry(response: Optional[requests.Response], backoff: float) -> float:
    """Waits, preferring the server's Retry-After header, and returns the next backoff."""
    wait = backoff
    if response is not None:
        header = response.headers.get("Retry-After")
        if header:
            try:
                wait = float(header)
            except ValueError:
                pass
    wait = min(wait, MAX_BACKOFF)
    time.sleep(wait)
    return min(backoff * 2, MAX_BACKOFF)


def request_with_backoff(
    url: str,
    *,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: float = 4.0,
) -> Optional[requests.Response]:
    """
    GETs a URL, retrying on 429 and 5xx with doubling backoff.

    Unlike the legacy loop, a transport exception also waits before retrying, so a
    DNS or timeout failure cannot burn every attempt instantly.
    """
    backoff = INITIAL_BACKOFF

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = session.get(url, params=params, headers=headers, timeout=timeout)
        except requests.RequestException as err:
            logger.warning("Request to %s failed (attempt %d/%d): %s", url, attempt, MAX_ATTEMPTS, err)
            backoff = _sleep_for_retry(None, backoff)
            continue

        if response.status_code == 200:
            return response

        if response.status_code == 429:
            logger.warning(
                "Rate limited by %s (attempt %d/%d). Backing off.", url, attempt, MAX_ATTEMPTS
            )
            backoff = _sleep_for_retry(response, backoff)
            continue

        if 500 <= response.status_code < 600:
            logger.warning(
                "Upstream %d from %s (attempt %d/%d).", response.status_code, url, attempt, MAX_ATTEMPTS
            )
            backoff = _sleep_for_retry(response, backoff)
            continue

        # 4xx other than 429 will not improve on retry.
        logger.debug("Non-retryable %d from %s.", response.status_code, url)
        return None

    logger.error("Exhausted %d attempts against %s.", MAX_ATTEMPTS, url)
    return None


# --------------------------------------------------------------------------
# Skill name normalization and taxonomy context
# --------------------------------------------------------------------------

def normalize_skill_name(raw_name: str) -> str:
    return str(raw_name).replace("\xa0", " ").replace("  ", " ").strip()


def clean_skill_name(raw_name: str) -> str:
    """Strips taxonomy noise words, falling back to the original if nothing survives."""
    cleaned = NOISE_CLEANER.sub("", normalize_skill_name(raw_name)).strip()
    cleaned = cleaned.replace("  ", " ")
    return cleaned if cleaned else normalize_skill_name(raw_name)


def derive_entity_suffix(category_title: str) -> str:
    """
    Recovers the disambiguation hint that NOISE_CLEANER strips out.

    The noise words are removed from the lookup term but carry real signal about what
    kind of entity we are looking for, so the O*NET category supplies it back.
    """
    category_lower = str(category_title or "").lower()
    if "language" in category_lower:
        return "programming language"
    if "database" in category_lower or "data base" in category_lower:
        return "database"
    return "software"


def _is_disambiguation(title: str, extract: str) -> bool:
    if title and "(disambiguation)" in title.lower():
        return True
    head = str(extract or "")[:200].lower()
    return any(marker in head for marker in DISAMBIGUATION_MARKERS)


# --------------------------------------------------------------------------
# Wikipedia and Wikidata retrieval
# --------------------------------------------------------------------------

def check_exact_title_match(title_string: str) -> Optional[str]:
    """Resolves a candidate title through Wikipedia redirects, or None if no page exists."""
    if not title_string:
        return None

    time.sleep(POLITENESS_DELAY)
    response = request_with_backoff(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "titles": unquote(title_string),
            "redirects": 1,
            "format": "json",
        },
        headers=WIKI_HEADERS,
        timeout=3.0,
    )
    if response is None:
        return None

    try:
        pages = response.json().get("query", {}).get("pages", {})
    except ValueError:
        return None

    if "-1" in pages:
        return None
    for page_id in pages:
        return pages[page_id].get("title")
    return None


def fetch_wikipedia_extract(resolved_title: str) -> str:
    """
    Two-tier summary retrieval: the extracts API first, then the REST summary endpoint.

    Both tiers retry 429 independently. Returns an empty string if neither yields text.
    """
    if not resolved_title:
        return ""

    time.sleep(POLITENESS_DELAY)
    response = request_with_backoff(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "titles": resolved_title,
            "redirects": 1,
            "format": "json",
        },
        headers=WIKI_HEADERS,
        timeout=4.0,
    )

    extract = ""
    if response is not None:
        try:
            pages = response.json().get("query", {}).get("pages", {})
            for page_id, page in pages.items():
                if page_id != "-1":
                    extract = str(page.get("extract", "")).strip()
                    if extract:
                        break
        except ValueError:
            extract = ""

    if extract:
        return extract

    sanitized = quote(resolved_title.replace(" ", "_"))
    rest_response = request_with_backoff(
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{sanitized}",
        headers=WIKI_HEADERS,
        timeout=4.0,
    )
    if rest_response is not None:
        try:
            return str(rest_response.json().get("extract", "")).strip()
        except ValueError:
            return ""
    return ""


def search_wikipedia_titles(cleaned_query: str, entity_suffix: str, limit: int = 6) -> List[str]:
    """Returns candidate page titles from the Wikipedia search index."""
    response = request_with_backoff(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": f"{cleaned_query} {entity_suffix} computer technology",
            "srlimit": limit,
            "format": "json",
        },
        headers=WIKI_HEADERS,
        timeout=3.0,
    )
    if response is None:
        return []
    try:
        results = response.json().get("query", {}).get("search", [])
    except ValueError:
        return []
    return [entry.get("title") for entry in results if entry.get("title")]


def fetch_wikidata_summary(query: str) -> Optional[Tuple[str, str]]:
    """Returns (label, description sentence) from Wikidata, or None."""
    response = request_with_backoff(
        "https://www.wikidata.org/w/api.php",
        params={
            "action": "wbsearchentities",
            "search": query,
            "language": "en",
            "format": "json",
        },
        headers=WIKI_HEADERS,
        timeout=4.0,
    )
    if response is None:
        return None
    try:
        results = response.json().get("search", [])
    except ValueError:
        return None
    if not results:
        return None

    label = str(results[0].get("label", query)).strip()
    description = str(results[0].get("description", "")).strip()
    if len(description) <= 10:
        return None
    return label, f"{label}: {description}."


# --------------------------------------------------------------------------
# Candidate resolution and cross-encoder selection
# --------------------------------------------------------------------------

def _score_candidate(query: str, title: str, extract: str) -> Optional[float]:
    if _is_disambiguation(title, extract):
        logger.debug("Rejecting disambiguation candidate %r.", title)
        return None
    return score_pair(query, title, extract)


def resolve_best_candidate(skill_name: str, category_title: str) -> Dict[str, Any]:
    """
    Finds the best reference page for a skill and scores it with the cross-encoder.

    Probing is tiered to keep request volume down: a hard mapping or exact-title hit
    that already clears the threshold wins immediately and the search pool is never
    queried. The pool only opens when the cheap path is unconvincing.
    """
    raw_input_name = normalize_skill_name(skill_name)
    if not raw_input_name:
        return {"title": None, "extract": "", "score": None, "source_name": None}

    cleaned_query = clean_skill_name(raw_input_name)
    entity_suffix = derive_entity_suffix(category_title)
    query = f"{cleaned_query} ({entity_suffix})"

    lookup_keys = [raw_input_name.lower(), cleaned_query.lower()]

    # Tier 1: canonical hard mappings.
    priority_seeds: List[str] = []
    for key in lookup_keys:
        mapped = EXACT_HARD_MAPPINGS.get(key)
        if mapped and mapped not in priority_seeds:
            priority_seeds.append(mapped)

    # Tier 2: constructed exact titles.
    if any(key in MAJOR_LANGUAGES_LIST for key in lookup_keys):
        priority_seeds.append(f"{cleaned_query.capitalize()} (programming language)")
    priority_seeds.extend([
        f"{cleaned_query.capitalize()} ({entity_suffix})",
        cleaned_query,
        raw_input_name,
    ])

    best = {"title": None, "extract": "", "score": None, "source_name": None}
    probed: set = set()

    def consider(candidate_title: str) -> None:
        resolved = check_exact_title_match(candidate_title)
        if not resolved or resolved in probed:
            return
        probed.add(resolved)

        extract = fetch_wikipedia_extract(resolved)
        if not extract:
            return

        score = _score_candidate(query, resolved, extract)
        if score is None:
            return
        if best["score"] is None or score > best["score"]:
            best.update({
                "title": resolved,
                "extract": extract,
                "score": score,
                "source_name": "Wikipedia",
            })

    for seed in priority_seeds:
        consider(seed)
        if best["score"] is not None and best["score"] >= CROSS_ENCODER_THRESHOLD:
            logger.debug(
                "Confident early match for %r: %r (%.4f). Skipping search pool.",
                raw_input_name, best["title"], best["score"],
            )
            return best

    # Tier 3: search pool, only reached when the cheap path was unconvincing.
    for candidate_title in search_wikipedia_titles(cleaned_query, entity_suffix):
        consider(candidate_title)
        if best["score"] is not None and best["score"] >= CROSS_ENCODER_THRESHOLD:
            return best

    # Tier 4: Wikidata, as a short-description fallback when Wikipedia yields nothing.
    if best["score"] is None:
        wikidata = fetch_wikidata_summary(cleaned_query)
        if wikidata:
            label, description = wikidata
            score = _score_candidate(query, label, description)
            if score is not None:
                best.update({
                    "title": label,
                    "extract": description,
                    "score": score,
                    "source_name": "Wikidata",
                })

    return best


# --------------------------------------------------------------------------
# Reviewer remediation
# --------------------------------------------------------------------------

# Error codes resolve_reviewer_url can return. Kept as a tuple so main.py can assert
# its message table covers every one of them rather than discovering a gap in
# production, where the fallback would be a generic sentence.
REMEDIATION_ERROR_CODES = (
    "empty",
    "bad_url",
    "not_wikipedia",
    "no_page",
    "disambiguation",
    "no_text",
)


def _extract_wikipedia_title(raw_reference: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Turns a reviewer-supplied reference into a Wikipedia page title.

    Accepts a full article URL or a bare page title. Returns (title, error_code) with
    exactly one of the two set.

    A non-Wikipedia URL is an error rather than being retried as a page title. Treating
    "https://github.com/apache/spark" as a title would send that whole string to the
    Wikipedia API, get no page, and report "no such page" -- which tells the reviewer
    nothing about the real problem, that this tool only reads Wikipedia.
    """
    reference = (raw_reference or "").strip()
    if not reference:
        return None, "empty"

    parsed = urlparse(reference)

    if not parsed.scheme:
        # Not a URL, so the whole string is the title. Strip any stray fragment.
        return reference.split("#", 1)[0].strip() or None, None

    if parsed.scheme not in ("http", "https"):
        return None, "bad_url"

    # English only, and checked against a fixed set rather than an endswith test.
    # fetch_wikipedia_extract queries en.wikipedia.org unconditionally, so accepting a
    # de.wikipedia.org link would look its title up in the English wiki and either find
    # nothing or, worse, find a DIFFERENT article that happens to share the name. An
    # endswith(".wikipedia.org") test would also accept wikipedia.org.evil.com.
    host = parsed.netloc.split(":", 1)[0].lower()
    if host not in ("en.wikipedia.org", "en.m.wikipedia.org", "wikipedia.org", "www.wikipedia.org"):
        return None, "not_wikipedia"

    path = unquote(parsed.path or "")
    if not path.startswith("/wiki/"):
        # Rules out /w/index.php?title=..., Special: pages, and the bare domain, none
        # of which name an article unambiguously.
        return None, "bad_url"

    title = path[len("/wiki/"):].replace("_", " ").strip()
    return (title, None) if title else (None, "bad_url")


def validate_wikipedia_reference(raw_reference: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Confirms a reference names a real Wikipedia page, WITHOUT fetching its text.

    Returns (resolved_title, error_code) with exactly one set, using the same error codes
    as resolve_reviewer_url.

    This is the link-only half of remediation, for the post-mortem edit path where the
    editor has already written the definition text by hand and only the recorded page
    needs checking. Fetching the extract there would overwrite what they typed.
    """
    title, error = _extract_wikipedia_title(raw_reference)
    if error:
        return None, error

    resolved = check_exact_title_match(title)
    if not resolved:
        return None, "no_page"
    return resolved, None


def resolve_reviewer_url(
    skill_name: str, category_title: str, raw_reference: str
) -> Dict[str, Any]:
    """
    Resolves a reference page a human reviewer supplied by hand.

    Same return shape as resolve_best_candidate, plus an `error` key that is None on
    success and one of REMEDIATION_ERROR_CODES otherwise.

    The cross-encoder score is RECORDED, NOT ENFORCED. A reviewer who supplies a page
    is asserting the thing the cross-encoder exists to guess, so re-applying
    CROSS_ENCODER_THRESHOLD here would reject exactly the cases the field was added
    for -- the hard-mapped aliases like pyspark against Apache Spark, which score near
    zero on lexical difference and are nonetheless correct. The score is stored so the
    card can show that the model still disagrees.

    A None score (ONNX inference failed) is not an error either. The text is real and
    the reviewer can still read it; the card renders "Not scored".

    Nothing here writes to the scrape cache. The cache answers the pipeline's own
    generated query, and a human override is not a better answer to that query.
    """
    failure = {"title": None, "extract": "", "score": None, "source_name": None}

    title, error = _extract_wikipedia_title(raw_reference)
    if error:
        return {**failure, "error": error}

    resolved = check_exact_title_match(title)
    if not resolved:
        return {**failure, "error": "no_page"}

    extract = fetch_wikipedia_extract(resolved)
    if not extract:
        return {**failure, "error": "no_text"}

    # Checked before scoring, and separately from _score_candidate, so a disambiguation
    # landing page reports itself as one instead of as a generic scoring failure.
    if _is_disambiguation(resolved, extract):
        return {**failure, "error": "disambiguation"}

    query = f"{clean_skill_name(normalize_skill_name(skill_name))} ({derive_entity_suffix(category_title)})"
    score = score_pair(query, resolved, extract)
    if score is None:
        logger.warning(
            "Cross-encoder could not score reviewer-supplied page %r for %r. Storing "
            "the text with no score.",
            resolved, skill_name,
        )

    logger.info(
        "Reviewer remediation for %r resolved to %r (score=%s).",
        skill_name, resolved, "none" if score is None else f"{score:.4f}",
    )
    return {
        "title": resolved,
        "extract": extract,
        "score": score,
        "source_name": "Wikipedia",
        "error": None,
    }


# --------------------------------------------------------------------------
# O*NET ingestion
# --------------------------------------------------------------------------

def fetch_all_onet_codes() -> Dict[str, str]:
    """
    Fetches active O*NET codes from CareerOneStop, falling back to a core tech list.

    The fallback is logged at ERROR because a silent fallback makes a broken full run
    look like a small successful one.
    """
    url = f"https://api.careeronestop.org/v1/occupation/{USER_ID}/0/Y/0/1000"
    response = request_with_backoff(
        url,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
        timeout=8.0,
    )

    if response is not None:
        try:
            occupations = response.json().get("OccupationList", [])
            onet_map = {
                item["OnetCode"]: item.get("OnetTitle", "Unknown")
                for item in occupations
                if "OnetCode" in item
            }
            if onet_map:
                logger.info("Retrieved %d O*NET codes from CareerOneStop.", len(onet_map))
                return onet_map
        except ValueError:
            pass

    logger.error(
        "CareerOneStop code listing unavailable. Falling back to 4 hardcoded codes; "
        "this is NOT a full taxonomy run."
    )
    return {
        "15-1252.00": "Software Developers",
        "15-1251.00": "Computer Programmers",
        "15-1243.00": "Database Architects",
        "15-1211.00": "Computer Systems Analysts",
    }


def search_onet_occupations(keyword: str, limit: int = 500) -> Dict[str, str]:
    """
    Returns {onet_code: onet_title} from the CareerOneStop occupation SEARCH endpoint.

    Used instead of fetch_all_onet_codes because the bulk listing URL that function
    calls answers 404 "No data available" -- it has been silently falling back to its
    four hardcoded codes on every run. This endpoint works and, usefully, accepts a
    SOC prefix as the keyword: "15-" returns the 38 computer and mathematical
    occupations.

    The match is FUZZY. Searching "11-" returns 80 records of which only 59 are
    actually 11- occupations, so the caller must filter by prefix rather than trusting
    the result set. select_onet_codes does exactly that.
    """
    if not keyword:
        return {}

    url = (
        f"https://api.careeronestop.org/v1/occupation/"
        f"{USER_ID}/{quote(str(keyword))}/N/0/{int(limit)}"
    )
    response = request_with_backoff(
        url,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
        timeout=20.0,
    )
    if response is None:
        logger.error("Occupation search for %r failed.", keyword)
        return {}

    try:
        occupations = response.json().get("OccupationList", []) or []
    except ValueError:
        logger.error("Occupation search for %r returned unparseable JSON.", keyword)
        return {}

    found = {
        item["OnetCode"]: item.get("OnetTitle", "Unknown")
        for item in occupations
        if item.get("OnetCode")
    }
    logger.info("Occupation search %r returned %d occupations.", keyword, len(found))
    return found


def fetch_onet_tools_and_tech(onet_code: str) -> Dict[str, Any]:
    """
    Pulls every tool and technology for an occupation, flagging which are Hot Tech.

    All tools are retained, not just hot ones, because Hot Tech status belongs to the
    (skill, occupation) relationship: discarding the non-hot ones made is_hot_tech
    universally True and therefore meaningless. The caller records every tool in
    Skill_Occupation_Map with its true flag, and spends the expensive path
    (candidate resolution, cross-encoder, Gemini audit) only on the hot subset.

    Full taxonomy context is preserved per skill -- category title, derived entity
    type, and the raw O*NET name -- so downstream cross-encoder queries have context.
    """
    url = (
        f"https://api.careeronestop.org/v1/occupation/{USER_ID}/{onet_code}/US"
        "?skills=false&toolsAndTechnology=true&tasks=false&alternateOnetTitles=false"
    )
    response = request_with_backoff(
        url,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        timeout=6.0,
    )

    empty = {"onet_code": onet_code, "onet_title": "Unknown", "skills": []}
    if response is None:
        logger.error("No usable response for O*NET code %s.", onet_code)
        return empty

    try:
        details = response.json().get("OccupationDetail", [])
    except ValueError:
        logger.error("Malformed JSON for O*NET code %s.", onet_code)
        return empty

    if not details:
        logger.warning("Empty OccupationDetail payload for O*NET code %s.", onet_code)
        return empty

    onet_title = str(details[0].get("OnetTitle", "Unknown Title")).strip()
    tools_tech = details[0].get("ToolsAndTechOccupationDetails", {}) or {}
    tech_node = tools_tech.get("Technology") or {}
    tools_node = tools_tech.get("Tools") or {}
    category_list = (tech_node.get("CategoryList") or []) + (tools_node.get("CategoryList") or [])

    # Keyed on the LOWERCASED name, for two reasons. First, O*NET repeats the same
    # tool under several categories within one occupation, and those occurrences can
    # disagree about Hot_Technology -- so hot status must be OR-aggregated rather than
    # first-wins, or a tool listed non-hot first and hot second would be recorded as
    # not hot and silently skipped for auditing. Second, Skills_Master.skill_name is
    # UNIQUE under a case-insensitive collation, so "Golang" and "GoLang" would
    # survive a case-sensitive dedupe here only to collide in SQL with the last
    # writer's flag winning. Lowercasing also matches the cache key used below.
    by_key: Dict[str, Dict[str, Any]] = {}

    for category in category_list:
        category_title = str(category.get("Title", "")).strip()
        for example in (category.get("Examples", []) or []):
            skill_name = normalize_skill_name(example.get("Name", ""))
            if not skill_name:
                continue

            is_hot = example.get("Hot_Technology") == "Y"
            key = skill_name.lower()
            existing = by_key.get(key)

            if existing is None:
                by_key[key] = {
                    "skill_name": skill_name,
                    "cleaned_name": clean_skill_name(skill_name),
                    "category": category_title,
                    "entity_suffix": derive_entity_suffix(category_title),
                    "is_hot_tech": is_hot,
                }
                continue

            # Hot wins. Prefer the hot occurrence's spelling and category too: the
            # category feeds derive_entity_suffix and the cross-encoder query, so the
            # taxonomy context of the occurrence we will actually audit is the useful one.
            if is_hot and not existing["is_hot_tech"]:
                existing.update({
                    "skill_name": skill_name,
                    "cleaned_name": clean_skill_name(skill_name),
                    "category": category_title,
                    "entity_suffix": derive_entity_suffix(category_title),
                    "is_hot_tech": True,
                })

    skills = list(by_key.values())
    hot_count = sum(1 for item in skills if item["is_hot_tech"])

    logger.info(
        "O*NET %s (%s): %d tools retained, %d hot.",
        onet_code, onet_title, len(skills), hot_count,
    )
    return {"onet_code": onet_code, "onet_title": onet_title, "skills": skills}


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def scrape_and_validate_skill(
    skill_item: Dict[str, Any],
    cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Resolves, gates, and (conditionally) audits a single skill.

    Returns a decision record. `auto_approve` is True only when the cross-encoder
    cleared the threshold AND Gemini confirmed credibility. `is_credible` is None
    whenever Gemini was never called, which keeps "judged not credible" distinct
    from "never judged".

    Only ever called for Hot Tech skills. The no_candidate_found fallback summary
    below hardcodes "hot technology asset" and stays accurate solely because of that;
    if non-hot skills are ever routed here, parameterize that wording first.
    """
    skill_name = normalize_skill_name(skill_item["skill_name"])
    category = skill_item.get("category", "Technology")
    cache_key = skill_name.lower()

    cached = cache.get(cache_key) if cache is not None else None
    if cached:
        logger.debug("Scrape cache hit for %r.", skill_name)
        candidate = {
            "title": cached.get("resolved_title"),
            "extract": cached.get("raw_extract", ""),
            "score": cached.get("cross_score"),
            "source_name": cached.get("source_name"),
        }
    else:
        candidate = resolve_best_candidate(skill_name, category)
        if cache is not None:
            cache[cache_key] = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "resolved_title": candidate["title"],
                "raw_extract": candidate["extract"],
                "cross_score": candidate["score"],
                "source_name": candidate["source_name"],
                "cached_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

    cross_score = candidate["score"]
    raw_extract = candidate["extract"]

    # No usable candidate at all. Queue it with taxonomy context so a human can fix it.
    if not raw_extract or cross_score is None:
        logger.info("No scoreable candidate for %r. Routing to manual review.", skill_name)
        return {
            "resolved_title": candidate["title"],
            "raw_extract": "",
            "summary": f"{skill_name} is a hot technology asset categorized under {category} in the O*NET framework.",
            "cross_score": None,
            "is_credible": None,
            "best_source_name": None,
            "auto_approve": False,
            "gate_reason": "no_candidate_found",
        }

    # Gate: a page the cross-encoder rejected is not worth a Gemini call.
    if cross_score < CROSS_ENCODER_THRESHOLD:
        logger.info(
            "Cross-encoder %.4f below %.2f for %r (%r). Manual review, no Gemini call.",
            cross_score, CROSS_ENCODER_THRESHOLD, skill_name, candidate["title"],
        )
        return {
            "resolved_title": candidate["title"],
            "raw_extract": raw_extract,
            "summary": raw_extract,
            "cross_score": cross_score,
            "is_credible": None,
            "best_source_name": None,
            "auto_approve": False,
            "gate_reason": "below_relevance_threshold",
        }

    # Page looks correct. Now ask Gemini whether it is credible, and distill it.
    try:
        evaluation = get_source_checker().evaluate_candidates(
            skill_name=skill_name,
            candidates=[{
                "source_name": candidate["source_name"] or "Wikipedia",
                "raw_text": raw_extract,
            }],
        )
    except AuditUnavailable:
        # Quota exhausted or the API is down. The page pull was good, so keep the raw
        # extract and send it to a human, with is_credible NULL to record that no
        # audit happened. Recording False here would blame the source for an outage.
        logger.warning(
            "Credibility audit unavailable for %r (cross=%.4f). Manual review.",
            skill_name, cross_score,
        )
        return {
            "resolved_title": candidate["title"],
            "raw_extract": raw_extract,
            "summary": raw_extract,
            "cross_score": cross_score,
            "is_credible": None,
            "best_source_name": candidate["source_name"],
            "auto_approve": False,
            "gate_reason": "audit_unavailable",
        }

    is_credible = bool(evaluation.get("is_credible"))
    clean_summary = evaluation.get("clean_summary")

    if is_credible and clean_summary:
        logger.info(
            "Auto-approving %r (cross=%.4f, source=%s).",
            skill_name, cross_score, evaluation.get("best_source_name"),
        )
        return {
            "resolved_title": candidate["title"],
            "raw_extract": raw_extract,
            "summary": clean_summary,
            "cross_score": cross_score,
            "is_credible": True,
            "best_source_name": evaluation.get("best_source_name") or candidate["source_name"],
            "auto_approve": True,
            "gate_reason": "auto_approved",
        }

    logger.info(
        "Gemini rejected credibility for %r despite cross=%.4f. Manual review.",
        skill_name, cross_score,
    )
    return {
        "resolved_title": candidate["title"],
        "raw_extract": raw_extract,
        "summary": clean_summary or raw_extract,
        "cross_score": cross_score,
        "is_credible": False,
        "best_source_name": evaluation.get("best_source_name"),
        "auto_approve": False,
        "gate_reason": "failed_credibility_audit",
    }
