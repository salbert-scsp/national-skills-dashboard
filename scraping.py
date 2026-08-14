"""
STAGE: Acquisition

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
from gemini_keys import ROLE_AUDIT
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

# Below this, the best candidate is not kept AT ALL -- not as a reference page, not as
# definition text. It is a different question from the threshold above: that one asks
# "is this good enough to approve without a human?", this one asks "is this even about
# the right subject?".
#
# The search API always returns something. For a discontinued vendor tool with no
# article, that something is whatever shares a word with the query, and the pipeline was
# storing it: DynaSCAPE Design resolved to "List of ZX Spectrum games" at 0.000, and the
# reviewer's card then showed that article's text as the definition. 143 of 171 queued
# items were in that state. Nothing between 0.0 and 0.35 has ever been the right page.
#
# The skill is NOT dropped. It becomes no_candidate_found, which keeps its card, keeps
# its O*NET boilerplate, and is the bucket the second pass writes a real definition for.
RESOLUTION_FLOOR = 0.35

# Cache entries written by an older resolution strategy are ignored rather than
# trusted. Bump this whenever candidate resolution or scoring changes.
#
# 3: the resolution floor above, and the probe and search-query fixes below. Without the
# bump every junk resolution cached under version 2 would survive the change and be
# replayed on the next scrape.
CACHE_SCHEMA_VERSION = 3

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
    """
    Strips taxonomy noise words, falling back to the original if nothing survives.

    A noise word wedged BETWEEN two capitalised tokens is kept, because there it is
    almost always part of a company name rather than a category suffix. Stripping
    everywhere turned "Cosmo Software Cosmo World" into "Cosmo Cosmo World" and
    "IEA Software Emerald" into "IEA Emerald" -- strings that name nothing, which were
    then sent to the search API as the query. "Statistical analysis software" still
    loses its suffix, because nothing capitalised follows it.
    """
    normalized = normalize_skill_name(raw_name)

    kept = []
    tokens = normalized.split()
    for index, token in enumerate(tokens):
        if not NOISE_CLEANER.fullmatch(token):
            kept.append(token)
            continue
        previous = tokens[index - 1] if index else ""
        following = tokens[index + 1] if index + 1 < len(tokens) else ""
        if previous[:1].isupper() and following[:1].isupper():
            kept.append(token)

    cleaned = " ".join(kept).strip()
    return cleaned if cleaned else normalized


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
    """
    Returns candidate page titles from the Wikipedia search index.

    The trailing "computer technology" looks like padding and is load-bearing. Removing
    it was measured over ten no-article skills: it recovered one real page (HydroCAD)
    and introduced two false positives, the worst being "Cosmo Software Cosmo World"
    matching "Cosmo's Cosmic Adventure", a 1992 platform game, at 0.995 -- high enough
    to reach the audit, which then PASSED it as credible. The padding biases the index
    toward the technology domain, and that bias is the difference between a wrong page
    that scores 0.002 and gets queued, and one that scores 0.995 and gets approved.

    The cost is real and accepted: a product whose article does not read as technology
    to the search index is missed, and the second pass writes it a definition instead.
    """
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


# Vendor names O*NET prefixes onto a product. Wikipedia titles the ARTICLE by the product
# -- "Windchill (software)", not "PTC Windchill" -- so the prefix has to come off before
# the title will match. Kept as an explicit list rather than "drop the first token of any
# multi-word name", which would turn "Adobe Photoshop" into "Photoshop" and "Microsoft
# Excel" into "Excel"; those resolve correctly today and must not be disturbed.
VENDOR_PREFIXES = frozenset({
    "ptc", "bentley", "autodesk", "oracle", "ibm", "microsoft", "adobe", "sap", "carlson",
    "trimble", "esri", "dassault", "siemens", "ansys", "hexagon", "intergraph", "nemetschek",
    "graphisoft", "aveva", "emerson", "honeywell", "rockwell", "schneider", "yokogawa",
    "salesforce", "servicenow", "workday", "infor", "epicor", "sage", "intuit", "quest",
    "veritas", "symantec", "mcafee", "citrix", "vmware", "redhat", "novell", "borland",
    "embarcadero", "jetbrains", "atlassian", "progress", "tibco", "informatica", "talend",
    "qlik", "tableau", "alteryx", "mathworks", "wolfram", "minitab", "statsoft", "golden",
    "skyscape", "kodak", "nuance", "cerner", "epic", "allscripts", "meditech", "athenahealth",
})


def _disambiguated_seeds(
    cleaned_query: str, entity_suffix: str, raw_name: str = ""
) -> List[str]:
    """
    Extra exact-title probes for names Wikipedia files under a different title.

    Two shapes, both measured against the live API before being added:

        PTC Windchill              -> "Windchill (software)"           1.000
        Tool command language Tcl  -> "Tcl (programming language)"     1.000

    Neither was reachable before. The first needs the VENDOR PREFIX dropped, because
    O*NET writes "PTC Windchill" and the article is titled by the product alone. The
    second needs the LAST TOKEN probed on its own: NOISE_CLEANER strips both "Tool" and
    "language" from that name, leaving the unmatchable "command Tcl", and the last token
    reaches the real article regardless of what the cleaner did to the rest.

    These are exact-title lookups, so they cost a Wikipedia round trip each and no Gemini
    quota. They are appended AFTER the plain seeds, so a name that already resolves keeps
    resolving by the same route it always did.
    """
    tokens = cleaned_query.split()
    if len(tokens) < 2:
        return []

    # Suffixes to try on the last token. The derived one is read from the O*NET CATEGORY,
    # which does not always say what the raw NAME does: "Tool command language Tcl" sits
    # under "Development environment software", so the derived suffix is "software" and
    # the real article is "Tcl (programming language)". NOISE_CLEANER has already removed
    # the word "language" from the cleaned form, so the raw name is the only place that
    # signal survives.
    suffixes = [entity_suffix]
    if "language" in str(raw_name).lower() and entity_suffix != "programming language":
        suffixes.append("programming language")

    seeds: List[str] = []
    if tokens[0].lower() in VENDOR_PREFIXES:
        tail = " ".join(tokens[1:])
        seeds.append(f"{tail} ({entity_suffix})")
        seeds.append(tail)

    # Only worth probing when the last token looks like a name rather than a common word;
    # "... software" or "... system" as a bare title is a category page, not this product.
    last = tokens[-1]
    if len(last) > 2 and not NOISE_CLEANER.fullmatch(last):
        for suffix in suffixes:
            seeds.append(f"{last} ({suffix})")

    # Deduped, order preserved. The probe loop dedupes by RESOLVED title anyway, but a
    # repeat here would still cost a title lookup before that dedupe could see it.
    seen = set()
    unique = []
    for seed in seeds:
        key = seed.strip().lower()
        if seed.strip() and key not in seen:
            seen.add(key)
            unique.append(seed.strip())
    return unique


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
        # NOT .capitalize(). That lowercases everything after the first character, so
        # "IBM Content Manager" was probed as "Ibm content manager (software)" -- and
        # Wikipedia case-folds only the FIRST letter of a title, so the probe could
        # never match a multi-word proper noun. One wasted round trip per skill.
        f"{cleaned_query} ({entity_suffix})",
        cleaned_query,
        raw_input_name,
    ])
    priority_seeds.extend(_disambiguated_seeds(cleaned_query, entity_suffix, raw_input_name))

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
    # Reviewer-supplied links to anywhere else on the web.
    "bad_scheme",
    "blocked_host",
    "too_long",
    "fetch_failed",
    "not_html",
    "fragment_not_found",
    "needs_javascript",
    "too_thin",
)

# How long a pasted reference may be.
#
# The 2,083-character number people remember is IE's limit on a URL in a REQUEST LINE.
# Nothing here puts the reference in one: it is typed into a form field and travels in a
# JSON body, so header limits never apply to it. What sets the real floor is Chrome's
# "copy link to highlight", which appends a text fragment -- one sentence lands around
# 100-300 characters, a long multi-sentence selection with prefix and suffix context can
# pass 1,000. This covers any realistic one and still rejects a pasted essay.
MAX_REFERENCE_URL_CHARS = 4096

# Guards on fetching a URL a form supplied. This is the one place in the pipeline that
# retrieves an address chosen by whoever is at the keyboard.
EXTERNAL_TIMEOUT = 8.0
MAX_EXTERNAL_BYTES = 2_000_000

# What a stored definition may be. Measured against the store this has to sit alongside:
# Wikipedia-derived definitions run a median of 191 characters and a p90 of 279, and
# agentic_source_check.DEFINITION_SPEC asks the model for 200-400.
#
# The old bound was a single MAX of 4000 with no floor, and a stress test over 32 real
# URLs showed the results were bimodal and never definition-shaped: either a navigation
# stub (61, 137, 163 chars) or the whole page slammed against the cap (3135-4000). Both
# ends are wrong, and the long end is worse than untidy -- the extract IS the text the
# bi-encoder scores, so a 4000-character page dump of nginx.org measured ml_pipeline_sim
# 0.299 against 0.114 for a real one-line definition, crossing ML_PIPELINE_FLOOR and
# classifying the skill AI Enabling where Wikipedia had it Not AI. A vendor-sourced
# skill would systematically outrank the same skill sourced from an encyclopedia.
MIN_EXTERNAL_EXTRACT_CHARS = 120
MAX_EXTERNAL_EXTRACT_CHARS = 600

# Readable text that is not text/html. docs.docker.com serves text/markdown and was
# refused outright as not_html, which is a false refusal: markdown is plain prose.
EXTERNAL_CONTENT_TYPES = (
    "text/html",
    "text/plain",
    "text/markdown",
    "application/xhtml",
    "application/xml",
    "text/xml",
)

# Structural furniture that is on the page but is not what the page says.
_BOILERPLATE_TAGS = (
    "script", "style", "noscript", "template", "svg",
    "nav", "header", "footer", "aside", "form", "button",
)

# The same furniture on sites that build it out of divs instead of semantic tags, which
# is most commercial sites. Matched on ARIA roles and on class names as whole words, so
# "site-header" and "nav-primary" go and "mainContent" stays.
_CHROME_ROLES = ("navigation", "banner", "contentinfo", "search", "complementary", "menu")
_CHROME_CLASS_PATTERN = re.compile(
    r"(?:^|[\s_-])(?:nav|navbar|navigation|menu|header|masthead|footer|sidebar|breadcrumb"
    r"|cookie|consent|banner|subscribe|newsletter|social|share|skip-link|toolbar)"
    r"(?:[\s_-]|$)",
    re.I,
)

# What a site serves to a client that does not run JavaScript. There is no browser here,
# so this is the page, and storing it would put "please enable JavaScript" on a card as
# though it were a definition. Matched near the top of the extracted text only, so an
# article that happens to discuss JavaScript further down is unaffected.
_SCRIPT_REQUIRED_MARKERS = (
    "enable javascript",
    "javascript is disabled",
    "javascript is required",
    "requires javascript",
    "turn on javascript",
    "displays a fallback because interactive scripts did not run",
    "your browser does not support",
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
# Reviewer-supplied links to anywhere else on the web
#
# Wikipedia is where the pipeline looks by itself, and it is not where every product is
# documented. A discontinued vendor tool often has one authoritative page in the world
# and it is on the vendor's own site. These functions let a reviewer point a card at it.
#
# Same standing as a reviewer-supplied Wikipedia page throughout: the cross-encoder score
# is RECORDED, NOT ENFORCED, and no Gemini call is spent. A human choosing the source is
# the thing the audit layer exists to approximate, and spending quota to second-guess
# them contradicts both the quota rule and the point of the field.
# --------------------------------------------------------------------------

def _is_wikipedia_url(raw_reference: str) -> bool:
    """True for anything the existing Wikipedia path should handle, including a bare title."""
    text = (raw_reference or "").strip()
    if not text:
        return False
    if "://" not in text:
        # A bare page title, which is what the field has always accepted.
        return True
    host = (urlparse(text).hostname or "").lower()
    return host.endswith("wikipedia.org")


def _host_is_blocked(host: str) -> bool:
    """
    True for addresses a reviewer's link has no business reaching.

    This function is the reason server-side fetching of a pasted URL is acceptable at
    all: without it, the review form is a proxy into whatever the server can reach --
    the loopback interface it is serving from, the metadata endpoint on a cloud host,
    anything else on the LAN.

    Resolution happens here and the fetch happens later, so a hostile DNS record could
    in principle change in between. Closing that fully means fetching by pinned IP with
    a Host header, which breaks TLS verification on plenty of ordinary sites. For a
    single-operator tool behind a login, the check is proportionate; if this ever faces
    untrusted users, pin the address.
    """
    import ipaddress
    import socket

    if not host:
        return True

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        # Unresolvable. Not blocked as such, but nothing can be fetched from it either,
        # and reporting it here gives a better sentence than a timeout would.
        return True

    for info in infos:
        address = info[4][0]
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            return True
        if (
            parsed.is_private or parsed.is_loopback or parsed.is_link_local
            or parsed.is_reserved or parsed.is_multicast or parsed.is_unspecified
        ):
            logger.warning("Refusing to fetch %s: resolves to %s.", host, address)
            return True
    return False


def _parse_text_fragment(fragment: str) -> Optional[Dict[str, str]]:
    """
    Reads a `#:~:text=` fragment into its parts.

    The syntax is `[prefix-,]textStart[,textEnd][,-suffix]`, percent-encoded. This is
    what a browser writes when someone highlights a passage and copies a link to it, and
    it is the most useful thing a reviewer can paste: it names not just the page but the
    sentence on it that defines the product.

    Returns None when there is no fragment. Only the first fragment is read; a link can
    carry several, and there is no sensible way to score two disjoint passages as one
    definition.
    """
    if ":~:" not in (fragment or ""):
        return None

    directive = fragment.split(":~:", 1)[1]
    for part in directive.split("&"):
        if not part.startswith("text="):
            continue

        pieces = [unquote(piece) for piece in part[len("text="):].split(",")]
        parsed = {"prefix": "", "start": "", "end": "", "suffix": ""}
        if pieces and pieces[0].endswith("-"):
            parsed["prefix"] = pieces.pop(0)[:-1]
        if pieces and pieces[-1].startswith("-"):
            parsed["suffix"] = pieces.pop()[1:]
        if pieces:
            parsed["start"] = pieces[0]
        if len(pieces) > 1:
            parsed["end"] = pieces[1]
        return parsed if parsed["start"] else None
    return None


# The share of a document's text above which an element is content rather than
# furniture, whatever its class says. Menus, banners and footers are small; a wrapper
# holding a third of the page is the article.
_CHROME_MAX_SHARE = 0.33


def _decompose_if_small(element, total_chars: int) -> None:
    """Removes a chrome-classed element unless it holds most of the page's text."""
    if element.decomposed:
        return
    if total_chars <= 0:
        element.decompose()
        return
    if len(element.get_text(" ", strip=True)) / total_chars <= _CHROME_MAX_SHARE:
        element.decompose()


def _extract_page_text(html: bytes) -> Tuple[str, str, str]:
    """
    Pulls a title, readable body text, and the page's own description out of a document.

    Takes RAW BYTES, not a string, so BeautifulSoup can read the encoding out of the
    document's own meta tag. requests falls back to ISO-8859-1 for any text/* response
    that does not name a charset in its header, which turns every em dash and curly
    quote on a UTF-8 page into mojibake before the parser ever sees it.

    Returns (title, body_text, description). The two texts are returned SEPARATELY
    rather than resolved to one here, because the caller needs both for different jobs:
    a `#:~:text=` fragment has to be matched against the full body, while a card with no
    fragment is far better served by the description. See resolve_external_url.

    Boilerplate tags come out first, then the main content element if the page marks
    one, because a vendor page is mostly navigation and a naive get_text() returns the
    menu.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    title = ""
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        title = og_title["content"].strip()
    elif soup.title and soup.title.string:
        title = soup.title.string.strip()

    # Read BEFORE the tree is stripped: these live in <head>, which survives either way,
    # but reading first means a future change to _BOILERPLATE_TAGS cannot silently take
    # the description with it.
    description = ""
    for finder in (
        lambda: soup.find("meta", property="og:description"),
        lambda: soup.find("meta", attrs={"name": "description"}),
    ):
        meta = finder()
        if meta and meta.get("content"):
            description = " ".join(meta["content"].split()).strip()
            if description:
                break

    for tag in soup(list(_BOILERPLATE_TAGS)):
        tag.decompose()

    # Chrome that is not a semantic tag. _BOILERPLATE_TAGS removes <nav> and <footer>,
    # which does nothing for the many sites that build the same furniture out of divs:
    # bentley.com yielded "All Software / CAD Modeling and Visualization / MicroSta..."
    # with every semantic tag already gone.
    #
    # Guarded by SIZE, not by the class name alone. A class name cannot reliably tell
    # furniture from content -- blender.org wraps its whole article in
    # "type-page ps-about has-header", and any pattern that catches "site-header" also
    # catches "has-header". Chrome is by definition a small part of a page, so an element
    # holding most of the document's text is content wearing a state class, and removing
    # it took blender.org from 3,005 characters to 15.
    total = len(soup.get_text(" ", strip=True))
    for element in soup.find_all(attrs={"role": _CHROME_ROLES}):
        _decompose_if_small(element, total)
    for element in soup.find_all(class_=_CHROME_CLASS_PATTERN):
        _decompose_if_small(element, total)

    # In preference order. role="main" and the id/class names below cover the static
    # site generators that predate <main>, which is most documentation on the web and
    # therefore most of what a reviewer will paste.
    main = (
        soup.find("article")
        or soup.find("main")
        or soup.find(attrs={"role": "main"})
        or soup.find(id=re.compile(r"^(content|main)$", re.I))
        or soup.find(class_=re.compile(r"^(content|main|body)$", re.I))
        or soup.body
        or soup
    )
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", main.get_text("\n", strip=True))
    text = re.sub(r"\n{2,}", "\n", text)

    return title, text, description


def _trim_to_sentence(text: str, limit: int) -> str:
    """
    Cuts to at most `limit` characters, preferring a sentence boundary.

    A hard slice ends mid-word and leaves a fragment that reads as though the source was
    truncated by accident. Falls back to the hard slice when there is no sentence end in
    the back half, rather than returning something much shorter than asked for.
    """
    text = " ".join(text.split()).strip()
    if len(text) <= limit:
        return text

    window = text[:limit]
    cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    if cut >= limit // 2:
        return window[:cut + 1].strip()
    return window.strip()


def _best_definition_text(body: str, description: str) -> str:
    """
    Picks the text most likely to read as a definition, and bounds it.

    The page's OWN description wins when it is long enough. og:description and
    <meta name="description"> are hand-written one-paragraph summaries of exactly the
    shape wanted here, and they are server-rendered even on sites whose body is built by
    JavaScript -- which is most vendor product pages. Preferring them is the single
    change that turns these pages from menu dumps into usable definitions.

    Body text is the fallback, trimmed to a sentence boundary. It was previously the only
    source, capped at 4000 characters, which stored the whole page including its
    navigation and measurably changed how the skill classified.

    A description SHORTER than the floor is not preferred: a 40-character tagline is no
    better than the stub it would replace, and the body may still carry real prose.
    """
    description = " ".join((description or "").split()).strip()
    if len(description) >= MIN_EXTERNAL_EXTRACT_CHARS:
        return _trim_to_sentence(description, MAX_EXTERNAL_EXTRACT_CHARS)

    trimmed_body = _trim_to_sentence(body or "", MAX_EXTERNAL_EXTRACT_CHARS)
    if len(trimmed_body) >= MIN_EXTERNAL_EXTRACT_CHARS:
        return trimmed_body

    # Neither clears the floor on its own. Return whichever is longer and let the caller
    # refuse it as too_thin, so the reason reported is the honest one.
    return trimmed_body if len(trimmed_body) >= len(description) else description


def _passage_for_fragment(text: str, fragment: Dict[str, str]) -> Optional[str]:
    """
    Finds the highlighted passage in the page text, or None.

    Matching is case-insensitive and whitespace-insensitive, because the fragment was
    written from the rendered DOM and this text came from the markup -- the words agree,
    the line breaks do not.
    """
    def flatten(value: str) -> str:
        return re.sub(r"\s+", " ", value or "").strip().lower()

    # One pass builds the searchable string AND a map from each of its characters back
    # to its index in the original, so the passage is returned with its own
    # capitalisation and spacing rather than the flattened copy used for searching.
    chars, positions = [], []
    previous_space = True
    for index, char in enumerate(text):
        if char.isspace():
            if previous_space:
                continue
            chars.append(" ")
            previous_space = True
        else:
            chars.append(char.lower())
            previous_space = False
        positions.append(index)
    if chars and chars[-1] == " ":
        chars.pop()
        positions.pop()

    flat = "".join(chars)
    if not flat:
        return None

    start_needle = flatten(fragment.get("start"))
    if not start_needle:
        return None

    search_from = 0
    prefix = flatten(fragment.get("prefix"))
    if prefix:
        anchor = flat.find(prefix)
        if anchor < 0:
            return None
        search_from = anchor + len(prefix)

    begin = flat.find(start_needle, search_from)
    if begin < 0:
        return None

    end_needle = flatten(fragment.get("end"))
    if end_needle:
        tail = flat.find(end_needle, begin + len(start_needle))
        if tail < 0:
            return None
        finish = tail + len(end_needle)
    else:
        finish = begin + len(start_needle)

    return text[positions[begin]:positions[finish - 1] + 1].strip()


def resolve_external_url(
    skill_name: str, category_title: str, raw_reference: str
) -> Dict[str, Any]:
    """
    Resolves a reference page a reviewer supplied, anywhere on the web.

    Same return shape as resolve_reviewer_url, and Wikipedia links are handed straight
    to it so title validation and disambiguation detection are not reimplemented here.

    For anything else: fetch, strip the page to its readable text, and score that text
    exactly as a Wikipedia extract would be scored -- recorded, not enforced. If the URL
    carries a `#:~:text=` fragment, the highlighted passage IS the definition, and a
    fragment that cannot be found is an error rather than a quiet fall back to the whole
    page: the reviewer chose those words on purpose.

    No Gemini call, matching resolve_reviewer_url. `source_name` becomes the host, so a
    card sourced from a vendor site says so.
    """
    failure = {"title": None, "extract": "", "score": None, "source_name": None, "url": None}

    reference = (raw_reference or "").strip()
    if not reference:
        return {**failure, "error": "empty"}
    if len(reference) > MAX_REFERENCE_URL_CHARS:
        return {**failure, "error": "too_long"}

    if _is_wikipedia_url(reference):
        resolved = resolve_reviewer_url(skill_name, category_title, reference)
        # Wikipedia pages are stored by title, as they always have been, so no url is
        # recorded for them and the card keeps building the en.wikipedia.org link.
        return {**resolved, "url": None}

    parsed = urlparse(reference)
    if parsed.scheme not in ("http", "https"):
        return {**failure, "error": "bad_scheme"}
    if _host_is_blocked(parsed.hostname or ""):
        return {**failure, "error": "blocked_host"}

    response = request_with_backoff(
        reference,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/plain;q=0.9"},
        timeout=EXTERNAL_TIMEOUT,
    )
    if response is None:
        return {**failure, "error": "fetch_failed"}

    content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if content_type and not content_type.startswith(EXTERNAL_CONTENT_TYPES):
        logger.info("Refusing %s for %r: content type %s.", reference, skill_name, content_type)
        return {**failure, "error": "not_html"}

    title, text, description = _extract_page_text(response.content[:MAX_EXTERNAL_BYTES])
    if not text.strip() and not description.strip():
        return {**failure, "error": "no_text"}

    # Scanned over the WHOLE extract, not the first 300 characters. Kafka's documentation
    # page and salesforce.com both put their real content behind script and their notice
    # below that window, so a head-only check passed a redirect stub as a definition.
    if any(marker in text.lower() for marker in _SCRIPT_REQUIRED_MARKERS):
        logger.info("%s serves a script-required page to a plain client.", reference)
        return {**failure, "error": "needs_javascript"}

    fragment = _parse_text_fragment(parsed.fragment)
    if fragment:
        # Matched against the FULL body. The reviewer highlighted words they saw on the
        # page, which is not necessarily the description.
        passage = _passage_for_fragment(text, fragment)
        if not passage:
            logger.info(
                "Highlighted passage not found on %s for %r.", reference, skill_name
            )
            return {**failure, "error": "fragment_not_found"}
        extract = passage
    else:
        extract = _best_definition_text(text, description)

    extract = extract.strip()
    if not extract:
        return {**failure, "error": "no_text"}
    # The floor does NOT apply to a highlighted passage. A reviewer who selected one
    # sentence chose those words deliberately, and this is the same principle that makes
    # the cross-encoder score recorded rather than enforced on reviewer-supplied paths:
    # a human asserting the text is the thing the automation is guessing at. Refusing a
    # short highlight told a reviewer their own selection was too thin.
    if not fragment and len(extract) < MIN_EXTERNAL_EXTRACT_CHARS:
        # A navigation stub, a JS redirect shim, or a bot wall that answered 200. Refusing
        # is the point: this used to be stored as the card's definition, and kafka.apache
        # .org's 12-character "Documentation Redirect" scored 0.996 doing it.
        logger.info(
            "Only %d characters of usable text on %s for %r; refusing it.",
            len(extract), reference, skill_name,
        )
        return {**failure, "error": "too_thin"}

    host = (parsed.hostname or "").lower()
    display_title = title or host

    query = f"{clean_skill_name(normalize_skill_name(skill_name))} ({derive_entity_suffix(category_title)})"
    score = score_pair(query, display_title, extract)

    logger.info(
        "Reviewer supplied %s for %r (%s, score=%s, %d chars).",
        host, skill_name, "highlighted passage" if fragment else "page text",
        "none" if score is None else f"{score:.4f}", len(extract),
    )
    return {
        "title": display_title,
        "extract": extract,
        "score": score,
        "source_name": host,
        "url": reference,
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

def resolve_and_gate_skill(
    skill_item: Dict[str, Any],
    cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Resolves a page and applies the cross-encoder gate. NO Gemini call, ever.

    Returns either a finished decision record (`needs_audit` False), for the two
    outcomes that never reach the audit layer, or a pending record (`needs_audit` True)
    carrying what the audit needs: skill_name, title, extract, score and source_name.

    Split out of scrape_and_validate_skill so a whole occupation can be resolved first
    and then audited in batches of AUDIT_BATCH_SIZE. The resolve leg is Wikipedia and local
    ONNX, both unmetered; the audit leg is the one the free-tier quota counts, and
    interleaving them one skill at a time is what made a run cost one request per skill.

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

    # No usable candidate, either because nothing resolved or because what resolved is
    # not about this subject. Both queue the skill with taxonomy context so a human --
    # or the second pass, which drafts a definition for exactly this bucket -- can fix
    # it. The junk candidate is discarded rather than recorded: presenting an unrelated
    # article as this skill's reference page, with that article's text as its
    # definition, is worse than admitting nothing was found.
    if raw_extract and cross_score is not None and cross_score < RESOLUTION_FLOOR:
        logger.info(
            "Best candidate for %r was %r at %.4f, below the %.2f floor. Discarding it; "
            "no reference page.",
            skill_name, candidate["title"], cross_score, RESOLUTION_FLOOR,
        )
        raw_extract = ""
        cross_score = None
        # The title goes too. The branch below reports candidate["title"] as the
        # resolved page, and a page that was rejected for not being about this subject
        # must not be recorded as this subject's reference.
        #
        # Safe to mutate: the cache entry above was built from these values as its own
        # dict, so the resolve result stays cached and only this decision drops it.
        candidate["title"] = None

    if not raw_extract or cross_score is None:
        logger.info("No scoreable candidate for %r. Routing to manual review.", skill_name)
        return {
            "skill_name": skill_name,
            "needs_audit": False,
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
            "skill_name": skill_name,
            "needs_audit": False,
            "resolved_title": candidate["title"],
            "raw_extract": raw_extract,
            "summary": raw_extract,
            "cross_score": cross_score,
            "is_credible": None,
            "best_source_name": None,
            "auto_approve": False,
            "gate_reason": "below_relevance_threshold",
        }

    # Page looks correct. The audit is the caller's to run, one skill or six at a time.
    return {
        "skill_name": skill_name,
        "needs_audit": True,
        "resolved_title": candidate["title"],
        "raw_extract": raw_extract,
        "cross_score": cross_score,
        "source_name": candidate["source_name"],
    }


def apply_audit_result(
    gated: Dict[str, Any], evaluation: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Turns one audit verdict into the final decision record. PURE: no IO, no network.

    `evaluation` is None when the audit could not run at all -- quota exhausted, the API
    down, or the skill's id missing from a batched answer. That records is_credible NULL
    and gate_reason audit_unavailable, NEVER False: an outage must not be filed as a
    judgement about the source.
    """
    skill_name = gated["skill_name"]
    cross_score = gated["cross_score"]
    raw_extract = gated["raw_extract"]

    if evaluation is None:
        # The page pull was good, so keep the raw extract and send it to a human.
        logger.warning(
            "Credibility audit unavailable for %r (cross=%.4f). Manual review.",
            skill_name, cross_score,
        )
        return {
            "skill_name": skill_name,
            "needs_audit": False,
            "resolved_title": gated["resolved_title"],
            "raw_extract": raw_extract,
            "summary": raw_extract,
            "cross_score": cross_score,
            "is_credible": None,
            "best_source_name": gated.get("source_name"),
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
            "skill_name": skill_name,
            "needs_audit": False,
            "resolved_title": gated["resolved_title"],
            "raw_extract": raw_extract,
            "summary": clean_summary,
            "cross_score": cross_score,
            "is_credible": True,
            "best_source_name": evaluation.get("best_source_name") or gated.get("source_name"),
            "auto_approve": True,
            "gate_reason": "auto_approved",
        }

    logger.info(
        "Gemini rejected credibility for %r despite cross=%.4f. Manual review.",
        skill_name, cross_score,
    )
    return {
        "skill_name": skill_name,
        "needs_audit": False,
        "resolved_title": gated["resolved_title"],
        "raw_extract": raw_extract,
        "summary": clean_summary or raw_extract,
        "cross_score": cross_score,
        "is_credible": False,
        "best_source_name": evaluation.get("best_source_name"),
        "auto_approve": False,
        "gate_reason": "failed_credibility_audit",
    }


def scrape_and_validate_skill(
    skill_item: Dict[str, Any],
    cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Resolves, gates, and (conditionally) audits a single skill. One Gemini call at most.

    Kept as the one-skill entry point. `auto_approve` is True only when the
    cross-encoder cleared the threshold AND Gemini confirmed credibility. `is_credible`
    is None whenever Gemini was never called, which keeps "judged not credible" distinct
    from "never judged".
    """
    gated = resolve_and_gate_skill(skill_item, cache=cache)
    if not gated["needs_audit"]:
        return gated

    try:
        evaluation = get_source_checker(ROLE_AUDIT).evaluate_candidates(
            skill_name=gated["skill_name"],
            candidates=[{
                "source_name": gated.get("source_name") or "Wikipedia",
                "raw_text": gated["raw_extract"],
            }],
        )
    except AuditUnavailable:
        evaluation = None

    return apply_audit_result(gated, evaluation)


def scrape_and_validate_batch(
    skill_items: List[Dict[str, Any]],
    cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Resolves and audits several skills, spending ONE Gemini request per AUDIT_BATCH_SIZE of
    them rather than one per skill. Returns {skill_name: decision}.

    Every skill is resolved and gated first, because the gate is free and decides who
    needs an audit at all: the below-threshold and no-candidate outcomes never cost a
    request, exactly as before. Only the survivors are packed into batches.

    DailyQuotaExhausted propagates, as it always has. What changes is granularity: the
    decisions already returned from earlier batches are the caller's to keep, and the
    batch in flight when the wall was hit is lost and re-derived on the next run. The
    resolve work behind it is cached, so re-deriving it is cheap.
    """
    from agentic_source_check import AUDIT_BATCH_SIZE

    decisions: Dict[str, Dict[str, Any]] = {}
    awaiting = []

    for item in skill_items:
        gated = resolve_and_gate_skill(item, cache=cache)
        if gated["needs_audit"]:
            awaiting.append(gated)
        else:
            decisions[gated["skill_name"]] = gated

    if not awaiting:
        return decisions

    checker = get_source_checker(ROLE_AUDIT)
    for start in range(0, len(awaiting), AUDIT_BATCH_SIZE):
        chunk = awaiting[start:start + AUDIT_BATCH_SIZE]
        try:
            # item_id is the skill name. It is unique across the store by construction
            # (it is the master dict's key) and it is what the caller maps results back
            # onto, so a separate synthetic id would only add a lookup table to get wrong.
            evaluations = checker.evaluate_candidates_batch([
                {
                    "item_id": gated["skill_name"],
                    "skill_name": gated["skill_name"],
                    "source_name": gated.get("source_name") or "Wikipedia",
                    "raw_text": gated["raw_extract"],
                }
                for gated in chunk
            ])
        except AuditUnavailable:
            # The whole chunk is unaudited. Not a verdict on any of them.
            evaluations = {}

        for gated in chunk:
            decisions[gated["skill_name"]] = apply_audit_result(
                gated, evaluations.get(gated["skill_name"])
            )

    return decisions
