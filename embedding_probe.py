"""
STAGE: Acquisition

Establishes whether a product actually ships AI features, by searching for it.

    from embedding_probe import search_embedding_evidence
    found = search_embedding_evidence("Adobe Photoshop Lightroom")

WHY THIS IS NOT SCORED. sortingalgorithmnew used to answer this with a cosine similarity
against an anchor whose vocabulary -- workspace, workflow, integration, automated,
interface -- appears in the prose of any design or productivity tool. Measured across the
live store it promoted LaTeX, Thomson EndNote, SofTech CADRA and Transoft AutoTURN into a
class called "Non-Technical Embedded AI". The definition text does not carry the fact,
largely because most of those definitions were written before the features existed. So
the fact is looked up instead of inferred.

THREE STATES, AND WHY THE THIRD ONE EXISTS. Every function here returns "no evidence
available" as something distinguishable from "evidence says no". DuckDuckGo blocks
aggressively -- measured from one machine: six queries inside ten seconds and every one
came back 202 with an empty result set, and it stayed blocked for about two minutes. If a
block read as False, an afternoon of rate limiting would silently strip the boost from
every skill it touched and look exactly like a finding.

WHAT THE PAYLOAD IS ACTUALLY LIKE. Good for a true positive:

    does Adobe Photoshop Lightroom embed AI
      Adobe Delivers New AI Innovations ... including Photoshop, Lightroom
      Enhance photos using generative AI | Lightroom - helpx.adobe.com

and actively misleading for a true negative:

    does LaTeX embed AI
      AI features - Overleaf, Online LaTeX Editor
      Prism | AI-native workspace ... integrates ChatGPT and Codex
      AI LaTeX Editor - Write LaTeX with AI - Underleaf

Every LaTeX result argues yes. The correct answer is no: LaTeX is a typesetting system
from 1984 and those are third-party editors built around it. This module deliberately
does NOT try to solve that with query tuning or result filtering -- the distinction needs
to know what the product is, so it belongs to the grader, which is given the skill's
stored definition alongside these snippets. See AgenticSourceChecker.grade_embedding_batch.
"""

import logging
import os
import re
import threading
import time
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

# Read from the environment rather than imported from scraping, which would make this
# module depend on the whole Wikipedia resolver -- and, since scraping imports
# agentic_source_check and agentic_source_check imports the formatter below, would close
# an import cycle. The value is the same one scraping reads.
USER_AGENT = os.getenv("USER_AGENT", "AISkillsAnalyticsDashboard/1.0")

SEARCH_URL = "https://html.duckduckgo.com/html/"

# MEASURED, not guessed, and the whole reason a full pass takes hours:
#
#     6 queries in ~10s        -> 202 with zero results, on every one
#       still blocked at 6s and at 12s spacing, 5 of 5 failed each time
#       recovered after ~120s idle
#     8 queries at 25s spacing -> 200 with 10 results, 8 of 8
#
# 25 seconds is the cadence that holds. Lower it and the pass does not run faster, it
# runs to completion returning `unknown` for everything, which is worse than slow because
# it looks like progress. Re-measure before changing it.
POLITENESS_DELAY = 25.0

TIMEOUT = 15.0
MAX_RESULTS = 8

# Snippets are the whole payload, so they are kept longer than a page extract would be,
# but bounded: eight of these go into a batched prompt alongside seven other skills.
MAX_SNIPPET_CHARS = 320

# DuckDuckGo answers a rate-limited client with 202 and a page that is structurally a
# result page with nothing in it. It is not an error status and requests will not raise
# on it, so it is checked explicitly -- this is the single most likely failure in a long
# run and the one that must never be mistaken for "no AI features found".
CHALLENGE_STATUS = 202

# A DuckDuckGo result link wraps the destination in a redirector. The real host is the
# useful part for a grader deciding whether a source is the vendor or a blog.
_REDIRECT_HOSTS = ("duckduckgo.com", "html.duckduckgo.com", "lite.duckduckgo.com")

_WHITESPACE = re.compile(r"\s+")

# Errors, so a caller can tell a block from a genuinely empty search.
ERROR_BLOCKED = "blocked"
ERROR_FETCH_FAILED = "fetch_failed"
ERROR_NO_RESULTS = "no_results"
ERROR_EMPTY_QUERY = "empty_query"

# Serialises the delay across threads. The pass is single-threaded today; this is here so
# that a future caller running two probes in parallel gets the delay rather than the ban.
_lock = threading.Lock()
_last_request_at = 0.0


def build_query(skill_name: str) -> str:
    """The search that gets asked. One phrasing, so results stay comparable."""
    return f"does {skill_name} embed AI"


def _wait_for_slot() -> None:
    """
    Blocks until POLITENESS_DELAY has passed since the last request.

    Enforced HERE rather than in the caller on purpose. A caller that forgets the sleep
    does not get slightly worse results, it gets banned and a run's worth of `unknown`.
    """
    global _last_request_at
    with _lock:
        elapsed = time.monotonic() - _last_request_at
        if _last_request_at and elapsed < POLITENESS_DELAY:
            time.sleep(POLITENESS_DELAY - elapsed)
        _last_request_at = time.monotonic()


def _clean(text: str) -> str:
    return _WHITESPACE.sub(" ", (text or "").strip())


def _destination(href: str) -> str:
    """
    Unwraps a DuckDuckGo redirect to the URL it points at.

    A grader shown "duckduckgo.com/l/?uddg=..." for every result cannot tell helpx.adobe
    .com from a listicle, which is exactly the judgement it is being asked to make.
    """
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if parsed.hostname and parsed.hostname.lower().lstrip("www.") in _REDIRECT_HOSTS:
        target = parse_qs(parsed.query).get("uddg")
        if target:
            return target[0]
    return href


def parse_results(html: bytes) -> List[Dict[str, str]]:
    """
    Pulls (title, snippet, url) out of a DuckDuckGo HTML result page.

    Separate from the fetch so the LaTeX page captured in the tests can be replayed
    offline. The parser is the part most likely to break when DuckDuckGo reshuffles its
    markup, and it must not need the network to prove it still works.
    """
    soup = BeautifulSoup(html, "html.parser")
    results: List[Dict[str, str]] = []

    for block in soup.select(".result, .web-result"):
        anchor = block.select_one(".result__a")
        if not anchor:
            continue
        snippet = block.select_one(".result__snippet")
        title = _clean(anchor.get_text(" ", strip=True))
        if not title:
            continue
        results.append({
            "title": title[:MAX_SNIPPET_CHARS],
            "snippet": _clean(snippet.get_text(" ", strip=True))[:MAX_SNIPPET_CHARS]
                       if snippet else "",
            "url": _destination(anchor.get("href") or ""),
        })
        if len(results) >= MAX_RESULTS:
            break

    return results


def search_embedding_evidence(skill_name: str) -> Dict[str, Any]:
    """
    Runs one search and returns what it found, or WHY it found nothing.

        {"query": str, "results": [...], "error": None | str}

    A non-empty `error` means the answer is `unknown`, never False. Callers must not
    collapse the two: `no_results` and `blocked` both produce an empty list, and only one
    of them is a statement about the product.

    Blocks for up to POLITENESS_DELAY before issuing the request.
    """
    name = (skill_name or "").strip()
    if not name:
        return {"query": "", "results": [], "error": ERROR_EMPTY_QUERY}

    query = build_query(name)
    _wait_for_slot()

    try:
        response = requests.get(
            SEARCH_URL,
            params={"q": query},
            headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
            timeout=TIMEOUT,
        )
    except requests.RequestException as error:
        logger.warning("Search for %r failed: %s", name, error)
        return {"query": query, "results": [], "error": ERROR_FETCH_FAILED}

    if response.status_code == CHALLENGE_STATUS:
        logger.warning(
            "DuckDuckGo is rate limiting (202) on %r. Nothing is concluded about this "
            "skill; it stays unknown and will be retried.", name,
        )
        return {"query": query, "results": [], "error": ERROR_BLOCKED}

    if response.status_code != 200:
        logger.warning("Search for %r returned HTTP %s.", name, response.status_code)
        return {"query": query, "results": [], "error": ERROR_FETCH_FAILED}

    results = parse_results(response.content)
    if not results:
        # A 200 with no parseable results is ALSO unknown rather than a negative. It is
        # far more likely that the markup moved than that the web has nothing to say
        # about a product O*NET lists, and a silent parser break would otherwise mark
        # every skill in the store as not embedding AI.
        logger.warning(
            "Search for %r returned 200 with no parseable results. Treating as unknown; "
            "if this repeats, the result markup has moved and parse_results needs it.",
            name,
        )
        return {"query": query, "results": [], "error": ERROR_NO_RESULTS}

    logger.info("Search for %r returned %d result(s).", name, len(results))
    return {"query": query, "results": results, "error": None}


def format_results(results: List[Dict[str, str]]) -> str:
    """Renders search results as the numbered block a grading prompt receives."""
    lines = []
    for index, item in enumerate(results, 1):
        host = urlparse(item.get("url") or "").hostname or "unknown source"
        lines.append(f"  {index}. [{host}] {item.get('title', '')}")
        if item.get("snippet"):
            lines.append(f"     {item['snippet']}")
    return "\n".join(lines) if lines else "  (no results)"
