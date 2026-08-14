"""
FastAPI HITL review server.

Running it:

    cd /Users/keirawalker/SkillDashboard
    python3.11 main.py

    Review queue:  http://127.0.0.1:8000/
    Dashboard:     http://127.0.0.1:8000/dashboard

Two constraints that are invisible from the code and have both cost real time:

  - It must be `python3.11`. The default `python3` on this machine is 3.9.6 and has
    neither fastapi nor google-genai installed, so it fails at import.
  - It must run from the workspace directory. The StaticFiles mount and the Jinja2
    templates directory below are relative paths, so starting elsewhere breaks both.

HOST and PORT override the defaults. Add --reload only while editing, which needs the
explicit form: `python3.11 -m uvicorn main:app --reload`.

Storage is the local JSON store (skills_master.json, skills_timeseries.json), not SQL.
Reads go through dashboardtables and json_store; every write goes through
review_actions, which is the only module allowed to change a skill's review status.

Endpoint handlers are deliberately synchronous (`def`, not `async def`). Every one of
them performs blocking work -- file IO, ONNX inference, and in the ingest case network
scraping and Gemini calls. Declared async, they would run on the event loop and freeze
every other request for the duration. As plain `def`, FastAPI runs them in its
threadpool. Ingestion additionally runs as a BackgroundTask so the browser is not held
open for the length of a full occupation pull.
"""

import datetime
import json
import logging
import os
import re
import threading
from collections import Counter
from contextlib import asynccontextmanager
from urllib.parse import quote

from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, Form, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup
from pydantic import BaseModel, Field

import auth
import backlog as backlog_store
import run_state
from dashboardtables import (
    PRIORITY_ACTIONABLE,
    PRIORITY_NEEDS_HUMAN,
    PRIORITY_REPORTED,
    build_dashboard_rows,
    build_review_rows,
    load_dashboard,
)
from json_store import (
    STATUS_APPROVED,
    STATUS_PENDING,
    STATUS_REJECTED,
    latest_snapshots,
    load_master,
    load_timeseries,
    save_master,
    save_timeseries,
    upsert_snapshot,
)
from review_actions import (
    MAX_BATCH_DECISIONS,
    REPORT_REASONS,
    apply_machine_draft,
    apply_machine_suggestion,
    apply_review_batch,
    approve_skill,
    edit_approved_skill,
    mark_duplicate,
    reject_draft,
    reject_skill,
    remediate_skill,
    report_skill,
)
from scraping import (
    CROSS_ENCODER_THRESHOLD,
    MAX_REFERENCE_URL_CHARS,
    REMEDIATION_ERROR_CODES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Reads the store once at startup so a missing or damaged file is reported here
    rather than as a 500 on the first request.

    A missing store is normal on a fresh checkout and says so; a corrupt one is
    CRITICAL, because with no database there is no second copy to fall back to.

    Also settles two things that a restart would otherwise leave lying: a run record
    still claiming to be running (nothing is, so it is marked interrupted), and the
    configuration mistakes worth complaining about once rather than on discovery.
    """
    try:
        master = load_master()
        if master:
            logger.info("Local JSON store loaded: %d skills.", len(master))
        else:
            logger.warning(
                "The JSON store is empty. Run an ingestion first: python3.11 ingest.py 15-"
            )
    except Exception:
        logger.critical("The JSON store could not be read.", exc_info=True)

    # In-memory run state does not survive a restart, so a record still saying
    # "running" means the server died mid-run. Nothing is lost -- the backlog holds
    # every occupation that had not completed -- but the panel must not claim a run is
    # in progress when no thread exists to do it.
    try:
        run_state.reconcile_on_startup()
    except Exception:
        logger.exception("Could not reconcile the ingestion run state at startup.")

    auth.startup_warnings(os.getenv("HOST", "127.0.0.1"))
    yield


app = FastAPI(
    title="AI Skills Analytics HITL Review Engine",
    description="Server-rendered HITL validation interface over the local JSON store",
    version="4.0.0",
    lifespan=lifespan,
)

class RevalidatedStatics(StaticFiles):
    """
    Serves /static, telling browsers to check with us before reusing anything.

    Plain StaticFiles sends an ETag and a Last-Modified and NO Cache-Control, which
    leaves a browser free to apply its own heuristic and reuse a file for hours without
    asking. That cost real time: an edited review.js kept being cached client-side while
    the server was serving the new one byte for byte, so a feature that worked everywhere
    it was tested did not work in the browser looking at it.

    `no-cache` does not mean "do not cache". It means "revalidate before reusing", and
    with the ETag already being sent that revalidation is a 304 with no body. The pairing
    with the ?v= stamp below is belt and braces: the stamp changes the URL when a file
    changes, and this stops a stale copy being reused under the old URL in the meantime.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


# Resolved against THIS FILE, not the working directory. static/ and templates/ ship
# beside the code and never move, so a relative "static" only worked when the process
# happened to start in the project root -- importing main from anywhere else raised
# "Directory 'static' does not exist" at import time. Same reasoning as storage.MODEL_DIR.
_HERE = os.path.dirname(os.path.abspath(__file__))

app.mount("/static", RevalidatedStatics(directory=os.path.join(_HERE, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(_HERE, "templates"))


def asset_version(name: str) -> str:
    """
    A short stamp for a static file, changing whenever the file does.

    Appended to asset URLs so a changed file is a changed URL, which is the only thing
    that reliably defeats a cache that has already decided it knows what /static/x.js
    contains. One stat() per render, which is noise next to reading the store.
    """
    try:
        # _HERE, not a relative "static": started from another directory this silently
        # returned "0" for every asset, which is a constant, which is no cache busting
        # at all -- the failure mode being a stale asset rather than an error is exactly
        # why it is worth resolving properly.
        stat = os.stat(os.path.join(_HERE, "static", name))
    except OSError:
        return "0"
    return f"{int(stat.st_mtime)}-{stat.st_size}"


# Exposed to every template, so a new asset never needs a new context key.
templates.env.globals["asset_version"] = asset_version


def _no_stale_html(response):
    """
    Stops a browser reusing a rendered page without asking.

    Every HTML page here is a view of a store that changes underneath it: the queue after
    a commit, the dashboard after a scrape. A cached copy is a page showing work that has
    already been done, which is indistinguishable from work that did not happen. Static
    assets are handled separately, by RevalidatedStatics and the ?v= stamp.
    """
    if "text/html" in (response.headers.get("content-type") or ""):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.middleware("http")
async def require_login(request: Request, call_next):
    """
    Gates every route except the public dashboard and its assets.

    A GET redirects to the login form carrying where it was headed; a POST answers 401
    rather than redirecting, because a form post that silently becomes a login page
    looks to the caller like the action succeeded.

    Registered as middleware rather than a per-route dependency on purpose: a
    dependency has to be remembered on every new route, and the one that gets forgotten
    is the one that mutates the store.
    """
    if auth.is_public(request.url.path):
        return _no_stale_html(await call_next(request))

    if auth.valid(request.cookies.get(auth.COOKIE_NAME)):
        return _no_stale_html(await call_next(request))

    if request.method != "GET":
        return JSONResponse({"ok": False, "code": "auth"}, status_code=401)

    target = quote(auth.safe_next(request.url.path))
    return RedirectResponse(url=f"/login?next={target}", status_code=303)

PAGE_SIZE = 25

GATE_REASON_LABELS = {
    "below_relevance_threshold": "Page match below threshold",
    "failed_credibility_audit": "Failed credibility audit",
    "no_candidate_found": "No reference page found",
    "audit_unavailable": "Credibility audit did not run",
    "auto_approved": "Auto-approved",
    "reviewer_remediated": "Reviewer-supplied page",
    "machine_remediated": "Machine-suggested page",
    "reported_by_viewer": "Reported from the dashboard",
    # Text with no external source that reached the dashboard on two agreeing calls
    # rather than a click. Named plainly so nobody has to know what "model_authored"
    # means to understand that no source stands behind it.
    "model_authored_verified": "Written by the model, no source, verified",
}

# Sentences for the post-mortem edit form, same fixed-table pattern as remediation.
EDIT_ERRORS = {
    "no_summary": "The definition cannot be empty. Type the corrected text, or leave the "
                  "skill as it is.",
    "bad_reference": "That reference is not a resolvable English Wikipedia page. Paste an "
                     "en.wikipedia.org article link, type the page title, or clear the box "
                     "to leave the recorded page unchanged.",
    "not_found": "That skill no longer exists.",
    "not_approved": "That skill is not approved, so there is no snapshot to correct. Handle "
                    "it through the pending queue instead.",
    "no_snapshot": "That skill has never been scored, so there is nothing to correct.",
    "error": "The edit could not be saved. See the server log for details.",
}

EDIT_OK_MESSAGE = ("Saved. The latest snapshot was rescored and marked as an override; the "
                   "trend history was not duplicated.")

# Reviewer-facing sentences for remediation failures. Keyed by the codes
# review_actions.remediate_skill returns; anything unrecognized falls back to the generic
# sentence, so a hand-edited ?err= value can never put attacker text on the page.
REMEDIATION_ERRORS = {
    "empty": "Enter a Wikipedia URL or page title.",
    "bad_url": "That does not look like a Wikipedia article URL. Use a link of the form "
               "https://en.wikipedia.org/wiki/Page_Title, or just type the page title.",
    "not_wikipedia": "Only English Wikipedia pages can be fetched. Paste an "
                     "en.wikipedia.org article link, or type the page title.",
    "no_page": "No Wikipedia page by that name exists. Check the spelling or open the "
               "page in a browser and copy its URL.",
    "disambiguation": "That page is a disambiguation list rather than an article. Pick "
                      "the specific page it links to.",
    "no_text": "That page exists but returned no summary text to score.",
    "bad_scheme": "Only http and https links can be fetched. A file:// path or a "
                  "javascript: link is not a reference.",
    "blocked_host": "That address is on this machine or on a private network, so it is "
                    "not a source anyone else could check. Use a public URL.",
    "too_long": "That link is too long to store. Copy the link to the highlight again, "
                "selecting a shorter passage.",
    "fetch_failed": "That page could not be fetched. Check the link opens in a browser, "
                    "and note that sites behind a login or a bot check cannot be read "
                    "from here.",
    "not_html": "That link is a file rather than a web page. Link to a page that "
                "describes the product.",
    "needs_javascript": "That site only builds its text in a browser, so there is "
                        "nothing to read from here. Link to a documentation or "
                        "reference page that works with JavaScript off, or type the "
                        "definition in yourself.",
    "too_thin": "That page had almost no readable text on it -- usually a menu, a "
                "redirect stub, or a bot check that answered with an empty shell. What "
                "little was there would have become the definition, so it was refused. "
                "Link to a page that describes the product in prose, or type the "
                "definition in yourself.",
    "fragment_not_found": "The highlighted passage was not found on the page. The site "
                          "may render its text with JavaScript, or the page may have "
                          "changed. Paste the plain link instead, or type the "
                          "definition in yourself.",
    "not_pending": "That item is no longer pending, so it was not changed.",
    "not_found": "That skill is no longer in the store.",
    "no_suggestion": "There is no stored suggestion for that item any more. Re-run the "
                     "second pass, or supply a page yourself.",
    "no_draft": "There is no stored draft definition for that item any more. Re-run the "
                "second pass, or write the definition yourself.",
    "error": "The page could not be fetched. See the server log for details.",
}

GENERIC_REMEDIATION_ERROR = "The reference page could not be used."

# Login failures, looked up in a fixed table like every other reflected code here, so a
# hand-edited ?err= cannot put chosen text on the page.
LOGIN_ERRORS = {
    "bad_password": "That password is not right.",
}

# Every code scraping can produce must have a sentence here. A missing one would
# silently degrade to the generic message, which is a worse reviewer experience than a
# loud complaint at import time.
_missing_messages = set(REMEDIATION_ERROR_CODES) - set(REMEDIATION_ERRORS)
if _missing_messages:
    logger.error(
        "No reviewer message defined for remediation error codes: %s.",
        ", ".join(sorted(_missing_messages)),
    )


def _parse_page(raw) -> int:
    """
    Coerces any page input to a positive integer, never raising.

    Declaring `page: int` on the handler would return a 422 JSON body for ?page=abc, and
    Query(ge=1) would 422 on ?page=0. That is the wrong failure mode for a no-JavaScript
    review tool reached by hand-edited URLs and the browser back button, so bad input
    lands on page 1 instead.

    It also bounds the value before it is reflected into a redirect Location header,
    which removes any header-injection or open-redirect surface.
    """
    try:
        page = int(str(raw).strip())
    except (TypeError, ValueError):
        return 1
    return page if page > 0 else 1


def _page_window(page: int, total_pages: int, span: int = 2) -> list:
    """
    Page numbers to show, with None marking an elided run.

    Computed here rather than in the template so the template stays declarative.
    """
    if total_pages <= 1:
        return [1]

    wanted = {1, total_pages}
    wanted.update(range(max(1, page - span), min(total_pages, page + span) + 1))

    window = []
    previous = 0
    for number in sorted(wanted):
        if previous and number > previous + 1:
            window.append(None)
        window.append(number)
        previous = number
    return window


def queue_skill_names() -> list:
    """
    Names in the actionable review queue, to autocomplete the duplicate-merge box.

    The QUEUE, not every pending entry. Measured on the live store: 185 queued names is
    a 9 KB datalist, while all 3,984 pending entries is 202 KB added to every load of
    this page. The queue is also where duplicates actually surface, because that is what
    a reviewer is looking at -- both halves of the MicroSurvey CAD pair are in it.

    This is a CONVENIENCE, not a constraint. The input is free text and the server
    validates the name against the whole of master, so merging into an entry the queue
    filters out still works; it just has to be typed in full.

    Names only. The page already carries a page of full records, and adding hundreds
    more to power an autocomplete would be a much larger payload for a smaller feature.
    """
    try:
        return sorted(
            row["skill_name"] for row in build_review_rows(load_master(), STATUS_PENDING)
        )
    except Exception:
        logger.exception("Could not build the queue-name list; merging by hand still works.")
        return []


def fetch_pending_page(page: int) -> dict:
    """
    One page of the review queue, read from the JSON store.

    dashboardtables.build_review_rows already applies the rule that matters: an entry
    is only in the queue if a human can act on it, which excludes the several hundred
    tools that were mapped from O*NET and deliberately never audited. Paging over that
    list is then a slice.

    The SQL version needed a QueueID tie-break because created_at was not unique and
    each page was a separate query that could order tied rows differently. Here the
    whole list is sorted once, in-process, by a unique key (skill name), so the
    ordering is total by construction and a row cannot appear on two pages.
    """
    empty = {
        "items": [], "page": 1, "total_pages": 1, "total_pending": 0,
        "first_index": 0, "last_index": 0, "window": [1],
        "band_actionable": 0, "band_reported": 0, "band_needs_human": 0,
    }
    try:
        rows = build_review_rows(load_master(), STATUS_PENDING)
        total_pending = len(rows)

        # Counted over the whole queue, not the page, so the header can say how much of
        # the cheap work is left rather than how much of it is on screen.
        bands = Counter(row["priority"] for row in rows)

        # max(1, ...) so an empty queue is "page 1 of 1" rather than "page 1 of 0".
        total_pages = max(1, -(-total_pending // PAGE_SIZE))
        page = min(max(page, 1), total_pages)
        offset = (page - 1) * PAGE_SIZE
        items = rows[offset:offset + PAGE_SIZE]

        # The card's only derived field. The flattened O*NET code and title that used to
        # be built here went with the card's O*NET row: the occupation a skill was
        # discovered through is not something a reviewer judges the definition on, and
        # it is still on the entry in the store for anything that does need it.
        for item in items:
            item["gate_label"] = GATE_REASON_LABELS.get(
                item.get("gate_reason"), item.get("gate_reason") or "Flagged"
            )

        return {
            "items": items,
            "page": page,
            "total_pages": total_pages,
            "total_pending": total_pending,
            "first_index": offset + 1 if items else 0,
            "last_index": offset + len(items),
            "window": _page_window(page, total_pages),
            "band_actionable": bands.get(PRIORITY_ACTIONABLE, 0),
            "band_reported": bands.get(PRIORITY_REPORTED, 0),
            "band_needs_human": bands.get(PRIORITY_NEEDS_HUMAN, 0),
        }
    except Exception:
        logger.exception("Could not read pending queue items.")
        return empty


SKILL_SEARCH_LIMIT = 10


def search_approved_skills(term: str) -> dict:
    """
    Finds approved, scored skills by name for the post-mortem edit form.

    Returns {"results": [...], "total": n, "truncated": bool}. The total is counted
    before the cap so the UI can say "10 of 34" rather than implying ten is all of it.

    The list key is "results" and NOT "items": Jinja resolves `search.items` to the
    dict's bound items() method rather than the key, which renders as a TypeError at
    best and a silently-always-truthy value in an `{% if %}` at worst.
    """
    empty = {"results": [], "total": 0, "truncated": False}
    cleaned = (term or "").strip().lower()
    if not cleaned:
        return empty

    try:
        master = load_master()
        timeseries = load_timeseries()
        newest = latest_snapshots(timeseries)

        matched = []
        for name, entry in master.items():
            if entry.get("status") != STATUS_APPROVED:
                continue
            if cleaned not in name.lower():
                continue
            snapshot = newest.get(name)
            if not snapshot:
                # Approved but never scored: there is no snapshot to correct, so the
                # edit form has nothing to act on.
                continue
            matched.append({
                "skill_id": name,          # the store is keyed by name; no integer ids
                "skill_name": name,
                "category": entry.get("category") or "",
                "summary_text": entry.get("wikipedia_summary") or "",
                "wiki_title": entry.get("resolved_title") or "",
                "snapshot_date": snapshot.get("snapshot_date"),
                "quarter": snapshot.get("quarter"),
                "ai_correlation_score": snapshot.get("ai_score"),
                "is_override": False,
                "open_reports": 1 if entry.get("report_note") else 0,
            })

        matched.sort(key=lambda row: row["skill_name"])
        return {
            "results": matched[:SKILL_SEARCH_LIMIT],
            "total": len(matched),
            "truncated": len(matched) > SKILL_SEARCH_LIMIT,
        }
    except Exception:
        logger.exception("Skill search failed for %r.", term)
        return empty


def _backlog_status() -> dict:
    """
    Summarizes queued ingestion work for the review page banner.

    Kept deliberately small: the page states that work is owed and why, and the CLI is
    where it gets drained. A "run it now" button would start a multi-hour job from a
    web request with nowhere to report progress.
    """
    try:
        data = backlog_store.load_backlog()
        targets = data.get("targets", [])
        return {
            "count": len(targets),
            "summary": backlog_store.describe() if targets else "",
            "quota_stopped": data.get("reason") == backlog_store.REASON_QUOTA,
            "targets": [entry["target"] for entry in targets[:12]],
        }
    except Exception:
        logger.exception("Could not read the ingestion backlog.")
        return {"count": 0, "summary": "", "quota_stopped": False, "targets": []}


def fetch_queue_stats() -> dict:
    """
    Counts per review status.

    `pending` counts only entries a human can act on, so the header tile agrees with
    the length of the queue below it. The raw pending count also includes the tools
    that were mapped but never audited -- hundreds of them -- and surfacing that
    number here would make the tile look like an enormous backlog that does not exist.
    Those are reported separately as `not_audited`.
    """
    zeroed = {
        "pending": 0, "approved": 0, "rejected": 0,
        "not_audited": 0, "superseded": 0, "other": 0, "total": 0,
    }
    try:
        master = load_master()
        stats = dict(zeroed)
        stats["total"] = len(master)

        for entry in master.values():
            status = entry.get("status")
            if status == STATUS_APPROVED:
                stats["approved"] += 1
            elif status == STATUS_REJECTED:
                stats["rejected"] += 1
            elif status == STATUS_PENDING:
                if (entry.get("wikipedia_summary") or "").strip():
                    stats["pending"] += 1
                else:
                    stats["not_audited"] += 1
            else:
                # An unrecognized status is a bug, not a category. Counted so the
                # tiles still sum to the total, and logged so it cannot hide.
                stats["other"] += 1

        if stats["other"]:
            logger.error("%d skills carry an unrecognized status.", stats["other"])
        return stats
    except Exception:
        logger.exception("Could not read queue statistics.")
        return zeroed


@app.get("/", response_class=HTMLResponse)
def render_review_queue(
    request: Request,
    page: str = None,
    err: str = None,
    item: str = None,
    skill_q: str = None,
    edited: str = None,
    edit_err: str = None,
    ingest_ok: str = None,
    ingest_err: str = None,
    second_pass_ok: str = None,
    sp_count: str = None,
    scrape_ok: str = None,
    scrape_err: str = None,
    rescore_ok: str = None,
    recheck_ok: str = None,
    recheck_err: str = None,
):
    """
    Renders one page of the pending review queue.

    `page` is accepted as a string and normalized, so a malformed or out-of-range value
    renders a sensible page instead of returning a 422 JSON body.

    `err` and `item` carry a remediation failure back from the POST redirect, since
    there is no session or flash-message machinery here. `err` is looked up in a fixed
    table and `item` is coerced to an int, so neither can put arbitrary text on the page.
    """
    error_message = None
    error_item = None
    if err:
        # COMMIT_ERRORS rather than REMEDIATION_ERRORS: it is a superset built from it,
        # and it is the only table carrying the merge codes, so the JavaScript-off
        # duplicate form reports a real sentence instead of the generic fallback.
        error_message = COMMIT_ERRORS.get(err, GENERIC_REMEDIATION_ERROR)
        try:
            error_item = int(str(item).strip())
        except (TypeError, ValueError):
            error_item = None

    # Post-mortem edit feedback. `edited` is a skill name echoed back purely so the
    # confirmation can name what was saved; Jinja autoescapes it on the way out.
    edit_message = None
    edit_error = None
    if edit_err:
        edit_error = EDIT_ERRORS.get(edit_err, EDIT_ERRORS["error"])
    elif edited:
        edit_message = EDIT_OK_MESSAGE

    # Second-pass confirmation, rebuilt here from two validated values rather than
    # reflected from the redirect, so the sentence on the page is always one of ours.
    second_pass_message = ""
    if second_pass_ok in SECOND_PASS_MODES:
        try:
            started_count = min(max(int(str(sp_count).strip()), 1), SECOND_PASS_MAX_LIMIT)
        except (TypeError, ValueError):
            started_count = SECOND_PASS_DEFAULT_LIMIT
        consequence = {
            "suggest": "No card will be changed; findings appear as proposals you accept.",
            # Kept for links and bookmarks predating the button's removal. It behaves as
            # suggest now, and must not claim otherwise.
            "apply": "No card will be changed; findings appear as proposals you accept.",
            "auto": "Only clean matches -- over the score threshold and past the audit -- "
                    "are applied and approved. Everything else is staged as a proposal.",
        }[second_pass_ok]
        second_pass_message = (
            f"Started in the background over up to {started_count} item(s). "
            f"{consequence} Reload in a few minutes to see the results."
        )

    paged = fetch_pending_page(_parse_page(page))
    # Current Starlette signature takes the request first; passing the name first is
    # interpreted as the request and fails inside template lookup.
    return templates.TemplateResponse(
        request,
        "review.html",
        {
            "stats": fetch_queue_stats(),
            "threshold": CROSS_ENCODER_THRESHOLD,
            # Every QUEUED name, not just this page's, because the duplicate of a card
            # is very often paged away from it -- "MicroSurveyCAD" and "MicroSurvey
            # Software MicroSurvey CAD" sort apart. Names only, so this stays small.
            "queued_names": queue_skill_names(),
            "error_message": error_message,
            "error_item": error_item,
            "skill_q": skill_q or "",
            "search": search_approved_skills(skill_q),
            "search_limit": SKILL_SEARCH_LIMIT,
            # Echoed back after an ingest submission. Both are re-parsed rather than
            # reflected raw, so only recognized targets can reach the page.
            "ingest_ok": ", ".join(parse_ingest_targets(ingest_ok)[0]) if ingest_ok else "",
            "ingest_err": ", ".join(parse_ingest_targets(ingest_err)[1]) if ingest_err else "",
            # Queued work from an interrupted or quota-stopped run.
            "backlog": _backlog_status(),
            # Both looked up in fixed tables, never reflected, so a hand-edited query
            # string cannot put chosen text on the page.
            "scrape_ok": SCRAPE_MESSAGES.get(scrape_ok) if scrape_ok else None,
            "scrape_err": SCRAPE_ERRORS.get(scrape_err) if scrape_err else None,
            "scrape": run_state.snapshot(),
            "second_pass_ok": second_pass_message,
            "second_pass_err": "",
            "rescore_ok": bool(rescore_ok),
            # The skill name IS reflected, unlike the messages above, because the whole
            # point is confirming which one was queued. Jinja autoescapes it, and the
            # worst a hand-edited query string achieves is showing the reader a name
            # nothing was done to.
            "recheck_ok": (recheck_ok or "").strip(),
            "recheck_err": bool(recheck_err),
            "edit_message": edit_message,
            "edit_error": edit_error,
            "edited_skill": edited or "",
            **paged,
        },
    )


def _embed_json(value) -> Markup:
    """
    Serializes a payload for embedding inside a <script type="application/json"> block.

    Jinja's autoescaping would turn & and quotes into HTML entities and break JSON.parse,
    so the value is marked safe -- which means the dangerous sequences have to be
    neutralized here instead. Escaping < and > stops a summary containing "</script>"
    from closing the block early, and escaping & stops entity interpretation. All three
    are valid JSON string escapes, so the parsed result is unchanged.
    """
    encoded = (
        json.dumps(value, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    return Markup(encoded)


@app.get("/dashboard", response_class=HTMLResponse)
def render_dashboard(request: Request):
    """
    Renders the analytics dashboard.

    Synchronous on purpose, like every other handler here: it does blocking file IO
    and as `async def` it would occupy the event loop and stall other requests.
    """
    data = load_dashboard()
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {"data": data, "payload": _embed_json(data)},
    )


@app.get("/login", response_class=HTMLResponse)
def render_login(request: Request, next: str = "/", err: str = None):
    """
    The password form.

    `next` is sanitized on the way in AND again on the way out of the POST, so a
    hand-edited link cannot turn this page into an open redirect.
    """
    return templates.TemplateResponse(
        request,
        "login.html",
        {
            "next": auth.safe_next(next),
            "error": LOGIN_ERRORS.get(err) if err else None,
            "configured": auth.configured(),
        },
        status_code=401 if err else 200,
    )


@app.post("/login")
def handle_login(password: str = Form(""), next: str = Form("/")):
    """Checks the password and sets the session cookie."""
    target = auth.safe_next(next)

    if not auth.check_password(password):
        # The reason is not distinguished for the user: "wrong password" and "no
        # password configured on the server" look identical from outside, and only one
        # of them is worth telling an anonymous caller about. The server log says which.
        logger.warning("Failed login attempt.")
        return RedirectResponse(
            url=f"/login?next={quote(target)}&err=bad_password", status_code=303
        )

    response = RedirectResponse(url=target, status_code=303)
    response.set_cookie(
        auth.COOKIE_NAME,
        auth.issue(),
        max_age=auth.MAX_AGE,
        httponly=True,       # not readable from JavaScript
        samesite="lax",      # still sent on the top-level navigations the forms use
        path="/",
    )
    logger.info("Login succeeded.")
    return response


@app.post("/logout")
def handle_logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(auth.COOKIE_NAME, path="/")
    return response


# Skill names arrive as path segments. They can contain slashes ("SAS/CONNECT"), plus
# signs ("C++") and spaces, so every route below takes the `:path` converter and the
# templates URL-encode the name. Without :path, "SAS/CONNECT" would 404 as an unknown
# two-segment route.


@app.post("/approve/{skill_name:path}")
def handle_approve_form(
    skill_name: str,
    override_summary: str = Form(...),
    page: str = Form("1"),
):
    """
    Scores the (possibly edited) summary, approves the skill, and returns to the page.

    The reviewer keeps their place instead of being thrown back to page 1. Note that
    offset pagination shifts rows backward as items leave the queue, so the first row of
    the next page moves onto this one -- expected, though it looks like a bug if you are
    not expecting it. If this was the last item on the final page, the redirect targets a
    page that no longer exists and the GET clamps down to the new last page.
    """
    parsed_page = _parse_page(page)
    ok, code = approve_skill(skill_name, override_summary)
    if ok:
        return RedirectResponse(url=f"/?page={parsed_page}", status_code=303)
    return RedirectResponse(
        url=f"/?page={parsed_page}&err={code}&item={quote(skill_name)}", status_code=303
    )


@app.post("/remediate/{skill_name:path}")
def handle_remediate_form(
    skill_name: str,
    reference: str = Form(""),
    page: str = Form("1"),
):
    """
    Repoints a pending item at a reviewer-supplied Wikipedia page and re-renders it.

    Synchronous and NOT a BackgroundTask, unlike ingestion: the whole point is that the
    reviewer sees the refetched text and its new score before deciding, so the work has
    to finish before the redirect. It is two Wikipedia calls and one cross-encoder pass,
    on the order of a couple of seconds.

    On failure the code and the skill name go back as query parameters. The code comes
    from a fixed table and the name is quoted, so neither can inject. The submitted URL
    is deliberately NOT echoed anywhere in the response.
    """
    parsed_page = _parse_page(page)
    ok, code = remediate_skill(skill_name, reference)
    if ok:
        return RedirectResponse(url=f"/?page={parsed_page}", status_code=303)
    return RedirectResponse(
        url=f"/?page={parsed_page}&err={code}&item={quote(skill_name)}", status_code=303
    )


@app.post("/reject/{skill_name:path}")
def handle_reject_form(skill_name: str, page: str = Form("1")):
    ok, code = reject_skill(skill_name)
    if not ok:
        logger.error("Rejection failed for %r (%s).", skill_name, code)
    return RedirectResponse(url=f"/?page={_parse_page(page)}", status_code=303)


@app.post("/reject-draft/{skill_name:path}")
def handle_reject_draft_form(skill_name: str, page: str = Form("1")):
    """
    The JavaScript-off path for sending a model-written definition back to be redrafted.

    Leaves the skill pending and eligible, so the next second pass rewrites it under the
    current DEFINITION_SPEC rather than the rules it was first written under.
    """
    parsed_page = _parse_page(page)
    ok, code = reject_draft(skill_name)
    if ok:
        return RedirectResponse(url=f"/?page={parsed_page}", status_code=303)
    logger.error("Rejecting the draft for %r failed (%s).", skill_name, code)
    return RedirectResponse(
        url=f"/?page={parsed_page}&err={code}&item={quote(skill_name)}", status_code=303
    )


@app.post("/mark-duplicate/{skill_name:path}")
def handle_mark_duplicate_form(
    skill_name: str, target: str = Form(...), page: str = Form("1")
):
    """
    The JavaScript-off path for merging one queued skill into another.

    Errors come back through the same ?err= channel the remediation form uses, so a bad
    target name is reported on the page rather than swallowed. The target is echoed only
    through `item`, which is the skill name Jinja escapes; the failure sentence itself
    comes from the fixed COMMIT_ERRORS table.
    """
    parsed_page = _parse_page(page)
    ok, code = mark_duplicate(skill_name, target)
    if ok:
        return RedirectResponse(url=f"/?page={parsed_page}", status_code=303)
    logger.error("Marking %r a duplicate of %r failed (%s).", skill_name, target, code)
    return RedirectResponse(
        url=f"/?page={parsed_page}&err={code}&item={quote(skill_name)}", status_code=303
    )


class StagedDecision(BaseModel):
    """
    One decision a reviewer queued on a card without committing it.

    `summary` carries the textarea contents for an approve, so the text that gets scored
    is the text the reviewer was looking at when they clicked. `reference` carries the
    URL for a remediate. `target` carries the surviving skill name for a mark-duplicate.
    Each is ignored by the actions that do not use it.
    """

    skill: str = Field(min_length=1, max_length=300)
    action: str = Field(min_length=1, max_length=32)
    summary: Optional[str] = Field(default=None, max_length=20000)
    # Bounded the same as `skill`, because it is one: the name of the record being merged
    # into. Whether it EXISTS is settled in review_actions against the loaded store, which
    # is the only place that can answer it without a second read.
    target: Optional[str] = Field(default=None, max_length=300)
    # Deliberately LOOSER than scraping's MAX_REFERENCE_URL_CHARS. The real limit is
    # enforced there, where it can answer with a sentence telling the reviewer to
    # highlight less; rejecting it here would produce a bare 422 for what is an ordinary
    # mistake. This bound exists only so a hostile payload cannot be unbounded.
    reference: Optional[str] = Field(default=None, max_length=MAX_REFERENCE_URL_CHARS * 8)


class CommitPayload(BaseModel):
    decisions: List[StagedDecision] = Field(default_factory=list, max_length=MAX_BATCH_DECISIONS)


# Sentences for a staged decision that did not land. Built on the remediation table,
# which already covers every code the underlying appliers return, plus the four this
# route can produce on its own. Same fixed-table rule as everywhere else: nothing from
# the request reaches the page except the skill name, which Jinja and JSON both escape.
COMMIT_ERRORS = {
    **REMEDIATION_ERRORS,
    "no_summary": EDIT_ERRORS["no_summary"],
    "bad_decision": "That decision was malformed and was not applied.",
    "duplicate": "Two decisions of the same kind were staged for that skill, so only the "
                 "first was applied.",
    "source_failed": "The page or definition this was waiting on did not land, so this "
                     "was not applied. The reason is listed above it.",
    "too_many": f"Too many decisions in one commit; the limit is {MAX_BATCH_DECISIONS}.",
    "unknown_target": "The skill this was to be merged into is not in the store, so "
                      "nothing was changed. Check the name against the queue.",
    "duplicate_self": "A skill cannot be marked as a duplicate of itself.",
    "no_draft": "There is no model-written definition on that card to send back.",
    "not_pending": "That skill has already been decided, so its draft cannot be sent back.",
    "target_rejected": "The skill this was to be merged into has already been rejected. "
                       "Merging into it would put the occupation mappings somewhere "
                       "nothing reads them, so nothing was changed.",
}


@app.post("/commit-review")
def handle_commit_review(payload: CommitPayload):
    """
    Applies a page of staged reviewer decisions in one write.

    The review page stages every click in the browser and posts them here on Commit, so
    the store is rewritten once per page of review rather than once per click. Answers
    JSON rather than redirecting, for the same reason /report does: the caller is
    fetch() and it needs the per-item outcome, not a new page.

    This is the ONE reviewer-facing write that takes the store-write lock. It WAITS for
    it rather than refusing on the spot: a person is holding the page open expecting
    their decisions to land, and the longest thing they can now be waiting on is one
    occupation of a scrape. It used to be the whole scrape, which is why a commit made
    during a run appeared to do nothing.

    The response reports the status of every skill READ BACK from the saved store, so
    "committed" is an observation rather than a claim.

    Always HTTP 200 with an `ok` flag, again matching /report: the status describes the
    request being understood, and a partial commit is a normal, reportable outcome
    rather than a transport failure.
    """
    decisions = [d.model_dump() for d in payload.decisions]
    if not decisions:
        return JSONResponse({
            "ok": False, "applied": 0, "applied_items": [], "failures": [],
            "message": "Nothing was staged, so nothing was committed.",
        })

    try:
        with run_state.store_writer(
            "a reviewer commit", timeout=run_state.STORE_WAIT_SECONDS
        ):
            applied, applied_items, failures = apply_review_batch(decisions)
    except run_state.StoreBusy as busy:
        logger.warning(
            "%s A commit of %d decision(s) was refused; nothing was changed.",
            busy, len(decisions),
        )
        return JSONResponse({
            "ok": False, "applied": 0, "applied_items": [], "failures": [],
            "message": (
                f"Waited {run_state.STORE_WAIT_SECONDS:.0f}s and "
                f"{busy.holder or 'a background job'} still holds the store. Nothing was "
                f"changed and your decisions are still staged, so press Commit again in "
                f"a moment."
            ),
        })
    except Exception:
        logger.exception("Commit of %d staged decision(s) failed.", len(decisions))
        return JSONResponse({
            "ok": False, "applied": 0, "applied_items": [], "failures": [],
            "message": "The commit failed and nothing was saved. See the server log.",
        })

    detailed = [
        {**failure, "message": COMMIT_ERRORS.get(failure["code"], GENERIC_REMEDIATION_ERROR)}
        for failure in failures
    ]

    # Counted from what the store says now, not from what was asked for. The three
    # refetching actions leave an item PENDING by design, and a reviewer who staged
    # those and watched the queue not shrink deserves to be told that in as many words.
    #
    # Counted per SKILL, not per decision: a chained "use this definition, then approve
    # it" applies two decisions to one skill, and reporting "2 approved" for one card
    # would overstate what happened.
    status_by_skill = {item["skill"]: item["status"] for item in applied_items}
    by_status = Counter(status_by_skill.values())
    parts = [f"{count} {status}" for status, count in sorted(by_status.items())]
    still_pending = by_status.get(STATUS_PENDING, 0)

    if applied:
        skills = len(status_by_skill)
        scope = f"{applied} decision" + ("" if applied == 1 else "s")
        if skills != applied:
            scope += f" on {skills} skill" + ("" if skills == 1 else "s")
        message = f"Committed {scope}: {', '.join(parts)}."
        if still_pending:
            message += (
                f" The {still_pending} still pending {'is' if still_pending == 1 else 'are'} "
                f"refetched pages, which stay in the queue by design so you can read the "
                f"new text before deciding."
            )
    else:
        message = "Nothing was committed."

    if detailed:
        message += (
            f" {len(detailed)} could not be applied, listed below; "
            f"{'it is' if len(detailed) == 1 else 'they are'} still staged."
        )

    logger.info(
        "Commit applied %d decision(s) (%s), %d failed. Skills: %s",
        applied, ", ".join(parts) or "none", len(detailed),
        ", ".join(item["skill"] for item in applied_items) or "none",
    )

    return JSONResponse({
        "ok": applied > 0,
        "applied": applied,
        "applied_items": applied_items,
        "failures": detailed,
        "message": message,
    })


@app.post("/report")
def handle_report(
    skill_name: str = Form(...),
    reason: str = Form(...),
    note: str = Form(""),
):
    """
    Files a viewer's report that a skill's page or definition is wrong.

    Answers JSON rather than redirecting, because this is posted by fetch() from the
    dashboard drawer and the drawer should stay open to show the outcome. Every other form
    in the app is a plain POST-redirect-GET; this one is the exception because there is no
    page to return to that would not throw away the reader's scroll position and filters.

    Always HTTP 200 with an `ok` flag. The status code describes the report being received
    and understood, not whether the report was accepted, and a 4xx here would make the
    drawer show a browser-level failure for what is really a validated form response.
    """
    ok, code = report_skill(skill_name, reason, note)
    messages = {
        "ok": "Thanks. This skill has been queued for review; its score is unchanged for now.",
        "already_pending": "Thanks. This skill was already awaiting review, so your note was "
                           "added to the open item.",
        "bad_reason": "Pick one of the listed reasons.",
        "not_found": "That skill no longer exists.",
        "no_evidence": "This skill has no stored definition to review, so it cannot be "
                       "reported from here.",
        "error": "The report could not be filed. See the server log.",
    }
    return JSONResponse({
        "ok": ok,
        "code": code,
        "message": messages.get(code, messages["error"]),
    })


@app.post("/edit-skill/{skill_name:path}")
def handle_edit_skill(
    skill_name: str,
    override_summary: str = Form(...),
    reference: str = Form(""),
    skill_q: str = Form(""),
):
    """
    Applies a post-mortem correction to an approved skill and returns to the search.

    `skill_q` is carried through so the editor lands back on the same result list rather
    than an empty search box. It is passed through quote() on the way into the redirect;
    without that, a search term containing & or # would truncate the Location header.
    """
    ok, code = edit_approved_skill(skill_name, override_summary, reference)
    query = f"skill_q={quote(skill_q.strip())}" if skill_q.strip() else ""
    tail = f"edited={quote(skill_name)}" if ok else f"edit_err={code}"
    joined = "&".join(part for part in (query, tail) if part)
    return RedirectResponse(url=f"/?{joined}", status_code=303)


def _run_ingestion(targets: list) -> None:
    """
    Runs ingestion in the background, backlog first.

    Submitted targets are queued on the backlog BEFORE anything runs, then the whole
    backlog is drained. Two reasons: a submission made on a day when the quota is
    already spent is preserved rather than silently dropped, and work owed from a
    previous run is not jumped by a new request.

    run_backlog removes each target only once it completes, so a quota wall part-way
    through leaves the rest queued for tomorrow.
    """
    from definitions_algorithm import run_backlog

    backlog_store.add_targets(targets, reason=backlog_store.REASON_REQUESTED)

    # No lock here any more. Serialization happens per occupation, inside
    # definitions_algorithm, around each load_master -> mutate -> save_master window.
    # Taking it for the whole run is what stopped a reviewer committing anything while
    # an ingestion was going, and an ingestion can run for a very long time.
    try:
        result = run_backlog()
    except Exception:
        logger.exception("Background ingestion failed for %s.", ", ".join(targets))
        return

    if result.get("quota_exhausted"):
        logger.error(
            "ALL GEMINI TOKENS EXHAUSTED FOR TODAY. %d target(s) remain queued and "
            "will resume on the next run. Nothing was written unaudited.",
            result.get("remaining", 0),
        )


SECOND_PASS_MODES = ("suggest", "apply", "auto")
SECOND_PASS_DEFAULT_LIMIT = 25
SECOND_PASS_MAX_LIMIT = 200


def _run_second_pass(limit: int, apply_mode: str) -> None:
    """
    Runs the second pass in the background.

    No backlog bookkeeping, unlike ingestion. Second-pass work is re-derived for free by
    rescanning the pending queue, so an interrupted run needs nothing recorded to resume:
    the entries it never reached simply still have no marker. Queueing them on the
    ingestion backlog would make the next ingest re-scrape whole occupations that need
    nothing.
    """
    from second_pass import run_second_pass

    # Held for the whole pass, unlike ingestion. This one writes per item through
    # review_actions rather than in one window at the end, and it is a job of minutes,
    # so holding it is both necessary and cheap.
    #
    # Refused rather than queued: a job that "waits" is a job nobody knows is waiting.
    try:
        with run_state.store_writer("a second pass over the queue"):
            result = run_second_pass(limit=limit, apply_mode=apply_mode)
    except run_state.StoreBusy as busy:
        logger.warning(
            "%s The second pass did not start and nothing was changed. Try again in a "
            "moment.", busy,
        )
        return
    except Exception:
        logger.exception("Background second pass failed.")
        return

    if result.get("quota_exhausted"):
        logger.error(
            "ALL GEMINI TOKENS EXHAUSTED FOR TODAY. The second pass stopped after %d "
            "item(s); %d were left untouched and will be picked up on the next run.",
            result.get("processed", 0), result.get("remaining", 0),
        )


# Every column the export carries, in a fixed order. Explicit rather than derived from
# whatever keys a row happens to have: a spreadsheet whose columns move between exports
# cannot be diffed against last quarter's, which is most of the point of having one.
#
# The score is split into its parts -- the measurement, then each addition separately --
# because "why is this 0.33" is the question a reader opens the file to answer, and a
# single total cannot answer it.
CSV_COLUMNS = (
    ("skill_name", "Skill"),
    ("category", "O*NET category"),
    ("category_bucket", "AI class"),
    ("sub_category", "Sub-category"),
    # The banded, reader-facing score and the raw cosine behind it. BOTH, deliberately:
    # this export is for analysis, and every threshold in the engine is on the raw scale,
    # so dropping the raw value would make it impossible to check a decision against the
    # bar it was decided by. See compute_display_ai_score in dashboardtables.py.
    ("display_ai_score", "AI Score"),
    ("ai_score", "AI score (raw cosine)"),
    ("ai_score_base", "AI score measured (raw cosine)"),
    ("ai_engineering_sim", "AI engineering pole"),
    ("ai_generative_sim", "AI generative pole"),
    ("embedded_ai_boost", "Boost: embedded AI"),
    ("lexical_ai_boost", "Boost: AI phrase"),
    ("lexical_ai_terms", "AI phrases found"),
    ("tech_base_sim", "Language and tooling"),
    ("ml_pipeline_sim", "ML infrastructure"),
    ("embedded_ai_sim", "Embedded AI similarity"),
    ("contrast_sim", "Document software similarity"),
    ("legacy_sim", "Legacy software similarity"),
    ("embeds_ai", "AI embedded"),
    ("embeds_ai_evidence", "Embedding evidence"),
    ("embeds_ai_evidence_url", "Embedding evidence URL"),
    ("embeds_ai_checked_at", "Embedding checked"),
    ("decision_metric", "Decided on"),
    ("decision_threshold", "Decision threshold"),
    ("decision_margin", "Decision margin"),
    ("in_semantic_variance_band", "Marginal"),
    ("classification_confidence", "Confidence"),
    ("is_generic_category", "Is a category"),
    ("flagship_version", "Measured as"),
    ("flagship_source", "Flagship source"),
    ("best_source_name", "Source"),
    ("resolved_title", "Source title"),
    ("reference_url", "Source URL"),
    ("occupation_count", "Occupations"),
    ("is_hot_tech_anywhere", "Hot tech"),
    ("snapshot_date", "Snapshot"),
    ("wikipedia_summary", "Definition"),
)


def _csv_cell(value) -> str:
    """
    One value, rendered the way a spreadsheet should read it.

    A list becomes a semicolon-joined string rather than Python's "['a', 'b']", and
    None becomes empty rather than the word "None" -- which Excel would happily sort
    alphabetically among real values. True/False are written as words on purpose: the
    embedded-AI column is three-state, and blank has to mean "never established"
    distinctly from "no".
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        return "; ".join(str(item) for item in value)
    return str(value)


def build_scores_csv(rows) -> str:
    """Renders the dashboard rows as CSV text, header included."""
    import csv
    import io

    buffer = io.StringIO()
    # QUOTE_MINIMAL with the default dialect: definitions contain commas and quotes, and
    # csv handles the escaping correctly. Writing this by hand with join() is how a
    # definition containing a comma silently shifts every column after it.
    writer = csv.writer(buffer)
    writer.writerow([label for _, label in CSV_COLUMNS])
    for row in rows:
        writer.writerow([_csv_cell(row.get(key)) for key, _ in CSV_COLUMNS])
    return buffer.getvalue()


@app.get("/admin/scores.csv")
def handle_scores_csv():
    """
    Every scored skill and every part of its score, as a spreadsheet.

    Behind the admin password like the rest of `/admin`. The same data is already public
    through the dashboard, so this is not a disclosure boundary -- it is here because it
    sits with the other operator tools and because a file that downloads on click is an
    odd thing to hang off a public page.
    """
    rows = build_dashboard_rows(load_master(), load_timeseries())
    rows.sort(key=lambda row: (row.get("ai_score") is None, -(row.get("ai_score") or 0.0)))

    body = build_scores_csv(rows)
    stamp = datetime.date.today().isoformat()
    logger.info("Exported %d scored skill(s) as CSV.", len(rows))
    return PlainTextResponse(
        body,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="ai_skill_scores_{stamp}.csv"',
        },
    )


def _run_rescore() -> None:
    """
    Re-measures every approved, scored skill from text already on disk.

    NO NETWORK OF ANY KIND. No Wikipedia, no DuckDuckGo, no Gemini, no quota. It
    re-embeds each stored definition against the current anchors and rewrites the
    current quarter's snapshot. That is what makes a scoring change deployable at all:
    changing an anchor invalidates every stored score, and without this the only way to
    refresh them would be a full re-scrape of pages that have not changed.

    Distinct from reclassify_snapshots.py, which re-DECIDES stored numbers without
    re-measuring them. That is the right tool when only a threshold moved; this is the
    one for a changed anchor or a changed scoring surface.

    Goes through definitions_algorithm.record_snapshot, the same writer a human approval
    uses, so an automatic rescore and a manual approval produce structurally identical
    records.
    """
    from definitions_algorithm import record_snapshot

    try:
        with run_state.store_writer("a full rescore"):
            master = load_master()
            timeseries = load_timeseries()

            rescored = 0
            changed = []
            for skill_name, entry in master.items():
                if entry.get("status") != STATUS_APPROVED:
                    continue
                before = None
                for record in timeseries:
                    if record.get("skill_name") == skill_name:
                        before = record.get("category_bucket")
                metrics = record_snapshot(entry, timeseries)
                rescored += 1
                if before is not None and metrics["category_bucket"] != before:
                    changed.append((skill_name, before, metrics["category_bucket"]))

            save_master(master)
            save_timeseries(timeseries)
    except run_state.StoreBusy as busy:
        logger.warning("%s The rescore did not start and nothing was changed.", busy)
        return
    except Exception:
        logger.exception("Background rescore failed.")
        return

    logger.warning(
        "Rescored %d approved skill(s) from stored text. %d changed class.",
        rescored, len(changed),
    )
    for skill_name, before, after in changed:
        logger.warning("  %s: %s -> %s", skill_name, before, after)


@app.post("/admin/rescore")
def handle_rescore(background_tasks: BackgroundTasks):
    """
    Redoes the maths on every approved skill. Spends nothing.

    Admin-only by virtue of living off `/`, which auth.py protects. The public dashboard
    deliberately has no such control: it would let anyone rewrite every score on the site.
    """
    logger.warning("Scheduling a full rescore from stored text. No network calls.")
    background_tasks.add_task(_run_rescore)
    return RedirectResponse(url="/?rescore_ok=1", status_code=303)


def _run_recheck_embedding(skill_name: str) -> None:
    """Re-probes one skill's embedded-AI verdict, ignoring the never-re-ask rule."""
    import storage
    from embedding_pass import apply_to_entry, plan_run, record_verdict, score_entry
    from embedding_probe import search_embedding_evidence
    from agentic_source_check import AuditUnavailable, get_source_checker
    from gemini_keys import ROLE_EMBEDDING, DailyQuotaExhausted

    try:
        with run_state.store_writer(f"an embedding re-check for {skill_name!r}"):
            master = load_master()
            timeseries = load_timeseries()
            cache = storage.load_embedding_cache()

            entry = master.get(skill_name)
            if entry is None:
                logger.warning("Re-check asked for %r, which is not in the store.", skill_name)
                return

            plan = plan_run(master, timeseries, cache, forced={skill_name})
            if not any(name == skill_name for name, _ in plan["due"]):
                logger.warning(
                    "%r is not in scope for an embedding check: it must be approved and "
                    "scored. Nothing was searched.", skill_name,
                )
                return

            found = search_embedding_evidence(skill_name)
            if found["error"]:
                # Unknown, never False. A blocked search is a gap, not a finding.
                record_verdict(cache, skill_name, None, error=found["error"])
                storage.save_embedding_cache(cache)
                logger.warning(
                    "Search for %r failed (%s). It stays unestablished and can be "
                    "re-checked again.", skill_name, found["error"],
                )
                return

            try:
                answers = get_source_checker(ROLE_EMBEDDING).grade_embedding_batch([{
                    "item_id": skill_name,
                    "skill_name": skill_name,
                    "definition": entry.get("wikipedia_summary") or "",
                    "results": found["results"],
                }])
            except (AuditUnavailable, DailyQuotaExhausted) as err:
                record_verdict(cache, skill_name, None, error="grader_unavailable")
                storage.save_embedding_cache(cache)
                logger.warning("Grader unavailable for %r: %s", skill_name, err)
                return

            verdict = answers.get(skill_name)
            record = record_verdict(cache, skill_name, verdict,
                                    error=None if verdict else "missing_from_batch")
            storage.save_embedding_cache(cache)

            if verdict is None:
                logger.warning("No verdict came back for %r; it stays unestablished.", skill_name)
                return

            previous = entry.get("embeds_ai")
            apply_to_entry(entry, record)
            metrics = score_entry(skill_name, entry)
            upsert_snapshot(timeseries, skill_name, metrics,
                            entry.get("onet_codes", []), entry.get("onet_titles", []))
            save_master(master)
            save_timeseries(timeseries)

            logger.warning(
                "Re-checked %r: embeds_ai %s -> %s (%s). %s",
                skill_name, previous, record["embeds_ai"],
                metrics["category_bucket"], record["evidence"][:120],
            )
    except run_state.StoreBusy as busy:
        logger.warning("%s The re-check did not start.", busy)
    except Exception:
        logger.exception("Background embedding re-check failed.")


@app.post("/admin/recheck-embedding")
def handle_recheck_embedding(background_tasks: BackgroundTasks, skill_name: str = Form(...)):
    """
    Re-probes one skill's embedded-AI verdict, overriding the never-re-ask rule.

    The only way past "a confirmed finding is never re-asked", and it takes an explicit
    name rather than being a blanket flag, so it cannot accidentally re-run the whole
    store. Costs one search and one Gemini call.
    """
    name = (skill_name or "").strip()
    if not name:
        return RedirectResponse(url="/?recheck_err=empty", status_code=303)

    logger.warning("Scheduling an embedding re-check for %r.", name)
    background_tasks.add_task(_run_recheck_embedding, name)
    return RedirectResponse(url=f"/?recheck_ok={quote(name)}", status_code=303)


@app.post("/second-pass")
def handle_second_pass_form(
    background_tasks: BackgroundTasks,
    limit: str = Form(str(SECOND_PASS_DEFAULT_LIMIT)),
    apply_mode: str = Form("suggest"),
):
    """
    Queues a second pass over the pending queue.

    apply_mode is validated against a fixed tuple rather than passed through, and both
    the default and the fallback are the mode that changes nothing. The modes differ in
    how much a machine may do unattended, so an unrecognized value has to land on the
    least of them rather than on whatever the form happened to send.

    "apply" is still accepted but no longer offered by the form, and second_pass.decide
    now treats it as suggest: a resolution that misses the high-confidence bar is staged
    for a click instead of being written over the card.
    """
    mode = apply_mode if apply_mode in SECOND_PASS_MODES else "suggest"

    try:
        count = int(str(limit).strip())
    except (TypeError, ValueError):
        count = SECOND_PASS_DEFAULT_LIMIT
    count = min(max(count, 1), SECOND_PASS_MAX_LIMIT)

    if mode == "auto":
        # Greppable afterwards, and deliberately noisier than the other two: this is the
        # only web action that can approve and score a skill with no human involved.
        logger.warning(
            "Scheduling a second pass over %d item(s) WITH AUTO-APPROVAL enabled.", count
        )
    elif mode == "apply":
        logger.info(
            "Scheduling a second pass over %d item(s) in the retired 'apply' mode, which "
            "now behaves as suggest. No card will be changed.", count
        )
    else:
        logger.info(
            "Scheduling a proposal-only second pass over %d item(s). No card will be "
            "changed.", count
        )

    background_tasks.add_task(_run_second_pass, count, mode)

    # Only the mode and the count go back, and both are re-validated on the way in. The
    # confirmation sentence itself is built from a fixed table on the GET side, so a
    # hand-edited query string cannot put chosen text on the page.
    return RedirectResponse(
        url=f"/?second_pass_ok={mode}&sp_count={count}", status_code=303
    )


@app.post("/apply-draft/{skill_name:path}")
def handle_apply_draft(skill_name: str, page: str = Form("1")):
    """
    Accepts a stored model-written definition on one item.

    The sibling of /apply-suggestion for products with no article to point at. No fetch
    and no Gemini call: the sentence was written and paid for during the pass, and the
    reviewer is choosing to put it on the card.
    """
    parsed_page = _parse_page(page)
    ok, code = apply_machine_draft(skill_name)
    if ok:
        return RedirectResponse(url=f"/?page={parsed_page}", status_code=303)
    return RedirectResponse(
        url=f"/?page={parsed_page}&err={code}&item={quote(skill_name)}", status_code=303
    )


@app.post("/apply-suggestion/{skill_name:path}")
def handle_apply_suggestion(skill_name: str, page: str = Form("1")):
    """
    Accepts a stored second-pass suggestion on one item.

    Synchronous for the same reason as remediation: the reviewer must see the refetched
    text and its score before deciding, so the fetch has to finish before the redirect.
    """
    parsed_page = _parse_page(page)
    ok, code = apply_machine_suggestion(skill_name)
    if ok:
        return RedirectResponse(url=f"/?page={parsed_page}", status_code=303)
    return RedirectResponse(
        url=f"/?page={parsed_page}&err={code}&item={quote(skill_name)}", status_code=303
    )


TARGET_PATTERN = re.compile(r"^\d{2}-(\d{4}\.\d{2})?$")


def parse_ingest_targets(raw: str) -> tuple:
    """
    Splits a comma-separated field into O*NET codes and SOC prefixes.

    One field accepts both, distinguished by shape: a full code carries a dot
    ("15-2051.00"), a family prefix does not ("15-"). A bare "15" is normalized to
    "15-" because the trailing hyphen is easy to forget.

    Returns (targets, rejected). Anything unrecognized is returned rather than dropped,
    so a typo is reported instead of silently ingesting only the parts that parsed.
    """
    targets, rejected = [], []
    for part in (raw or "").replace(" ", "").split(","):
        if not part:
            continue
        candidate = part if ("." in part or part.endswith("-")) else f"{part}-"
        if TARGET_PATTERN.match(candidate):
            if candidate not in targets:
                targets.append(candidate)
        else:
            rejected.append(part)
    return targets, rejected


@app.post("/ingest")
def handle_ingest_form(background_tasks: BackgroundTasks, onet_code: str = Form(...)):
    """
    Queues an ingestion run in the background.

    A single occupation is dozens of rate-limited network calls and a whole SOC family
    is dozens of occupations, so the response returns immediately and the queue
    populates as work completes.
    """
    targets, rejected = parse_ingest_targets(onet_code)

    if rejected:
        logger.error("Ignoring unrecognized ingest targets: %s.", ", ".join(rejected))
        return RedirectResponse(
            url=f"/?ingest_err={quote(', '.join(rejected))}", status_code=303
        )

    if targets:
        families = [t for t in targets if "." not in t]
        if families:
            # Worth a warning line: one prefix is dozens of occupations and hundreds of
            # skills, and the run will take a long time and spend real Gemini quota.
            logger.warning(
                "Scheduling FAMILY ingestion for %s. This covers every occupation in "
                "those groups and will run for a while.", ", ".join(families),
            )
        logger.info("Scheduling background ingestion for %s.", ", ".join(targets))
        background_tasks.add_task(_run_ingestion, targets)
        return RedirectResponse(
            url=f"/?ingest_ok={quote(', '.join(targets))}", status_code=303
        )

    return RedirectResponse(url="/", status_code=303)


# ---------------------------------------------------------------------------
# The scrape panel
# ---------------------------------------------------------------------------

SCRAPE_ERRORS = {
    "already_running": "A scrape is already running. Pause it before starting another.",
    "busy": "Another background job is writing the store right now. Try again shortly.",
    "nothing_queued": "There is nothing queued to resume. Start a scrape instead.",
    "not_running": "Nothing is running, so there was nothing to pause.",
    "confirm": "Type DISCARD in the box to confirm. Nothing was changed.",
}

SCRAPE_MESSAGES = {
    "started": "Scrape started. Planning takes about thirty seconds before the bar moves.",
    "resumed": "Resumed. The bar picks up where it left off.",
    "pausing": "Pausing. The occupation in flight will finish first, which takes a minute or so.",
    "discarded": "Queue discarded. Nothing already ingested was touched.",
}


def _scrape_worker(scope: str, force: bool) -> None:
    """
    Runs a full scrape on its own thread until it finishes, stalls, or is paused.

    NOT a BackgroundTask. Those run on the same capped anyio threadpool that serves
    requests, and a job measured in weeks would hold one of those slots for the whole
    time. A dedicated thread costs one thread and takes nothing from request handling.

    Planning happens HERE rather than in the POST handler because it is 23 network
    calls and about thirty seconds; a request that did it inline would look hung. The
    panel shows a "planning" state for that window.
    """
    from definitions_algorithm import plan_full_scrape, run_backlog

    final_state = run_state.FINISHED
    error = None

    # No store lock around the run. It used to be held from here to the end, which for a
    # full scrape is weeks -- and every reviewer commit made in that window was refused.
    # definitions_algorithm takes it per occupation instead, so a reviewer waits for one
    # occupation and the two can no longer overwrite each other. The run slot claimed
    # before this thread started is still what stops two scrapes running at once.
    try:
        if scope == "full":
            plan = plan_full_scrape()
            run_state.set_total(plan["total"], baseline=plan["baseline"])
            logger.warning(
                "Full scrape planned: %d occupations across %d SOC groups. This "
                "runs for weeks of daily quota, not hours.",
                plan["total"], len(plan["prefixes"]),
            )
        else:
            # Resume: whatever is already queued is the plan. The total is what is
            # owed right now, so the bar starts at zero and fills across this run.
            queued = len(backlog_store.target_list())
            run_state.set_total(queued, baseline=0)
            logger.info("Resuming a scrape over %d queued occupation(s).", queued)

        result = run_backlog(
            force=force,
            progress=run_state.on_event,
            should_continue=run_state.should_continue,
        )

        if result.get("quota_exhausted"):
            final_state = run_state.QUOTA_STOPPED
            logger.error(
                "ALL GEMINI TOKENS EXHAUSTED FOR TODAY. %d occupation(s) remain "
                "queued and resume tomorrow. Nothing was written unaudited.",
                result.get("remaining", 0),
            )
        elif result.get("stopped"):
            final_state = run_state.PAUSED
        else:
            final_state = run_state.FINISHED

    except Exception as err:
        # Any escape leaves the run slot claimed forever and wedges the button, so the
        # release below is unconditional and this only records what happened.
        final_state = run_state.INTERRUPTED
        error = str(err)
        logger.exception("The scrape thread failed.")
    finally:
        run_state.release(final_state, error=error)


def _spawn_scrape(scope: str, force: bool = False) -> None:
    """
    Starts the worker on a daemon thread.

    daemon=True so Ctrl-C still exits the server. The cost is that the occupation in
    flight is abandoned without its finally block -- at most one occupation's work,
    which is still queued on the backlog because remove_target only fires after a
    successful save.
    """
    threading.Thread(
        target=_scrape_worker, args=(scope, force), name="scrape-worker", daemon=True
    ).start()


@app.post("/admin/scrape/start")
def handle_scrape_start(scope: str = Form("full")):
    """
    Claims the run slot and starts a full scrape.

    The claim is a compare-and-set, so two people clicking at the same moment produce
    one run and one clear refusal rather than two runs racing over the store.
    """
    if scope != "full":
        scope = "full"

    run_id = run_state.try_claim("All 23 SOC major groups")
    if run_id is None:
        return RedirectResponse(url="/?scrape_err=already_running", status_code=303)

    logger.warning("Full scrape requested from the web UI (run %s).", run_id)
    _spawn_scrape("full")
    return RedirectResponse(url="/?scrape_ok=started", status_code=303)


@app.post("/admin/scrape/resume")
def handle_scrape_resume():
    """Drains whatever is still queued, without re-planning."""
    if backlog_store.is_empty():
        return RedirectResponse(url="/?scrape_err=nothing_queued", status_code=303)

    run_id = run_state.try_claim("Resuming queued occupations")
    if run_id is None:
        return RedirectResponse(url="/?scrape_err=already_running", status_code=303)

    logger.info("Scrape resumed from the web UI (run %s).", run_id)
    _spawn_scrape("resume")
    return RedirectResponse(url="/?scrape_ok=resumed", status_code=303)


@app.post("/admin/scrape/pause")
def handle_scrape_pause():
    """
    Asks the worker to stop after the occupation it is in.

    Returns immediately; the worker may take a minute to reach the boundary. Everything
    unfinished is already on the backlog, so nothing is lost in the meantime.
    """
    if not run_state.request_stop():
        return RedirectResponse(url="/?scrape_err=not_running", status_code=303)
    return RedirectResponse(url="/?scrape_ok=pausing", status_code=303)


@app.post("/admin/scrape/discard")
def handle_scrape_discard(confirm: str = Form("")):
    """
    Empties the queue of planned occupations.

    Behind a typed confirmation because it throws away a plan that cost 23 network
    calls and, once a run is underway, represents weeks of intended work. It does not
    touch anything already ingested.
    """
    if confirm.strip().upper() != "DISCARD":
        return RedirectResponse(url="/?scrape_err=confirm", status_code=303)

    count = len(backlog_store.target_list())
    backlog_store.clear_backlog()
    logger.warning("Discarded %d queued occupation(s) from the web UI.", count)
    return RedirectResponse(url="/?scrape_ok=discarded", status_code=303)


@app.get("/admin/progress", response_class=HTMLResponse)
def render_progress_fragment(request: Request):
    """
    The progress panel, as its own tiny page for embedding in an iframe.

    Separate document rather than part of the review page because it refreshes itself
    every few seconds while a run is live. Refreshing review.html instead would wipe
    half-typed corrections out of the textareas in the queue below and reset scroll,
    every five seconds, for as long as the scrape runs.
    """
    status = run_state.snapshot()
    return templates.TemplateResponse(
        request,
        "_progress.html",
        {
            "status": status,
            "quota_reset": _quota_reset_note(),
            "queue_size": status.get("queued", 0),
        },
    )


def _quota_reset_note() -> dict:
    """
    The date the daily Gemini quota resets, which is always tomorrow.

    Formatted here rather than in Jinja because strftime("%-d") is not portable off
    glibc and BSD. A date only, never a time: Gemini's free-tier window is Pacific and
    this machine may not be, so the hour would be a guess dressed as a fact.
    """
    tomorrow = run_state.quota_reset_date()
    return {
        "date": tomorrow.isoformat(),
        "label": f"{tomorrow.strftime('%A')}, {tomorrow.strftime('%B')} {tomorrow.day}",
    }


if __name__ == "__main__":
    # Without this block `python3.11 main.py` silently does nothing: it builds the app
    # object above, falls off the end of the file, and exits. Every other runnable module
    # here (ingest.py, dashboardtables.py) already has a __main__ block, so main.py
    # not having one was the surprise.
    #
    # uvicorn is imported here rather than at module top because importing this module
    # (from a test or a probe) should not require a server package.
    #
    # The app OBJECT is passed, not the "main:app" import string: the string form makes
    # uvicorn re-import by module name, which depends on the working directory.
    #
    # Host defaults to loopback, not 0.0.0.0. There is no authentication and the review
    # forms mutate the JSON store, so binding every interface would hand the queue to
    # anything on the same network. Set HOST deliberately if that is actually wanted.
    #
    # No reload=True: it needs the import-string form, runs a supervisor plus a worker,
    # and re-runs initialize_schema() on every save. Use the explicit uvicorn command.
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))

    # Logged before uvicorn.run, which blocks.
    logger.info("Review queue:  http://%s:%d/", host, port)
    logger.info("Dashboard:     http://%s:%d/dashboard", host, port)

    uvicorn.run(app, host=host, port=port)
