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

import json
import logging
import os
import re
from contextlib import asynccontextmanager
from urllib.parse import quote

from fastapi import BackgroundTasks, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup

import auth
import backlog as backlog_store
import run_state
from dashboardtables import build_review_rows, load_dashboard
from json_store import (
    STATUS_APPROVED,
    STATUS_PENDING,
    STATUS_REJECTED,
    latest_snapshots,
    load_master,
    load_timeseries,
)
from review_actions import (
    REPORT_REASONS,
    apply_machine_draft,
    apply_machine_suggestion,
    approve_skill,
    edit_approved_skill,
    reject_skill,
    remediate_skill,
    report_skill,
)
from scraping import CROSS_ENCODER_THRESHOLD, REMEDIATION_ERROR_CODES

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

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


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
        return await call_next(request)

    if auth.valid(request.cookies.get(auth.COOKIE_NAME)):
        return await call_next(request)

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
    }
    try:
        rows = build_review_rows(load_master(), STATUS_PENDING)
        total_pending = len(rows)

        # max(1, ...) so an empty queue is "page 1 of 1" rather than "page 1 of 0".
        total_pages = max(1, -(-total_pending // PAGE_SIZE))
        page = min(max(page, 1), total_pages)
        offset = (page - 1) * PAGE_SIZE
        items = rows[offset:offset + PAGE_SIZE]

        for item in items:
            item["gate_label"] = GATE_REASON_LABELS.get(
                item.get("gate_reason"), item.get("gate_reason") or "Flagged"
            )
            # The template renders one occupation line per card; the store holds a
            # list, so surface the first as the representative and the count with it.
            occupations = item.get("occupations") or []
            item["onet_code"] = occupations[0]["onet_code"] if occupations else ""
            item["onet_title"] = occupations[0]["onet_title"] if occupations else ""
            item["occupation_count"] = len(occupations)

        return {
            "items": items,
            "page": page,
            "total_pages": total_pages,
            "total_pending": total_pending,
            "first_index": offset + 1 if items else 0,
            "last_index": offset + len(items),
            "window": _page_window(page, total_pages),
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
        error_message = REMEDIATION_ERRORS.get(err, GENERIC_REMEDIATION_ERROR)
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
            "apply": "Better pages will be applied in place. Nothing will be approved.",
            "auto": "Better pages will be applied, and clean matches approved automatically.",
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
            "second_pass_ok": second_pass_message,
            "second_pass_err": "",
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

    try:
        result = run_second_pass(limit=limit, apply_mode=apply_mode)
    except Exception:
        logger.exception("Background second pass failed.")
        return

    if result.get("quota_exhausted"):
        logger.error(
            "ALL GEMINI TOKENS EXHAUSTED FOR TODAY. The second pass stopped after %d "
            "item(s); %d were left untouched and will be picked up on the next run.",
            result.get("processed", 0), result.get("remaining", 0),
        )


@app.post("/second-pass")
def handle_second_pass_form(
    background_tasks: BackgroundTasks,
    limit: str = Form(str(SECOND_PASS_DEFAULT_LIMIT)),
    apply_mode: str = Form("suggest"),
):
    """
    Queues a second pass over the pending queue.

    apply_mode is validated against a fixed tuple rather than passed through, and both
    the default and the fallback are the mode that changes nothing. The three modes
    differ in how much a machine may do unattended, so an unrecognized value has to land
    on the least of them rather than on whatever the form happened to send.
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
        logger.warning(
            "Scheduling a second pass over %d item(s) that will REWRITE cards in place. "
            "Nothing will be approved.", count
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
