"""
STAGE: Review & Triage

Sends model-written definitions back to be rewritten under the current DEFINITION_SPEC.

    python3.11 redraft_definitions.py                  # dry run, writes nothing
    python3.11 redraft_definitions.py --apply
    python3.11 redraft_definitions.py --include-approved --apply

WHY: the drafting rules changed. Definitions written before agentic_source_check's
DEFINITION_SPEC run about 150 characters and name a category, a vendor and an audience:

    "DynaSCAPE Design is a computer-aided design software developed by DynaSCAPE
     Software specifically for landscape architects and professional designers."

where the spec now asks for 200-400 covering the SPECIFIC technical category, vendor
lineage, and the concrete tasks the thing is used for -- the shape a Wikipedia lead has.
That is not only a readability difference: the definition is the text the bi-encoder
scores, so a thin one measures thin against every anchor.

This does the same thing the review page's "Reject this draft and rewrite it" button
does, in bulk, for drafts nobody wants to click through one at a time.

TWO SCOPES, because they are not the same decision:

  - PENDING drafts (default). The card is still in the queue and no one has committed to
    the text. Sending it back costs nothing but the redraft.
  - APPROVED model-authored definitions (--include-approved). These are ON THE DASHBOARD.
    Sending one back returns the skill to PENDING, so it leaves the dashboard until it is
    redrafted and re-approved. That is a real change to what the dashboard shows and is
    why it is opt-in rather than the default.

Nothing here calls Gemini. It clears the settled state so the NEXT second pass rewrites
these; run second_pass afterwards to actually get the new text.
"""

import argparse
import logging
import sys
from typing import Any, Dict, List, Tuple

import json_store
from json_store import STATUS_APPROVED, STATUS_PENDING
from review_actions import (
    MODEL_AUTHORED_SOURCE_NAME,
    _apply_reject_draft,
    onet_boilerplate,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("redraft")


def find_pending_drafts(master: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Pending cards offering a drafted definition, or already carrying one."""
    found = []
    for name, entry in master.items():
        if entry.get("status") != STATUS_PENDING:
            continue
        block = entry.get("second_pass")
        offered = isinstance(block, dict) and (block.get("drafted_definition") or "").strip()
        carried = entry.get("best_source_name") == MODEL_AUTHORED_SOURCE_NAME
        if offered or carried:
            found.append((name, entry))
    found.sort(key=lambda row: row[0])
    return found


def find_no_article(master: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Pending entries settled as no_article, which the second pass will never revisit.

    These are terminal, so they sit in the queue permanently: 135 of them, of which 111
    were settled on 2026-08-11 and 2026-08-12 under the old drafting rules and before
    DEFINITION_SPEC existed. Reopening gives them exactly one pass under the current
    rules. `declined_to_define` is deliberately NOT included -- the model was asked and
    said it does not know the product, and asking the same question again is the waste
    the terminal rule exists to prevent.
    """
    found = [
        (name, entry) for name, entry in master.items()
        if entry.get("status") == STATUS_PENDING
        and isinstance(entry.get("second_pass"), dict)
        and entry["second_pass"].get("outcome") == "no_article"
    ]
    found.sort(key=lambda row: row[0])
    return found


def reopen(name: str, entry: Dict[str, Any]) -> None:
    """
    Clears the settled state so the second pass picks the entry up again.

    Same operation _apply_reject_draft performs on a draft: a non-terminal outcome and a
    reset attempt count, which together are what select_candidates reads. The audit trail
    in the block is kept.
    """
    block = entry["second_pass"]
    block["outcome"] = "draft_rejected"
    block["terminal"] = False
    block["attempts"] = 0
    entry["second_pass"] = block
    # The tag is the only thing that silences an entry, so reopening means removing it.
    entry.pop("ai_status", None)
    entry.pop("ai_status_detail", None)
    entry["last_updated"] = json_store.today().isoformat()


def find_approved_drafts(master: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Approved skills whose definition was written by the model."""
    found = [
        (name, entry) for name, entry in master.items()
        if entry.get("status") == STATUS_APPROVED
        and entry.get("best_source_name") == MODEL_AUTHORED_SOURCE_NAME
    ]
    found.sort(key=lambda row: row[0])
    return found


def send_approved_back(name: str, entry: Dict[str, Any]) -> None:
    """
    Returns an approved model-authored skill to the queue for redrafting.

    Its snapshot is deliberately NOT deleted. The time series records what the skill
    scored while it was approved, and erasing that would rewrite history rather than
    continue it; the dashboard filters on status, so the skill leaves the view either
    way. When it is redrafted and re-approved, the new score lands as the next point.
    """
    block = entry.get("second_pass")
    if isinstance(block, dict):
        existing = (block.get("drafted_definition") or "").strip()
        if existing or entry.get("wikipedia_summary"):
            block["rejected_definition"] = existing or entry.get("wikipedia_summary")
        block["drafted_definition"] = None
        block["outcome"] = "draft_rejected"
        block["terminal"] = False
        block["attempts"] = 0
        entry["second_pass"] = block

    entry.pop("ai_status", None)
    entry.pop("ai_status_detail", None)
    entry["status"] = STATUS_PENDING
    entry["wikipedia_summary"] = onet_boilerplate(name, entry.get("category") or "")
    entry["best_source_name"] = None
    entry["resolved_title"] = None
    entry["reference_url"] = None
    entry["is_credible"] = None
    entry["cross_score"] = None
    entry["gate_reason"] = "no_candidate_found"
    entry["last_updated"] = json_store.today().isoformat()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--apply", action="store_true",
                        help="write the changes; without it nothing is saved")
    parser.add_argument("--include-approved", action="store_true",
                        help="also return APPROVED model-authored skills to the queue")
    parser.add_argument("--include-no-article", action="store_true",
                        help="also reopen pending items settled as no_article, which the "
                             "second pass will otherwise never revisit")
    parser.add_argument("--show", type=int, default=10, help="how many to list (default 10)")
    args = parser.parse_args(argv)

    master = json_store.load_master()
    pending = find_pending_drafts(master)
    approved = find_approved_drafts(master) if args.include_approved else []
    stuck = find_no_article(master) if args.include_no_article else []

    print(f"Pending drafts to rewrite:  {len(pending)}")
    print(f"Approved to send back:      {len(approved)}"
          f"{'' if args.include_approved else '   (use --include-approved)'}")
    print(f"no_article to reopen:       {len(stuck)}"
          f"{'' if args.include_no_article else '   (use --include-no-article)'}\n")

    for label, rows in (("pending", pending), ("approved", approved), ("no_article", stuck)):
        for name, entry in rows[:args.show]:
            block = entry.get("second_pass") or {}
            text = (block.get("drafted_definition") or entry.get("wikipedia_summary") or "")
            print("  [%s] %-38s %3d chars  %s" % (
                label, name[:38], len(text.strip()), text.strip()[:60]
            ))
        if len(rows) > args.show:
            print(f"  ... and {len(rows) - args.show} more {label}")

    if not (pending or approved or stuck):
        print("Nothing to redraft.")
        return 0

    if not args.apply:
        print("\nDry run. Nothing was written. Re-run with --apply.")
        if not args.include_no_article and find_no_article(master):
            print(f"NOTE: {len(find_no_article(master))} pending items are settled as "
                  f"no_article and are TERMINAL, so the second pass will never revisit "
                  f"them. --include-no-article reopens them.")
        if not args.include_approved and find_approved_drafts(master):
            print(f"NOTE: {len(find_approved_drafts(master))} APPROVED skills also carry a "
                  f"model-written definition. --include-approved returns them to the queue, "
                  f"which removes them from the dashboard until they are re-approved.")
        return 0

    sent_back = 0
    failed = []
    for name, _ in pending:
        ok, code = _apply_reject_draft(master, name)
        if ok:
            sent_back += 1
        else:
            failed.append((name, code))

    for name, entry in approved:
        send_approved_back(name, entry)

    for name, entry in stuck:
        reopen(name, entry)

    json_store.save_master(master)

    print(f"\n{sent_back} pending draft(s) sent back to be rewritten.")
    if approved:
        print(f"{len(approved)} approved skill(s) returned to the queue; they have left the "
              f"dashboard until redrafted and re-approved. Their snapshots were kept.")
    if stuck:
        print(f"{len(stuck)} no_article item(s) reopened; the second pass will look at them "
              f"again under the current definition rules.")
    for name, code in failed:
        print(f"  skipped {name}: {code}")
    print("\nNow run the second pass to write the new definitions:")
    print("  the review page's 'Propose changes' button, or second_pass.run_second_pass()")
    return 0


if __name__ == "__main__":
    sys.exit(main())
