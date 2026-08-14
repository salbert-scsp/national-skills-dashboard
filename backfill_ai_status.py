"""
Writes the ai_status tag onto entries settled before the tag existed.

    python3.11 backfill_ai_status.py            # dry run, writes nothing
    python3.11 backfill_ai_status.py --apply

WHY: the second pass used to skip anything whose outcome was in TERMINAL_OUTCOMES, which
was invisible -- a queue the automation had given up on looked exactly like one it had
not reached. Skipping is now driven by an explicit `ai_status` tag on the entry, so an
untagged entry is always re-attempted.

That inversion is the point, and it means every entry settled under the old rule is
currently untagged. Without this backfill the next run would re-ask ALL of them,
including the ones already holding an answer a reviewer has not clicked yet -- spending
quota to overwrite proposals nobody has read.

WHAT IT WRITES, and the distinction that matters:

  AWAITING_REVIEW   the model produced an answer and is waiting on a human: a drafted
                    definition, a suggested page, a confirmed original. Tagged, so it is
                    not re-asked, because re-asking would overwrite the proposal.

  CANNOT_DETERMINE  the model looked and gave up: not_a_product, unresolvable. Tagged and
                    badged on the card, which is how "these need a person" becomes
                    visible.

DELIBERATELY LEFT UNTAGGED: no_article and declined_to_define. Those were settled before
DEFINITION_SPEC and before no_article learned to clear the page it contradicts, so they
have earned exactly one attempt under the current rules. Leaving them untagged IS that
retry -- it is the same reopen redraft_definitions.py performs, expressed as an absence.
Whatever they settle to next time will carry a tag.
"""

import argparse
import logging
import sys
from collections import Counter
from typing import Any, Dict

import json_store
from json_store import STATUS_PENDING
from second_pass import (
    AI_AWAITING_REVIEW,
    AI_CANNOT_DETERMINE,
    NO_ARTICLE_OUTCOMES,
    ai_status_for,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("backfill")


def plan(master: Dict[str, Any]) -> Dict[str, list]:
    """
    What each pending entry would be tagged. Returns {action: [(name, outcome), ...]}.

    Pure: nothing is written here, so the dry run and the apply see the same decisions.
    """
    result = {"awaiting": [], "cannot": [], "retry": [], "already_tagged": []}

    for name, entry in master.items():
        if entry.get("status") != STATUS_PENDING:
            continue
        block = entry.get("second_pass")
        if not isinstance(block, dict):
            continue

        outcome = block.get("outcome")
        if entry.get("ai_status"):
            result["already_tagged"].append((name, outcome))
            continue

        # The one-shot retry, spelled out rather than falling out of the map.
        if outcome in NO_ARTICLE_OUTCOMES:
            result["retry"].append((name, outcome))
            continue

        status = ai_status_for(outcome)
        if status == AI_AWAITING_REVIEW:
            result["awaiting"].append((name, outcome))
        elif status == AI_CANNOT_DETERMINE:
            result["cannot"].append((name, outcome))
        else:
            result["retry"].append((name, outcome))

    for rows in result.values():
        rows.sort()
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--apply", action="store_true",
                        help="write the tags; without it nothing is saved")
    parser.add_argument("--show", type=int, default=8, help="how many to list per group")
    args = parser.parse_args(argv)

    master = json_store.load_master()
    grouped = plan(master)

    print(f"Already tagged, untouched : {len(grouped['already_tagged'])}")
    print(f"-> awaiting_review        : {len(grouped['awaiting'])}"
          f"   (has an answer; will not be re-asked)")
    print(f"-> cannot_determine       : {len(grouped['cannot'])}"
          f"   (badged 'AI cannot determine')")
    print(f"left untagged, retried    : {len(grouped['retry'])}"
          f"   (one pass under the current rules)\n")

    for label in ("awaiting", "cannot", "retry"):
        rows = grouped[label]
        if not rows:
            continue
        print(f"  {label}:")
        counts = Counter(outcome for _, outcome in rows)
        for outcome, count in counts.most_common():
            print("    %-34s %d" % (outcome or "(none)", count))
        for name, outcome in rows[:args.show]:
            print("      %-44s %s" % (name[:44], outcome))
        if len(rows) > args.show:
            print(f"      ... and {len(rows) - args.show} more")
        print()

    if not args.apply:
        print("Dry run. Nothing was written. Re-run with --apply.")
        return 0

    for name, outcome in grouped["awaiting"]:
        master[name]["ai_status"] = AI_AWAITING_REVIEW
        master[name]["ai_status_detail"] = outcome
    for name, outcome in grouped["cannot"]:
        master[name]["ai_status"] = AI_CANNOT_DETERMINE
        master[name]["ai_status_detail"] = outcome

    json_store.save_master(master)
    tagged = len(grouped["awaiting"]) + len(grouped["cannot"])
    print(f"Tagged {tagged} entr(ies). {len(grouped['retry'])} left untagged and will be "
          f"re-attempted by the next second pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
