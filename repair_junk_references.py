"""
STAGE: Review & Triage

Clears reference pages that were never about the skill they are attached to.

    python3.11 repair_junk_references.py            # dry run, writes nothing
    python3.11 repair_junk_references.py --apply

The resolver used to keep whatever scored highest, however low that was. The search API
always answers, so for a discontinued vendor tool with no article it answered with
whatever shared a word with the query: DynaSCAPE Design got "List of ZX Spectrum games"
at 0.000, ConEst SureCount got "DARPA". The card then showed that article's text as the
skill's definition. scraping.RESOLUTION_FLOOR stops it happening again; this fixes the
entries already carrying one.

WHAT IT TOUCHES: pending entries whose gate_reason is below_relevance_threshold and
whose cross_score is under the floor. It resets them to the no_candidate_found shape --
no title, no borrowed text, O*NET boilerplate as the summary -- which is what the
resolver would produce for them today.

WHAT IT LEAVES ALONE, and each for its own reason:
  - anything approved or rejected: a decision has been made and this is not a re-decision
  - reviewer_remediated and machine_remediated: the page was supplied, not guessed, and
    the score on those is recorded rather than enforced precisely because a low one can
    be correct (PySpark against Apache Spark measures 0.0)
  - anything at or above the floor: the resolver still stands behind it
  - the second_pass block: it is an audit trail of what was asked and answered, and a
    finding about a page is still a finding after the page is cleared

The skill is NOT dropped. It keeps its card, keeps its boilerplate text, and lands in
the bucket the second pass writes a real definition for.
"""

import argparse
import logging
import sys

import json_store
from json_store import STATUS_PENDING
from scraping import RESOLUTION_FLOOR

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("repair")

REPAIRABLE_GATE_REASON = "below_relevance_threshold"


def boilerplate(skill_name: str, category: str) -> str:
    """The same sentence resolve_and_gate_skill writes for a skill with no page."""
    return (
        f"{skill_name} is a hot technology asset categorized under "
        f"{category or 'Technology'} in the O*NET framework."
    )


def find_junk(master: dict) -> list:
    """Pending entries whose reference page scored below the floor."""
    found = []
    for name, entry in master.items():
        if entry.get("status") != STATUS_PENDING:
            continue
        if entry.get("gate_reason") != REPAIRABLE_GATE_REASON:
            continue
        score = entry.get("cross_score")
        if score is None or score >= RESOLUTION_FLOOR:
            continue
        found.append((name, entry))
    found.sort(key=lambda row: row[0])
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--apply", action="store_true",
                        help="write the changes; without it nothing is saved")
    parser.add_argument("--show", type=int, default=15,
                        help="how many to print (default 15)")
    args = parser.parse_args(argv)

    master = json_store.load_master()
    junk = find_junk(master)

    print(f"Floor: {RESOLUTION_FLOOR:.2f}. Pending entries citing a page below it: {len(junk)}\n")
    if not junk:
        print("Nothing to repair.")
        return 0

    for name, entry in junk[:args.show]:
        print("  %.3f  %-44s -> %s" % (
            entry.get("cross_score") or 0.0, name[:44], entry.get("resolved_title")
        ))
    if len(junk) > args.show:
        print(f"  ... and {len(junk) - args.show} more")

    if not args.apply:
        print(f"\nDry run. Nothing was written. Re-run with --apply to clear these "
              f"{len(junk)} reference(s).")
        return 0

    for name, entry in junk:
        entry["resolved_title"] = None
        entry["reference_url"] = None
        entry["cross_score"] = None
        entry["is_credible"] = None
        entry["best_source_name"] = None
        entry["wikipedia_summary"] = boilerplate(name, entry.get("category") or "")
        entry["gate_reason"] = "no_candidate_found"

    json_store.save_master(master)
    print(f"\nCleared {len(junk)} reference(s). Every one of those skills is still in the "
          f"queue, now asking for a definition rather than citing the wrong article.")
    print("Run the second pass over them to have definitions drafted:")
    print("  the review page's 'Propose changes' button, or second_pass.run_second_pass()")
    return 0


if __name__ == "__main__":
    sys.exit(main())
