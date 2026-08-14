"""
Applies the rules engine to snapshots that were written before it existed.

    python3.11 reclassify_snapshots.py            # dry run, writes nothing
    python3.11 reclassify_snapshots.py --apply

Every input the engine reads is ALREADY stored on each snapshot. Classification is a pure
function of them, so this re-derives the class without loading the model, re-embedding
anything, or making a single network call. It is the cheap half of the change;
embedding_pass.py and flagship_pass.py are the halves that cost searches and requests.

WHAT IT CHANGES: category_bucket, sub_category, the decision_* fields and
classification_confidence. The measured metrics are NOT touched -- this does not
re-score, it re-decides, so the time series stays comparable across the change.

embeds_ai IS READ AND NEVER INVENTED. A snapshot written before the probe existed has no
embeds_ai, which reads as None: nobody established it, so it earns no boost and fires no
rule. That is why running this before embedding_pass.py is safe and also why it is not
sufficient -- it will show the old Embedded AI class collapsing to nothing, and the
skills that genuinely belong there only come back once they have been searched for.

THE BOOST IS NOT RE-APPLIED HERE. ai_score on an existing snapshot is whatever was
measured and stored, and adding EMBEDDED_AI_BOOST to it would compound on every run.
Re-scoring with the boost is embedding_pass's job, which recomputes from the definition.

WHAT IT LEAVES ALONE: any snapshot missing one of the four metrics. A skill scored before
the three enabling sims existed cannot be classified by rules that read them, and
defaulting the missing ones to 0.0 would silently file it as Not AI.
"""

import argparse
import logging
import sys
from collections import Counter
from typing import Any, Dict, List

import json_store
from sortingalgorithmnew import classify

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("reclassify")

REQUIRED_METRICS = ("ai_score", "tech_base_sim", "ml_pipeline_sim", "embedded_ai_sim")


def classifiable(record: Dict[str, Any]) -> bool:
    return all(record.get(metric) is not None for metric in REQUIRED_METRICS)


def reclassify(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Re-decides every classifiable snapshot in place. Returns a report; writes nothing.

    Mutating in place is safe because the caller either saves the list or discards it,
    and the alternative -- rebuilding the list -- risks dropping a record whose shape
    this function did not anticipate.
    """
    before = Counter()
    after = Counter()
    transitions = Counter()
    subcategories = Counter()
    marginal = 0
    skipped = 0

    for record in records:
        if not classifiable(record):
            skipped += 1
            continue

        old_bucket = record.get("category_bucket")
        verdict = classify(
            record["ai_score"],
            record["tech_base_sim"],
            record["ml_pipeline_sim"],
            record["embedded_ai_sim"],
            record.get("embeds_ai"),
            # None on a snapshot written before the engineering floor landed, which
            # classify() reads as "skip the floor" rather than "fail it". That is the
            # right default here: this script re-DECIDES stored numbers and must not
            # demote a record on the strength of a field it never had. Those records
            # need a real rescore, not a reclassification.
            record.get("ai_engineering_sim"),
        )
        record.update(verdict)

        before[old_bucket] += 1
        after[verdict["category_bucket"]] += 1
        transitions[(old_bucket, verdict["category_bucket"])] += 1
        subcategories[verdict["sub_category"] or "(none)"] += 1
        if verdict["in_semantic_variance_band"]:
            marginal += 1

    return {
        "before": before,
        "after": after,
        "transitions": transitions,
        "subcategories": subcategories,
        "marginal": marginal,
        "skipped": skipped,
        "total": len(records),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--apply", action="store_true",
                        help="write the reclassified snapshots; without it nothing is saved")
    args = parser.parse_args(argv)

    timeseries = json_store.load_timeseries()
    report = reclassify(timeseries)

    print(f"Snapshots: {report['total']}, of which {report['skipped']} lack a metric "
          f"the engine reads and were left alone.\n")

    print("Class counts")
    for bucket in ("AI Skill", "AI Enabling Skill", "Not AI Skill"):
        print("  %-20s %4d -> %4d" % (bucket, report["before"][bucket], report["after"][bucket]))

    print("\nSub-categories")
    for name, count in report["subcategories"].most_common():
        print("  %-28s %4d" % (name, count))

    print("\nTransitions")
    for (old, new), count in sorted(report["transitions"].items(), key=lambda item: -item[1]):
        arrow = "  " if old == new else "->"
        print("  %-20s %s %-20s %4d" % (old, arrow, new, count))

    classified = report["total"] - report["skipped"]
    share = (100.0 * report["marginal"] / classified) if classified else 0.0
    print(f"\nWithin the semantic variance band: {report['marginal']} of {classified} "
          f"({share:.0f}%). Those are decisions a small shift in the embedding would flip.")

    if not args.apply:
        print("\nDry run. Nothing was written. Re-run with --apply to save.")
        return 0

    json_store.save_timeseries(timeseries)
    print(f"\nSaved. {classified} snapshot(s) reclassified; no measured metric was changed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
