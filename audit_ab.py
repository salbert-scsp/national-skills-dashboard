"""
Does batching the credibility audit change its verdicts?

    python3.11 audit_ab.py                # 40 skills, mixed
    python3.11 audit_ab.py --limit 20
    python3.11 audit_ab.py --plan          # show the sample, spend nothing

The audit is issued six skills to a request. That is a 6x cut in the free-tier requests a
run costs, and the objection to it is not cost but attention: a model handed six unrelated
sources might grade them differently than it grades one. This settles that by re-asking,
ONE SKILL PER REQUEST, for skills that already carry a batched verdict, and diffing.

WRITES NOTHING. Not the store, not the cache, not the key state beyond the rotation any
call performs. It is a measurement, and a measurement that repairs what it measures
cannot be run twice.

Read the agreement number, then set GEMINI_AUDIT_BATCH_SIZE accordingly:

    >= 90%    leave it at 6; batching is not what is filling the queue
    75-90%    GEMINI_AUDIT_BATCH_SIZE=3, GEMINI_AUDIT_TOKENS_PER_ITEM=200
    < 75%     GEMINI_AUDIT_BATCH_SIZE=1, and only the second-pass proposal stays batched

One caveat worth holding on to while reading the output: a disagreement is not
automatically the batch being wrong. These are judgement calls on borderline pages, and
two single calls on the same page can disagree with each other too. The number to act on
is the RATE, not any individual flip.
"""

import argparse
import logging
import statistics
import sys

import json_store
from agentic_source_check import (
    AUDIT_BATCH_SIZE,
    AuditUnavailable,
    get_source_checker,
)
from gemini_keys import ROLE_AUDIT, DailyQuotaExhausted

logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s %(name)s: %(message)s")
logger = logging.getLogger("audit_ab")

# Sampled from both sides of the verdict on purpose. Drawing only from auto_approved
# would measure whether the single call also says yes, which is the easy half: a batch
# that had gone soft would pass that test and fail the queue.
DEFAULT_LIMIT = 40
APPROVED_SHARE = 0.6


def select_sample(master: dict, limit: int) -> list:
    """Skills carrying a batched verdict, mixed across both outcomes."""
    approved, failed = [], []
    for name, entry in master.items():
        if not entry.get("resolved_title"):
            continue
        reason = entry.get("gate_reason")
        if reason == "auto_approved" and entry.get("is_credible") is True:
            approved.append((name, entry))
        elif reason == "failed_credibility_audit" and entry.get("is_credible") is False:
            failed.append((name, entry))

    approved.sort(key=lambda row: row[0])
    failed.sort(key=lambda row: row[0])

    want_approved = int(limit * APPROVED_SHARE)
    sample = approved[:want_approved] + failed[:limit - want_approved]
    # Backfill from whichever side has more, so a store thin on failures still returns
    # a full sample rather than silently measuring half as much as it was asked for.
    if len(sample) < limit:
        spare = approved[want_approved:] + failed[limit - want_approved:]
        sample += spare[:limit - len(sample)]
    return sample


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help=f"how many skills to re-audit (default {DEFAULT_LIMIT})")
    parser.add_argument("--plan", action="store_true",
                        help="print the sample and exit without calling Gemini")
    args = parser.parse_args(argv)

    master = json_store.load_master()
    sample = select_sample(master, max(1, args.limit))
    if not sample:
        print("No batched verdicts to re-audit. Run an ingestion first.")
        return 1

    print(f"Sample: {len(sample)} skill(s). Batched at {AUDIT_BATCH_SIZE} per request; "
          f"re-auditing at 1 per request.\n")
    if args.plan:
        for name, entry in sample:
            print(f"  {entry.get('is_credible')!s:<5}  {name}  ->  {entry['resolved_title']}")
        print(f"\nNothing was called. Drop --plan to spend {len(sample)} request(s).")
        return 0

    # Imported here so --plan never loads the ONNX session behind scraping.
    from scraping import fetch_wikipedia_extract

    checker = get_source_checker(ROLE_AUDIT)
    agree = disagree = unusable = 0
    flips = []
    batched_lengths, single_lengths = [], []

    for index, (name, entry) in enumerate(sample, 1):
        title = entry["resolved_title"]
        # Refetched rather than read from the entry: for an approved skill the stored
        # text IS the batched summary, and auditing a summary is not the same question
        # as auditing the page it came from.
        extract = fetch_wikipedia_extract(title)
        if not extract:
            print(f"  [{index}/{len(sample)}] {name}: page returned no text, skipped")
            unusable += 1
            continue

        try:
            result = checker.evaluate_candidates(
                skill_name=name,
                candidates=[{"source_name": "Wikipedia", "raw_text": extract}],
            )
        except DailyQuotaExhausted:
            print("\nAll keys hit their daily quota. Stopping; the numbers below cover "
                  "what was measured before the wall.")
            break
        except AuditUnavailable:
            print(f"  [{index}/{len(sample)}] {name}: audit unavailable, skipped")
            unusable += 1
            continue

        was = bool(entry.get("is_credible"))
        now = bool(result.get("is_credible"))
        if was == now:
            agree += 1
            mark = "same"
        else:
            disagree += 1
            flips.append((name, was, now, title))
            mark = "DIFFERENT"

        batched_lengths.append(len(entry.get("wikipedia_summary") or ""))
        single_lengths.append(len(result.get("clean_summary") or ""))
        print(f"  [{index}/{len(sample)}] {name}: batched={was} single={now}  {mark}")

    compared = agree + disagree
    print("\n" + "=" * 72)
    if not compared:
        print("Nothing was compared.")
        return 1

    rate = 100.0 * agree / compared
    print(f"Agreement: {agree}/{compared} = {rate:.1f}%   ({unusable} skipped)")

    if flips:
        print("\nDisagreements (batched -> single):")
        for name, was, now, title in flips:
            direction = "stricter alone" if was and not now else "softer alone"
            print(f"  {name}  [{title}]  {was} -> {now}  ({direction})")

    if batched_lengths:
        print(f"\nSummary length, median chars: batched {statistics.median(batched_lengths):.0f}, "
              f"single {statistics.median(single_lengths):.0f}")

    print("\nVerdict:")
    if rate >= 90:
        print("  Batching is not changing the audit. Leave GEMINI_AUDIT_BATCH_SIZE at 6.")
        print("  Wrong queue items are a resolver problem: run the second pass over the")
        print("  below_relevance_threshold and failed_credibility_audit buckets.")
    elif rate >= 75:
        print("  Some drift. Set GEMINI_AUDIT_BATCH_SIZE=3 and "
              "GEMINI_AUDIT_TOKENS_PER_ITEM=200, then re-run this.")
    else:
        print("  Batching is changing verdicts. Set GEMINI_AUDIT_BATCH_SIZE=1; the "
              "second-pass proposal can stay batched.")
    print("\nNothing was written. Re-auditing for real is a scrape with --force.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
