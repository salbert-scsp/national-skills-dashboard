"""
Ingestion entry point.

    python3.11 ingest.py                   drain the backlog, then prompt
    python3.11 ingest.py 11- 13- 15-       families
    python3.11 ingest.py 15-2051.00        a single occupation
    python3.11 ingest.py 15- --limit 3     first 3 occupations only
    python3.11 ingest.py 15- --force       re-audit even fresh definitions
    python3.11 ingest.py --skip-backlog    ignore queued work this run

Queued work always runs FIRST. If a previous run stopped on the Gemini daily quota,
the occupations it never reached are in ingestion_backlog.json, and this finishes them
before asking for anything new. If the quota wall is hit again, it stops cleanly, the
remaining work stays queued, and the next run resumes it -- there is nothing to reset
by hand.

Two constraints that are invisible from the code and have both cost real time:

  - It must be `python3.11`. The default `python3` on this machine is 3.9.6 and has
    neither onnxruntime nor google-genai installed, so it fails at import.
  - It must run from the workspace directory. model.onnx, tokenizer.json and both
    JSON store files are resolved as relative paths.

This writes skills_master.json and skills_timeseries.json. The UI is separate:

    python3.11 main.py            # then open http://127.0.0.1:8000/dashboard

The web UI (main.py) reads the same two files, and its ingest form can trigger this
same pipeline in the background.
"""

import argparse
import logging
import sys

import backlog as backlog_store
from definitions_algorithm import normalize_targets, run_backlog, run_targets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Shown when prompting, so the prefixes are not something to look up elsewhere.
FAMILY_HINTS = [
    ("11-", "Management"),
    ("13-", "Business and Financial Operations"),
    ("15-", "Computer and Mathematical"),
    ("17-", "Architecture and Engineering"),
    ("19-", "Life, Physical, and Social Science"),
    ("23-", "Legal"),
    ("25-", "Educational Instruction and Library"),
    ("27-", "Arts, Design, Entertainment, Sports, and Media"),
    ("29-", "Healthcare Practitioners and Technical"),
]


def prompt_for_prefixes() -> list:
    """
    Asks for comma-separated SOC prefixes.

    Returns an empty list on EOF or an empty answer, so running with no arguments in
    a non-interactive context exits cleanly instead of raising EOFError.
    """
    print("\nSOC major groups to ingest. Common ones:\n")
    for prefix, label in FAMILY_HINTS:
        print(f"    {prefix}  {label}")
    print()

    try:
        answer = input("Targets (comma-separated, e.g. 11-,13- or 15-2051.00): ").strip()
    except EOFError:
        return []

    return [part for part in answer.replace(" ", "").split(",") if part]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest O*NET tools and technologies into the local JSON store."
    )
    parser.add_argument(
        "prefixes", nargs="*", metavar="TARGET",
        help="SOC family prefixes (11- 13-) or full O*NET codes (15-2051.00). "
             "Prompts if omitted.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most this many occupations. Use it on a first run against a "
             "new family; some families hold dozens of occupations.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-scrape and re-audit even definitions inside the 90-day freshness "
             "window. Spends Gemini quota on skills that did not need it.",
    )
    parser.add_argument(
        "--skip-backlog", action="store_true",
        help="Ignore queued work from a previous interrupted run. The backlog is left "
             "untouched, not discarded.",
    )
    args = parser.parse_args(argv)

    if args.force:
        logger.warning(
            "--force is set: every hot skill will be re-audited regardless of "
            "freshness, which spends Gemini quota."
        )

    # ---- Backlog first -------------------------------------------------------
    #
    # Work owed from a previous run is finished BEFORE anything new is asked for.
    # Prompting first would let a fresh request jump the queue and, on a day where
    # quota is tight, mean the queued work never runs at all.
    if not backlog_store.is_empty() and not args.skip_backlog:
        print(f"\nResuming queued work. {backlog_store.describe()}\n")
        drain = run_backlog(force=args.force)

        if drain["quota_exhausted"]:
            return _exhausted(drain["remaining"])

        print(
            f"\nBacklog drained: {drain['processed']} target(s) completed"
            + (f", {drain['remaining']} still queued." if drain["remaining"] else ".")
            + "\n"
        )
    elif args.skip_backlog and not backlog_store.is_empty():
        logger.warning(
            "--skip-backlog: %d queued target(s) left untouched.",
            len(backlog_store.target_list()),
        )

    # ---- Then anything new ---------------------------------------------------
    raw_targets = args.prefixes or prompt_for_prefixes()
    if not raw_targets:
        # Not an error when the backlog just did real work; that WAS the run.
        if not backlog_store.is_empty() or not args.prefixes:
            print("Nothing further requested.\n")
            return 0
        logger.error("No SOC prefixes given. Nothing to do.")
        return 2

    try:
        targets = normalize_targets(raw_targets)
    except ValueError as err:
        logger.error("%s", err)
        return 2

    try:
        result = run_targets(targets, limit=args.limit, force=args.force)
    except Exception:
        logger.exception("Ingestion failed.")
        return 1

    if result.get("quota_exhausted"):
        return _exhausted(len(backlog_store.target_list()))

    if not result.get("occupations"):
        logger.error("No occupations were processed.")
        return 1

    print(
        f"\nDone. {result['occupations']} occupations processed, "
        f"{result.get('skills', 0)} skills in skills_master.json.\n"
        f"Review and explore with:  python3.11 main.py\n"
    )
    return 0


def _exhausted(remaining: int) -> int:
    """
    Reports the daily quota wall and exits without prompting for more.

    Exit code 2, matching the other "you asked for something that cannot run" paths.
    Deliberately does NOT prompt: offering to queue more work on a day when nothing can
    be audited is the wrong end of the interaction, and the queued work is safe on disk
    either way.
    """
    print(
        "\n"
        "  ALL GEMINI TOKENS EXHAUSTED FOR TODAY\n"
        "\n"
        f"  Every configured key has hit its daily quota. {remaining} target(s) are\n"
        "  queued and will be picked up automatically the next time you run:\n"
        "\n"
        "      python3.11 ingest.py\n"
        "\n"
        "  Nothing was left half-approved. Skills that were never audited were not\n"
        "  written to the store or the review queue.\n"
        "\n"
        "  Add more keys to gemini_keys.json to keep going today.\n"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
