"""
STAGE: Scoring Passes

Scores generic and enterprise skills as the version people actually deploy.

    python3.11 flagship_pass.py                 # dry run, writes nothing, spends nothing
    python3.11 flagship_pass.py --preview 12    # ask Gemini about 12, still write nothing
    python3.11 flagship_pass.py --apply         # ask, store, and rescore

WHY: O*NET names generic categories ("Word processing software") and enterprise suites
whose encyclopedia definitions predate the AI features now shipped in them. Scoring that
text measures the product as it was. Microsoft Teams reads embedded_ai_sim 0.191 from a
definition written before Copilot existed; asked about the flagship deployment it reads
0.451, which is the difference between Not AI and AI Enabling.

WHAT IT SPENDS: one request per FLAGSHIP_BATCH_SIZE eligible skills. Eligibility is
deliberately narrow -- see is_eligible -- because a request spent on a skill whose class
cannot change is a request wasted.

WHAT IT WRITES: the flagship fields on the master entry, then re-scores the skill so the
result reaches the time series. Two different things happen depending on the name:

  - a NAMED PRODUCT gets the note applied to embedded_ai_sim only; ai_score,
    tech_base_sim and ml_pipeline_sim are untouched by construction
  - a CATEGORY TERM is scored as its flagship product outright, because a category
    article outscores every product in its category ("Word processing software" ai 0.321
    against Microsoft Word's 0.192)

A skill the model declines is recorded as evaluated-and-declined so the next run does not
pay to ask again.

SCOPE, and a limitation worth knowing: only APPROVED skills are eligible, because only
those are on the dashboard. definitions_algorithm.record_snapshot applies the note at
approval time but NOT the category substitution, so a category approved after this ran
will be measured on its own text until this is run again. The pass is idempotent, so
re-running it after a batch of approvals is the intended remedy rather than a workaround.
"""

import argparse
import logging
import re
import sys
from typing import Any, Dict, List, Tuple

import json_store
from agentic_source_check import (
    FLAGSHIP_BATCH_SIZE,
    AuditUnavailable,
    get_source_checker,
)
from gemini_keys import ROLE_FLAGSHIP, DailyQuotaExhausted
from json_store import STATUS_APPROVED
from sortingalgorithmnew import (
    FLAGSHIP_METRICS,
    calculate_ai_correlation,
    enforce_flagship_ceiling,
    score_as_flagship,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("flagship")

# Names that denote a category rather than a product, which is exactly the case the
# flagship rule exists for. Matched on the whole name, so "Adobe Photoshop" is not caught
# by "software" appearing nowhere in it.
GENERIC_TERM = re.compile(
    r"\b(software|systems?|suite|platform|tools?|application|programs?|processing)\b",
    re.IGNORECASE,
)


def is_eligible(entry: Dict[str, Any], snapshot: Dict[str, Any]) -> bool:
    """
    Whether asking about this skill could change its class.

    ELIGIBILITY GUARDRAIL, to spend requests only where an answer can move something.
    A skill qualifies when its NAME LOOKS LIKE A GENERIC CATEGORY, which is the case
    flagship substitution exists for, regardless of where it currently scores.

    NARROWED, deliberately. This used to have a second branch: a skill whose
    embedded_ai_sim sat just under EMBEDDED_AI_FLOOR qualified, because a flagship note
    could carry it across and change its class. That floor no longer exists. Embedded AI
    is now established by searching -- see embedding_pass.py -- and embedded_ai_sim
    decides nothing, so the note can only move a number a reader looks at. That is worth
    displaying and it is not worth a Gemini request per skill, and this pass no longer
    spends one on it.

    What remains is the job only this pass does: a category article scores its own
    abstract vocabulary rather than the software anyone uses, and "Word processing
    software" reads ai 0.321 against Microsoft Word's 0.192 for that reason alone.

    Two exclusions:
      - already evaluated: the answer is on file, including a declined one
      - no snapshot: it has never been scored, so there is nothing to move

    An existing AI Skill is NOT excluded. Substitution can lower a category to its
    flagship, and several of the store's AI Skills are category names that this pass
    exists to correct downward.
    """
    if entry.get("status") != STATUS_APPROVED or not snapshot:
        return False
    if entry.get("is_flagship_version_evaluated") is not None:
        return False

    return bool(GENERIC_TERM.search(entry.get("skill_name") or ""))


def find_eligible(master: Dict[str, Any], timeseries: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    newest = json_store.latest_snapshots(timeseries)
    found = [
        (name, entry) for name, entry in master.items()
        if is_eligible(entry, newest.get(name) or {})
    ]
    found.sort(key=lambda row: row[0])
    return found


def chunks(items: List[Any], size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def normalize_product_name(name: str) -> str:
    """Casefolded, punctuation-stripped form, for matching a flagship to a store skill."""
    return re.sub(r"[^a-z0-9]+", " ", str(name or "").lower()).strip()


def resolve_flagship(flagship_version: str, master: Dict[str, Any]) -> str:
    """
    Matches a model-named flagship to a scored skill already in the store, or "".

    EXACT normalized match only, deliberately. A fuzzy match here does not produce a
    slightly-off score, it substitutes an entire unrelated product's four metrics into a
    category term -- "Version control software" landing on "Microsoft Visual SourceSafe"
    because both contain a word. Paying for one definition is much cheaper than that,
    so anything short of exact falls through to the model's own text.
    """
    if not flagship_version:
        return ""
    target = normalize_product_name(flagship_version)
    if not target:
        return ""
    for name in master:
        if normalize_product_name(name) == target:
            return name
    return ""


def apply_description(entry: Dict[str, Any], description: Dict[str, Any]) -> bool:
    """
    Records the answer on the entry. Returns whether a usable note was stored.

    A declined skill is still marked evaluated. The distinction that matters later is
    "asked and there is nothing" versus "never asked", and collapsing them would make
    every subsequent run pay to re-ask the same declined products.
    """
    evaluated = bool(description.get("is_flagship_version_evaluated"))
    summary = str(description.get("embedded_ai_summary") or "").strip()

    entry["is_generic_category"] = bool(description.get("is_generic_category"))
    entry["flagship_version"] = description.get("flagship_version") or None
    entry["flagship_definition"] = description.get("flagship_definition") or None

    if not evaluated or not summary:
        entry["is_flagship_version_evaluated"] = False
        entry["flagship_note"] = ""
        return False

    entry["is_flagship_version_evaluated"] = True
    entry["flagship_note"] = summary
    return True


def score_entry(
    name: str,
    entry: Dict[str, Any],
    master: Dict[str, Any],
    newest: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Scores one skill, substituting its flagship when the name is a category.

    Three paths, in order:

      1. CATEGORY whose flagship is itself a scored skill -- reuse that record's metrics
         outright. Costs nothing extra and guarantees the two agree exactly, which
         scoring the same product from two different texts would not.
      2. CATEGORY whose flagship is not in the store -- score the model's definition of
         the flagship.
      3. Anything else, including every named product -- measure the skill's own text,
         with the note raising embedded_ai_sim only.

    A category with no usable flagship falls through to path 3 rather than being
    half-substituted. Leaving it measured as it always was is honest; a partial
    substitution would be a score nothing stands behind.
    """
    note = entry.get("flagship_note", "") or ""
    flagship_version = entry.get("flagship_version") or None
    # The CATEGORY's own searched verdict, carried through both paths so this pass and
    # embedding_pass converge whichever order they are run in. Substitution borrows the
    # flagship's measurements, never its facts -- see score_as_flagship.
    embeds_ai = entry.get("embeds_ai")

    def substituted(flagship_metrics: Dict[str, Any], source: str) -> Dict[str, Any]:
        metrics = score_as_flagship(flagship_metrics, note, embeds_ai)
        enforce_flagship_ceiling(metrics, flagship_metrics)
        entry["flagship_source"] = source
        # Provenance rides on the metrics, not just the entry, because the snapshot is
        # what the dashboard reads and a substituted score is unreadable without it.
        metrics["flagship_version"] = flagship_version
        metrics["flagship_source"] = source
        return metrics

    if entry.get("is_generic_category"):
        store_name = resolve_flagship(flagship_version or "", master)
        stored = newest.get(store_name) if store_name else None

        if stored and all(stored.get(metric) is not None for metric in FLAGSHIP_METRICS):
            return substituted(stored, f"store:{store_name}")

        definition = entry.get("flagship_definition") or ""
        if definition:
            return substituted(
                calculate_ai_correlation(
                    flagship_version or name, entry.get("category", ""), definition
                ),
                "model",
            )

        logger.info(
            "%r is a category but no flagship could be used. Leaving it measured on its "
            "own text.", name,
        )

    entry["flagship_source"] = None
    metrics = calculate_ai_correlation(
        name, entry.get("category", ""), entry.get("wikipedia_summary", "") or "", note,
        embeds_ai,
    )
    metrics["flagship_version"] = None
    metrics["flagship_source"] = None
    return metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--apply", action="store_true",
                        help="store the answers and rescore; without it nothing is saved")
    parser.add_argument("--preview", type=int, default=0, metavar="N",
                        help="ask about only the first N eligible skills")
    parser.add_argument("--show", type=int, default=15,
                        help="how many eligible skills to list in the dry run (default 15)")
    args = parser.parse_args(argv)

    master = json_store.load_master()
    timeseries = json_store.load_timeseries()
    eligible = find_eligible(master, timeseries)

    limit = args.preview if args.preview > 0 else len(eligible)
    selected = eligible[:limit]
    requests = -(-len(selected) // FLAGSHIP_BATCH_SIZE)

    print(f"Eligible for a flagship lookup: {len(eligible)}")
    print(f"Selected this run: {len(selected)}  ->  {requests} request(s) "
          f"at {FLAGSHIP_BATCH_SIZE} per batch\n")

    newest = json_store.latest_snapshots(timeseries)
    for name, _ in selected[:args.show]:
        snapshot = newest.get(name) or {}
        print("  embedded %.3f  %s" % (snapshot.get("embedded_ai_sim") or 0.0, name[:56]))
    if len(selected) > args.show:
        print(f"  ... and {len(selected) - args.show} more")

    if not args.apply:
        print(f"\nDry run. No requests were sent and nothing was written. "
              f"Re-run with --apply to spend {requests} request(s).")
        return 0

    if not selected:
        print("Nothing to do.")
        return 0

    checker = get_source_checker(ROLE_FLAGSHIP)
    described = 0
    declined = 0
    moved = 0
    substituted = 0

    for batch in chunks(selected, FLAGSHIP_BATCH_SIZE):
        try:
            answers = checker.describe_flagship_versions_batch([
                {
                    "item_id": name,
                    "skill_name": name,
                    "category": entry.get("category") or "",
                    "current_definition": entry.get("wikipedia_summary") or "",
                }
                for name, entry in batch
            ])
        except DailyQuotaExhausted as err:
            # Every key is spent. Stop and save what has already been answered rather
            # than losing a run's worth of requests.
            logger.warning("Daily quota exhausted: %s. Saving what was answered.", err)
            break
        except AuditUnavailable:
            logger.warning("Gemini unreachable for a batch of %d. Skipping it.", len(batch))
            continue

        for name, entry in batch:
            description = answers.get(name)
            if description is None:
                continue
            if apply_description(entry, description):
                described += 1
            else:
                declined += 1

            previous = newest.get(name) or {}
            metrics = score_entry(name, entry, master, newest)
            json_store.upsert_snapshot(
                timeseries, name, metrics,
                entry.get("onet_codes", []), entry.get("onet_titles", []),
            )

            if entry.get("flagship_source"):
                substituted += 1
                logger.info(
                    "%s scored as %s (%s): ai %.3f -> %.3f, embedded %.3f -> %.3f, %s -> %s",
                    name, entry.get("flagship_version"), entry["flagship_source"],
                    previous.get("ai_score") or 0.0, metrics["ai_score"],
                    previous.get("embedded_ai_sim") or 0.0, metrics["embedded_ai_sim"],
                    previous.get("category_bucket"), metrics["category_bucket"],
                )
            elif (previous.get("embedded_ai_sim") is not None
                  and metrics["embedded_ai_sim"] > previous["embedded_ai_sim"] + 1e-9):
                moved += 1
                logger.info(
                    "%s: embedded %.3f -> %.3f (%s)",
                    name, previous["embedded_ai_sim"], metrics["embedded_ai_sim"],
                    metrics["category_bucket"],
                )

    json_store.save_master(master)
    json_store.save_timeseries(timeseries)

    print(f"\nEvaluated {described + declined}: {described} with AI features on file, "
          f"{declined} declined.")
    print(f"{substituted} category term(s) rescored as their flagship product.")
    print(f"{moved} named product(s) gained embedded-AI signal.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
