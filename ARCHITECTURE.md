# Architecture: Stages and Data Flow

This project is a seven-stage pipeline for AI skills data: ingestion from O*NET, candidate acquisition via search, triage via review and automation, offline scoring and export, all served via a web application with a JSON file store as the only persistence layer.

No file moves are made to enforce these groupings — they're enforced by import dependency, and moving files into subfolders would require rewriting nearly every import statement in the codebase. Instead, each file carries a `STAGE:` header in its docstring and this document lists the stages and their files in dependency order (leaf to root).

## STORAGE — The I/O boundary

**Files:** `storage.py`, `json_store.py`, `backlog.py`, `run_state.py`, `gemini_keys.py`

Nothing here imports anything above it. Every read and write passes through this layer.

- `storage.py` — File I/O primitives; loads `.onnx` models and JSON files by path.
- `json_store.py` — Schema and constants for skills_master.json and skills_timeseries.json.
- `backlog.py` — Resumable ingestion work queue.
- `run_state.py` — Live state of web-triggered runs (lock, progress).
- `gemini_keys.py` — Gemini API key pool with quota rotation and per-minute throttling.

## SCORING — The offline vector engine

**Files:** `sortingalgorithmnew.py`, `cross_encoder.py`, `dashboardtables.py`

Reads from storage, no network. Computes embeddings, anchors, similarity scores, classification rules, and the display-layer banding that maps raw scores to reader-facing bands.

- `sortingalgorithmnew.py` — L2-normalized embeddings, seven anchor vectors, 90/10 blend for poles, classification tree with thresholds, bucketing.
- `cross_encoder.py` — Offline ONNX relevance scoring for candidate pages (0.0–1.0).
- `dashboardtables.py` — Flattens the store into table rows for the UI; also emits the banded `display_ai_score`.

## ACQUISITION — Search and credibility audit

**Files:** `scraping.py`, `agentic_source_check.py`, `embedding_probe.py`

Finds candidate source pages and grades them on relevance (cross-encoder) and credibility (model-based evidence grading).

- `scraping.py` — DuckDuckGo HTML search, candidate resolution, cross-encoder threshold gates.
- `agentic_source_check.py` — Audits whether evidence text supports the skill claim (Gemini in a batch).
- `embedding_probe.py` — Dry-run classifier to answer "is this product genuinely AI-first?".

## INGESTION — Store population

**Files:** `definitions_algorithm.py`, `ingest.py`

Pulls O*NET occupations into the store; invokes the acquisition stage to find definitions for each.

- `definitions_algorithm.py` — Orchestrates the O*NET import (titles, descriptions, occupations) and writes initial snapshots to the store.
- `ingest.py` — Entry point: `python3.11 ingest.py` — calls definitions_algorithm and shows progress.

## REVIEW & TRIAGE — Human in the loop and automated passes

**Files:** `review_actions.py`, `second_pass.py`, `redraft_definitions.py`, `repair_junk_references.py`, `backfill_ai_status.py`

**IMPORTANT:** `review_actions.py` and `second_pass.py` import each other. This cycle is handled by deferring `review_actions`'s import of `second_pass` until function-call time (after both modules finish loading). If you ever refactor these into separate subpackages, preserve this ordering or rewrite one of the imports to use a late `import` statement inside the function body. Otherwise you will get an `ImportError` at startup time.

- `review_actions.py` — Write actions triggered by human reviewers: approve a skill, redraft its definition, file a report.
- `second_pass.py` — Automated triage pass: suggests redrafts for pending skills and re-runs the credibility audit on stale entries.
- `redraft_definitions.py` — Sends model-written definitions back to the model for rewrite when a reviewer rejects them.
- `repair_junk_references.py` — Admin cleanup: removes reference pages misattached to skills.
- `backfill_ai_status.py` — One-time migration: tags skills scored before `ai_status` existed.

## SCORING PASSES — Batch admin operations

**Files:** `embedding_pass.py`, `flagship_pass.py`, `reclassify_snapshots.py`, `audit_ab.py`

Re-measure or re-decide what's on disk.

- `embedding_pass.py` — Re-probes `embeds_ai` for approved skills (90-day window for False, always for unknown).
- `flagship_pass.py` — Scores generic vs. enterprise skill versions against their flagship product.
- `reclassify_snapshots.py` — Re-runs the classification tree on stored metrics without re-embedding (applies threshold changes).
- `audit_ab.py` — A/B test on batching strategy: does sending skills in batches of 6 to Gemini change verdicts?

## WEB APP — FastAPI server

**Files:** `main.py`, `auth.py`

The review queue, dashboard, and admin endpoints.

- `main.py` — FastAPI server: `/`, `/dashboard`, `/review`, `/report`, `/rescore`, `/export` endpoints; serves the UI and coordinates with all stages above.
- `auth.py` — Single shared password for the review UI (`/auth` endpoint).

## Data flow

```
O*NET       ->  INGESTION  ->  STORAGE  ->  SCORING  ->  WEB APP
                (pull)           ↑           (export)      (serve)
                  |              |
                  v              v
            ACQUISITION    REVIEW & TRIAGE  ->  SCORING PASSES (re-measure)
           (find sources)   (human review)          (admin ops)
```

Every write to the store passes through `json_store` or `storage` (they are not separate layers — storage handles the file I/O, json_store handles the schema). Reads for scoring/export go through `dashboardtables` so the display layer is concentrated in one place. Reads for review/admin operations use `json_store` directly.
