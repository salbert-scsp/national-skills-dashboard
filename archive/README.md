# archive/

**Nothing in here runs.** No module in the live system imports anything from this
directory, and `archive/` is never on `sys.path`. It is kept for reference, not for use.

Verified unreferenced at the time of archiving: no live module imports any of these, and
`__pycache__` held a compiled file for every live module and for none of these — runtime
evidence that none had been executed in a long time.

The live system is `main.py` (the web app) and `ingest.py` (the CLI), over the JSON store
in `json_store.py` / `storage.py`.

---

## `sql_era/` — the retired SQL Server pipeline

Superseded when storage moved from Microsoft SQL Server to the two-file JSON store. The
whole cluster is self-contained: `pipeline.py` was the entrypoint, and everything else
was reachable only from it.

| file | was | replaced by |
|---|---|---|
| `database.py` | pyodbc connection, schema creation and repair | `storage.py` + `json_store.py` |
| `pipeline.py` | ingestion orchestrator (the entrypoint) | `definitions_algorithm.py` |
| `score.py` | promotion and scoring, sole writer of `Skills_Historical_Metrics` | `review_actions.py` + `json_store.upsert_snapshot` |
| `dashboard_data.py` | SQL read model for the dashboard | `dashboardtables.py` |
| `Scoring_Algorithm.py` | 4-anchor vector scorer | `sortingalgorithmnew.py` (5 anchors) |
| `migrate_sql_to_json.py` | one-shot SQL → JSON migration | already run; kept in case it is ever needed again |
| `purge_superseded.py` | one-shot purge of superseded queue rows | already run; its output is in `archive/data/` |

**Hazard.** `sql_era/database.py` opens a real database using the `SQL_*` credentials
still present in `.env`. Those six variables are read by nothing else in the project. If
the SQL Server is gone, they are dead config; if it still exists, running anything in
here will connect to it.

**Duplication resolved by archiving.** Three constants had two live definitions before
this move and now have one each: `AI_SKILL_THRESHOLD` / `AI_ENABLING_THRESHOLD` (was also
in `dashboard_data.py`, now only `sortingalgorithmnew.py`), `REPORT_REASONS` (was also in
`score.py`, now only `review_actions.py`), and a third copy of `MAX_SEQUENCE_LENGTH` (was
also in `Scoring_Algorithm.py`).

---

## `unused/` — earlier UI and pipeline attempts

Moved here wholesale. This directory has never had an `__init__.py`, so it was not even
importable as a package.

| file | was |
|---|---|
| `app_interface_streamlit.py` | the Streamlit review UI, direct predecessor of `main.py`. Note it duplicated the review write logic inline instead of calling `review_actions.py`; if it is ever revived, that duplication is the first thing to fix |
| `app_interface.py` | older Streamlit dashboard reading a computed JSON file plus raw pyodbc |
| `dashboardtables.py` | the SQL-direct version of the flattening layer |
| `definitions_algorithm.py` | earlier three-source (Wikipedia / GitHub / PyPI) pipeline writing to SQL |
| `sortingalgorithmnew.py` | **the oldest scorer. Do not revive.** See below |
| `schema.sql` | standalone DDL for the retired SQL schema |
| `dashboard.md` | scratch notes |

### `unused/sortingalgorithmnew.py` must never be revived

Line 7 calls `Tokenizer.from_pretrained()`, which **downloads a tokenizer from Hugging
Face at import**. The live system loads `tokenizer.json` from disk with
`Tokenizer.from_file()` and makes no network call for models at all. That offline
property is deliberate and load-bearing: it is why scoring is reproducible, why it works
without credentials, and why an outage cannot silently change a score.

---

## `data/`

| file | what |
|---|---|
| `pre_refactor_20260807/` | copies of `skills_master.json`, `skills_timeseries.json` and `multisource_skills_cache.json` taken immediately before the storage-boundary refactor. The safety net for that change |
| `superseded_queue_backup_*.json` | one-off dump written by `purge_superseded.py` before it deleted superseded queue rows. Holds the full text of every deleted row |

---

## If you delete this directory

Nothing breaks. It is unreferenced by construction. The only losses are the reference
copies above and the pre-refactor data snapshot — check that the snapshot is no longer
needed first, since `skills_master.json` has no other version history beyond git.
