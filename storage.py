"""
STAGE: Storage

The single storage boundary. Every byte this project reads or writes passes through
here, and nothing else in the codebase opens a file.

Flip USE_CLOUD to move the whole system to Google Cloud without touching one line of
scoring, scraping, or review logic. Those modules receive dicts and lists and return
dicts and lists; they have no idea whether the bytes came from a laptop or a bucket,
and after this module exists there is no path by which they could find out.

WHY NAMED HELPERS INSTEAD OF ONE GENERIC read_json(path)
--------------------------------------------------------
Because the stores do not share a cloud destination. One generic reader could not
branch correctly:

    skills_master / skills_timeseries  -> Cloud SQL (Postgres). Two writers exist (the
                                          review endpoints and the background scrape),
                                          and last-write-wins on a 2 MB blob loses a
                                          reviewer's approvals. Also read on every
                                          dashboard render.
    scrape cache / backlog / run state / export
                                       -> GCS bucket. Write-rarely, whole-file,
                                          single-writer.
    gemini keys                        -> Secret Manager. Live API keys; a bucket ACL
                                          is a looser thing than secret IAM.
    gemini key state                   -> Firestore or GCS, NOT Secret Manager. It is
                                          mutable per-day state rewritten on every
                                          rotation, and Secret Manager versions are
                                          immutable -- this would burn one version per
                                          key per day.
    model.onnx / cross_encoder_model.onnx / tokenizer.json
                                       -> baked into the image. See model_path().

WHY read_json HAS NO FAILURE POLICY
------------------------------------
The five stores disagree, deliberately, about what a corrupt file means. A damaged
skills_master must stop the process before a save overwrites it with something emptier;
a damaged scrape cache must not stop anything. Baking either answer in here would force
the wrong one on the other, so absence returns the default and anything else raises
StorageUnreadable. Every caller keeps its own except block, log level and message
verbatim -- only the open() moved.
"""

import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

# READ-ONLY MODE. When SKILLS_NO_MODELS=1 is set (in .env or the shell), the scoring
# modules start without model.onnx / cross_encoder_model.onnx. The dashboard and review
# queue then run on the scores already stored in the JSON files; anything that needs to
# embed or cross-encode new text raises ModelsUnavailableError instead. Unset by default,
# so normal behaviour (fail at import when a model is missing) is unchanged.
ALLOW_NO_MODELS = os.getenv("SKILLS_NO_MODELS", "").strip().lower() in ("1", "true", "yes")


class ModelsUnavailableError(RuntimeError):
    """Raised when scoring is attempted in read-only mode (SKILLS_NO_MODELS=1)."""

# ==========================================================================
# THE SWITCH
# ==========================================================================
# False -> local laptop: plain files in the project directory. Current behaviour.
# True  -> Google Cloud: Cloud SQL + GCS + Secret Manager. Not implemented yet; every
#          cloud branch below raises NotImplementedError rather than silently doing
#          nothing, so a premature flip fails loudly instead of writing to a void.
USE_CLOUD = False

# Read but unused while USE_CLOUD is False. Present so the cloud branches read as real
# code rather than pseudocode, and so a deployment can set them before the switch.
GCP_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")
GCS_BUCKET = os.getenv("SKILLS_GCS_BUCKET", "")
CLOUD_SQL_DSN = os.getenv("CLOUD_SQL_DSN", "")


# ==========================================================================
# Every path in the project, in one place
# ==========================================================================
MASTER_FILE = os.getenv("SKILLS_MASTER_FILE", "skills_master.json")
TIMESERIES_FILE = os.getenv("SKILLS_TIMESERIES_FILE", "skills_timeseries.json")

# NOTE the default here is NOT the file actually in use. .env sets
# CACHE_FILE=multisource_skills_cache.json. This default is carried over verbatim from
# scraping.py; "correcting" it to the real filename would silently repoint anyone
# running without a .env at an empty cache and trigger a full re-scrape.
CACHE_FILE = os.getenv("CACHE_FILE", "wikipedia_skills_cache.json")

BACKLOG_FILE = os.getenv("INGESTION_BACKLOG_FILE", "ingestion_backlog.json")
RUN_STATE_FILE = os.getenv("INGESTION_RUN_STATE_FILE", "ingestion_run_state.json")
EXPORT_FILE = os.getenv("DASHBOARD_EXPORT_FILE", "dashboard_export.json")
GEMINI_KEYS_FILE = os.getenv("GEMINI_KEYS_FILE", "gemini_keys.json")
GEMINI_KEY_STATE_FILE = os.getenv("GEMINI_KEY_STATE_FILE", "gemini_key_state.json")
# Dated embedded-AI verdicts. Worth persisting separately from the store because a full
# pass costs about five and a half hours of DuckDuckGo cadence, and losing it would mean
# paying that again. See embedding_pass.py for the recheck policy that reads the dates.
EMBEDDING_CACHE_FILE = os.getenv("EMBEDDING_CACHE_FILE", "embedding_probe_cache.json")

# Directory holding the ONNX graphs and the tokenizer. Kept separate from the data
# paths above because on cloud these do not move to a bucket; they ride in the image.
#
# DEFAULTS TO THIS FILE'S DIRECTORY, not to ".". The models sit beside the code and never
# move, so resolving them against the working directory was wrong in a way that only
# showed up when something started from elsewhere: importing the scorer from any other
# directory failed at import with "Bi-encoder model './model.onnx' not found in the
# project root", which reads like a missing file rather than a missing chdir. Running the
# test suite from outside the project is the case that exposed it.
#
# The data paths above deliberately keep their relative defaults: those are per-deployment
# and are expected to follow the working directory.
MODEL_DIR = os.getenv("MODEL_DIR", os.path.dirname(os.path.abspath(__file__)))
BI_MODEL_NAME = "model.onnx"
CROSS_MODEL_NAME = "cross_encoder_model.onnx"
TOKENIZER_NAME = "tokenizer.json"


class StorageUnreadable(RuntimeError):
    """
    An object exists but could not be read or parsed.

    Deliberately NOT raised for absence: absence is normal for every store here on a
    first run. Callers decide whether this is fatal -- json_store re-raises it as
    StoreCorrupted and refuses to continue, the scrape cache logs it and starts empty.
    """


# ==========================================================================
# Primitives
# ==========================================================================

def read_json(path: str, default: Any = None) -> Any:
    """
    Reads one JSON object. Absence returns `default`; anything else raises.

    Has no failure policy of its own, on purpose -- see the module docstring.
    """
    if not USE_CLOUD:
        # ---- LOCAL LAPTOP -------------------------------------------------
        if not os.path.exists(path):
            return default
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (ValueError, UnicodeDecodeError) as err:
            raise StorageUnreadable(f"{path} exists but is not valid JSON ({err}).") from err
        except OSError as err:
            raise StorageUnreadable(f"{path} could not be read: {err}") from err

    # ---- GOOGLE CLOUD (placeholder, not wired up) -------------------------
    # Generic GCS blob read. Only the cache, backlog, run state and export route
    # through here; master and timeseries have their own helpers below that go to
    # Cloud SQL instead, for the reasons in the module docstring.
    #
    # from google.cloud import storage as gcs
    # from google.api_core import exceptions as gcp_errors
    #
    # blob = gcs.Client(project=GCP_PROJECT).bucket(GCS_BUCKET).blob(os.path.basename(path))
    # try:
    #     return json.loads(blob.download_as_text(encoding="utf-8"))
    # except gcp_errors.NotFound:
    #     return default
    # except (ValueError, gcp_errors.GoogleAPIError) as err:
    #     raise StorageUnreadable(
    #         f"gs://{GCS_BUCKET}/{os.path.basename(path)} unreadable: {err}"
    #     ) from err
    raise NotImplementedError(_CLOUD_NOT_READY)


def write_json(
    path: str,
    payload: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
    sort_keys: bool = False,
    atomic: bool = True,
) -> None:
    """
    Writes one JSON object, atomically by default.

    Atomic means: serialize to a temp file in the SAME directory (os.replace is only
    atomic within a filesystem), flush and fsync so the bytes are really on disk, then
    replace. A crash at any point leaves the previous file intact.

    The formatting keywords exist so each caller reproduces its current bytes exactly.
    They are NOT a style knob and should not be normalized -- flipping sort_keys on the
    569 KB scrape cache would rewrite every line of the file for no gain and make the
    next diff unreadable.
    """
    if not USE_CLOUD:
        # ---- LOCAL LAPTOP -------------------------------------------------
        if not atomic:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=indent, ensure_ascii=ensure_ascii,
                          sort_keys=sort_keys)
            return

        directory = os.path.dirname(os.path.abspath(path)) or "."
        handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory, prefix=".tmp_", suffix=".json",
            delete=False,
        )
        temp_path = handle.name
        try:
            with handle:
                json.dump(payload, handle, indent=indent, ensure_ascii=ensure_ascii,
                          sort_keys=sort_keys)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        except Exception:
            # Never leave the temp file behind on failure; the directory would silently
            # fill with .tmp_*.json across failed runs.
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise
        return

    # ---- GOOGLE CLOUD (placeholder, not wired up) -------------------------
    # A GCS object write is already all-or-nothing, so the temp-file dance has no
    # equivalent and no purpose here -- the `atomic` argument becomes a no-op.
    #
    # from google.cloud import storage as gcs
    # blob = gcs.Client(project=GCP_PROJECT).bucket(GCS_BUCKET).blob(os.path.basename(path))
    # blob.upload_from_string(
    #     json.dumps(payload, indent=indent, ensure_ascii=ensure_ascii, sort_keys=sort_keys),
    #     content_type="application/json",
    # )
    raise NotImplementedError(_CLOUD_NOT_READY)


_CLOUD_NOT_READY = (
    "USE_CLOUD is True but the Google Cloud backend is not implemented yet. "
    "Set USE_CLOUD = False in storage.py."
)


# ==========================================================================
# Per-store helpers
#
# Each returns or accepts plain data. The _raw suffix is the contract: no validation,
# no schema filtering, no upgrade, no logging policy. All of that belongs to the domain
# module that owns the store, and none of those modules ever sees a path.
# ==========================================================================

def load_master_raw() -> Optional[Dict[str, Any]]:
    """The master store as it sits, or None if it has never been written."""
    if not USE_CLOUD:
        return read_json(MASTER_FILE, default=None)

    # ---- GOOGLE CLOUD (placeholder) ---------------------------------------
    # Cloud SQL Postgres, NOT a GCS blob. Two reasons this store differs from the
    # cache and the backlog:
    #   1. Two writers exist -- the FastAPI review endpoints and the background scrape
    #      thread. Last-write-wins on a 2 MB blob loses a reviewer's approvals every
    #      time an ingest finishes mid-review. Rows do not.
    #   2. It is read on every dashboard render, so a blob download per request is a
    #      2 MB fetch per page view.
    #
    # import sqlalchemy
    # engine = sqlalchemy.create_engine(CLOUD_SQL_DSN, pool_pre_ping=True)
    # with engine.connect() as conn:
    #     rows = conn.execute(sqlalchemy.text(SELECT_ALL_SKILLS_SQL)).mappings()
    #     return {row["skill_name"]: _inflate(row) for row in rows}
    raise NotImplementedError(_CLOUD_NOT_READY)


def save_master_raw(master: Dict[str, Any]) -> None:
    """Persists the whole master store. Callers pass a dict; no path crosses the line."""
    if not USE_CLOUD:
        write_json(MASTER_FILE, master, sort_keys=True)
        return

    # ---- GOOGLE CLOUD (placeholder) ---------------------------------------
    # import sqlalchemy
    # engine = sqlalchemy.create_engine(CLOUD_SQL_DSN, pool_pre_ping=True)
    # with engine.begin() as conn:
    #     for name, entry in master.items():
    #         conn.execute(sqlalchemy.text(UPSERT_SKILL_SQL), _flatten(entry))
    #
    # NOTE: `occupations` is a nested list, so it becomes a skill_occupations join
    # table rather than a JSONB column -- the dashboard's by-occupation selector is a
    # GROUP BY, not a JSON scan.
    #
    # NOTE: this signature takes the WHOLE store, so a naive implementation rewrites
    # every row on every save. Under SQL that is the wrong shape; add row-level
    # get_skill(name) / put_skill(record) alongside it and migrate callers before this
    # goes live at scale.
    raise NotImplementedError(_CLOUD_NOT_READY)


def load_timeseries_raw() -> Optional[List[Dict[str, Any]]]:
    """The snapshot list as it sits, or None if it has never been written."""
    if not USE_CLOUD:
        return read_json(TIMESERIES_FILE, default=None)

    # ---- GOOGLE CLOUD (placeholder) ---------------------------------------
    # Cloud SQL, same table family as the master. Append-only in practice, so this is
    # the store that most wants a real query (WHERE quarter >= ...) rather than a
    # full load once it has a few years of history in it.
    raise NotImplementedError(_CLOUD_NOT_READY)


def save_timeseries_raw(records: List[Dict[str, Any]]) -> None:
    """Persists the whole snapshot list."""
    if not USE_CLOUD:
        write_json(TIMESERIES_FILE, records, sort_keys=True)
        return

    # ---- GOOGLE CLOUD (placeholder) ---------------------------------------
    raise NotImplementedError(_CLOUD_NOT_READY)


def load_scrape_cache_raw() -> Optional[Dict[str, Any]]:
    """The scrape cache as it sits. No schema filtering -- that is scraping.py's job."""
    if not USE_CLOUD:
        return read_json(CACHE_FILE, default=None)

    # ---- GOOGLE CLOUD (placeholder) ---------------------------------------
    # GCS blob. Whole-file and single-writer, so no database is warranted. Worth a
    # local /tmp copy per instance if reads become hot.
    raise NotImplementedError(_CLOUD_NOT_READY)


def save_scrape_cache_raw(cache_data: Dict[str, Any]) -> None:
    """Persists the scrape cache. indent=4 preserved from the original writer."""
    if not USE_CLOUD:
        write_json(CACHE_FILE, cache_data, indent=4)
        return

    # ---- GOOGLE CLOUD (placeholder) ---------------------------------------
    raise NotImplementedError(_CLOUD_NOT_READY)


def load_backlog_raw() -> Optional[Dict[str, Any]]:
    """The ingestion backlog as it sits."""
    if not USE_CLOUD:
        return read_json(BACKLOG_FILE, default=None)

    # ---- GOOGLE CLOUD (placeholder) ---------------------------------------
    raise NotImplementedError(_CLOUD_NOT_READY)


def save_backlog_raw(backlog: Dict[str, Any]) -> None:
    """Persists the ingestion backlog."""
    if not USE_CLOUD:
        write_json(BACKLOG_FILE, backlog, sort_keys=True)
        return

    # ---- GOOGLE CLOUD (placeholder) ---------------------------------------
    raise NotImplementedError(_CLOUD_NOT_READY)


def load_run_state_raw() -> Optional[Dict[str, Any]]:
    """The live scrape-run record as it sits."""
    if not USE_CLOUD:
        return read_json(RUN_STATE_FILE, default=None)

    # ---- GOOGLE CLOUD (placeholder) ---------------------------------------
    # On a scale-to-zero platform this is the file that decides whether a run survives
    # an instance swap, so it wants the same durability as the backlog it reports on.
    raise NotImplementedError(_CLOUD_NOT_READY)


def save_run_state_raw(state: Dict[str, Any]) -> None:
    """Persists the live scrape-run record."""
    if not USE_CLOUD:
        write_json(RUN_STATE_FILE, state)
        return

    # ---- GOOGLE CLOUD (placeholder) ---------------------------------------
    raise NotImplementedError(_CLOUD_NOT_READY)


def save_dashboard_export(path: str, payload: Dict[str, Any]) -> None:
    """
    Writes the flattened dashboard export.

    Atomic, unlike the raw open() this replaced: a crash mid-write used to leave a
    truncated dashboard_export.json behind.
    """
    if not USE_CLOUD:
        write_json(path, payload)
        return

    # ---- GOOGLE CLOUD (placeholder) ---------------------------------------
    # A public-read GCS object is the natural home if anything outside this app ever
    # consumes it. Nothing does today.
    raise NotImplementedError(_CLOUD_NOT_READY)


def load_gemini_keys_raw() -> Any:
    """
    The spare-key file as it sits: a list, a {"keys": [...]} wrapper, or None.

    Shape coercion stays in gemini_keys.py, which owns what a key looks like.
    """
    if not USE_CLOUD:
        return read_json(GEMINI_KEYS_FILE, default=None)

    # ---- GOOGLE CLOUD (placeholder) ---------------------------------------
    # Secret Manager. These are 5 live API keys; they must never become a GCS object,
    # because bucket ACLs are a different and looser thing than secret IAM.
    #
    # from google.cloud import secretmanager
    # client = secretmanager.SecretManagerServiceClient()
    # name = f"projects/{GCP_PROJECT}/secrets/gemini-api-keys/versions/latest"
    # return json.loads(client.access_secret_version(name=name).payload.data.decode("utf-8"))
    raise NotImplementedError(_CLOUD_NOT_READY)


def load_embedding_cache() -> Dict[str, Any]:
    """
    Dated embedded-AI verdicts, keyed by skill name.

    An unreadable cache is NOT fatal and is not raised. The worst outcome of losing it is
    re-probing, which costs time and no correctness; refusing to run would cost the whole
    pass. The recheck policy lives in embedding_pass.py, which owns what a stale verdict
    means.
    """
    if not USE_CLOUD:
        try:
            return read_json(EMBEDDING_CACHE_FILE, default={}) or {}
        except StorageUnreadable:
            logger.exception(
                "%s could not be read; every skill will be re-probed.",
                EMBEDDING_CACHE_FILE,
            )
            return {}

    # ---- GOOGLE CLOUD (placeholder) ---------------------------------------
    # GCS object, alongside the skills store. Not Firestore: this is written once per
    # pass rather than per decision, and it is read whole.
    raise NotImplementedError(_CLOUD_NOT_READY)


def save_embedding_cache(cache: Dict[str, Any]) -> None:
    """Persists the dated verdicts."""
    if not USE_CLOUD:
        write_json(EMBEDDING_CACHE_FILE, cache)
        return

    # ---- GOOGLE CLOUD (placeholder) ---------------------------------------
    raise NotImplementedError(_CLOUD_NOT_READY)


def load_gemini_key_state_raw() -> Optional[Dict[str, Any]]:
    """Which keys are spent today. The date check stays in gemini_keys.py."""
    if not USE_CLOUD:
        return read_json(GEMINI_KEY_STATE_FILE, default=None)

    # ---- GOOGLE CLOUD (placeholder) ---------------------------------------
    # See save_gemini_key_state_raw for why this is not Secret Manager.
    raise NotImplementedError(_CLOUD_NOT_READY)


def save_gemini_key_state_raw(state: Dict[str, Any]) -> None:
    """Persists which keys are spent today."""
    if not USE_CLOUD:
        write_json(GEMINI_KEY_STATE_FILE, state, ensure_ascii=True)
        return

    # ---- GOOGLE CLOUD (placeholder) ---------------------------------------
    # Firestore, NOT Secret Manager. This is mutable per-day state rewritten every time
    # a key is retired, and Secret Manager versions are immutable -- this would create
    # one version per key per day and eventually hit the version cap.
    #
    # from google.cloud import firestore
    # firestore.Client(project=GCP_PROJECT).collection("gemini").document("key_state").set(state)
    #
    # Worth noting for the migration: on a scale-to-zero platform this file lives on an
    # ephemeral filesystem, so without this branch rotation state is lost on every cold
    # start and the first request of each instance re-probes an already-spent key.
    raise NotImplementedError(_CLOUD_NOT_READY)


# ==========================================================================
# Models
# ==========================================================================

def model_path(filename: str) -> str:
    """
    Resolves a model or tokenizer file to something onnxruntime can open.

    Returns a FILESYSTEM PATH in both modes, deliberately. onnxruntime memory-maps the
    graph off disk; handing it bytes instead would hold model.onnx (86 MB) and
    cross_encoder_model.onnx (87 MB) in the heap on top of the runtime's own copy and
    roughly double peak memory for nothing.
    """
    if not USE_CLOUD:
        # ---- LOCAL LAPTOP -------------------------------------------------
        # One join. Startup cost is unmeasurable, which is the requirement: nothing
        # about local model loading may get slower.
        return os.path.join(MODEL_DIR, filename)

    # ---- GOOGLE CLOUD (placeholder) ---------------------------------------
    # PREFERRED: bake all three files into the image at /app/models and set
    # MODEL_DIR=/app/models. Then this branch never runs, the local branch serves both
    # modes, and cold start pays no network cost. 173 MB is well inside an image.
    #
    # FALLBACK, only if image size becomes a problem -- download once per container to
    # /tmp at startup, never per request:
    #
    # from google.cloud import storage as gcs
    # local = os.path.join(tempfile.gettempdir(), filename)
    # if not os.path.exists(local):
    #     gcs.Client(project=GCP_PROJECT).bucket(GCS_BUCKET).blob(
    #         f"models/{filename}"
    #     ).download_to_filename(local)
    # return local
    #
    # OFFLINE GUARANTEE, both branches: tokenizer.json is loaded with
    # Tokenizer.from_file() and never Tokenizer.from_pretrained(). There is no Hugging
    # Face call anywhere in the live system and this function must not introduce one.
    raise NotImplementedError(_CLOUD_NOT_READY)
