"""
SQL Server connectivity, schema creation, and in-place schema repair.

Connection handling note: `with pyodbc.connect(...)` is a TRANSACTION context
manager, not a closing one -- it commits or rolls back but leaves the socket open.
Every caller here therefore goes through db_cursor(), which closes explicitly.
"""

import logging
import os
from contextlib import contextmanager
from typing import Iterator

import pyodbc
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

# HITL_Validation_Queue.is_approved values. They live here rather than in score.py
# because database.py is the only local module with no local imports -- score.py already
# imports it, so defining them there and importing back would be a cycle. There is no
# CHECK constraint on the column, so adding a value needs no schema change.
#
# SUPERSEDED marks rows written by an earlier pipeline generation that can never be
# approved (no summary text to score). It keeps them on disk for audit while removing
# them from the pending queue and from the re-audit shadow check.
QUEUE_PENDING = 0
QUEUE_APPROVED = 1
QUEUE_REJECTED = -1
QUEUE_SUPERSEDED = -2

SQL_DRIVER = os.getenv("SQL_DRIVER", "ODBC Driver 18 for SQL Server").strip("{}")
SQL_SERVER = os.getenv("SQL_SERVER")
SQL_PORT = os.getenv("SQL_PORT", "1433")
SQL_DATABASE = os.getenv("SQL_DATABASE")
SQL_USERNAME = os.getenv("SQL_USERNAME")
SQL_PASSWORD = os.getenv("SQL_PASSWORD")


def get_db_connection(autocommit: bool = False) -> pyodbc.Connection:
    """Opens a connection. Callers are responsible for closing it; prefer db_cursor()."""
    missing_vars = [
        name
        for name, value in [
            ("SQL_SERVER", SQL_SERVER),
            ("SQL_DATABASE", SQL_DATABASE),
            ("SQL_USERNAME", SQL_USERNAME),
            ("SQL_PASSWORD", SQL_PASSWORD),
        ]
        if not value
    ]
    if missing_vars:
        raise ValueError(
            f"Missing required .env variables: {', '.join(missing_vars)}"
        )

    conn_str = (
        f"DRIVER={{{SQL_DRIVER}}};"
        f"SERVER={SQL_SERVER},{SQL_PORT};"
        f"DATABASE={SQL_DATABASE};"
        f"UID={SQL_USERNAME};"
        f"PWD={SQL_PASSWORD};"
        "Encrypt=Yes;"
        "TrustServerCertificate=Yes;"
    )
    return pyodbc.connect(conn_str, autocommit=autocommit)


@contextmanager
def db_cursor(autocommit: bool = False) -> Iterator[pyodbc.Cursor]:
    """
    Yields a cursor inside an explicit transaction and always closes the connection.

    With autocommit=False (the default) the whole block commits as one unit, or rolls
    back entirely on exception. Multi-statement writes must use this so a mid-sequence
    failure cannot leave half a skill promoted.
    """
    conn = get_db_connection(autocommit=autocommit)
    cursor = conn.cursor()
    try:
        yield cursor
        if not autocommit:
            conn.commit()
    except Exception:
        if not autocommit:
            try:
                conn.rollback()
            except pyodbc.Error:
                logger.exception("Rollback failed.")
        raise
    finally:
        try:
            cursor.close()
        finally:
            conn.close()


def check_db_health() -> bool:
    try:
        with db_cursor(autocommit=True) as cursor:
            cursor.execute("SELECT 1")
            row = cursor.fetchone()
            if row and row[0] == 1:
                logger.info(
                    "SQL Server connection passed. Connected to [%s] on %s.",
                    SQL_DATABASE, SQL_SERVER,
                )
                return True
    except Exception:
        logger.exception("SQL Server health check failed.")
        return False
    return False


TABLES_SQL = [
    """
    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'Occupations_Master')
    BEGIN
        CREATE TABLE Occupations_Master (
            onet_code NVARCHAR(50) PRIMARY KEY,
            onet_title NVARCHAR(150) NOT NULL,
            last_updated DATETIME DEFAULT GETDATE()
        );
    END;
    """,
    """
    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'Skills_Master')
    BEGIN
        CREATE TABLE Skills_Master (
            skill_id INT IDENTITY(1,1) PRIMARY KEY,
            skill_name NVARCHAR(150) NOT NULL UNIQUE,
            category NVARCHAR(100),
            is_approved BIT DEFAULT 0,
            created_at DATETIME DEFAULT GETDATE()
        );
    END;
    """,
    """
    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'Skill_Occupation_Map')
    BEGIN
        CREATE TABLE Skill_Occupation_Map (
            skill_id INT FOREIGN KEY REFERENCES Skills_Master(skill_id) ON DELETE CASCADE,
            onet_code NVARCHAR(50) FOREIGN KEY REFERENCES Occupations_Master(onet_code) ON DELETE CASCADE,
            first_seen_date DATE NOT NULL,
            last_seen_date DATE NOT NULL,
            -- Hot Tech is a property of the RELATIONSHIP, not the skill: a tool can be
            -- hot for Data Scientists and not hot for Anesthesiologists.
            is_hot_tech BIT NOT NULL DEFAULT 0,
            PRIMARY KEY (skill_id, onet_code)
        );
    END;
    """,
    """
    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'Skills_Historical_Metrics')
    BEGIN
        CREATE TABLE Skills_Historical_Metrics (
            metric_id INT IDENTITY(1,1) PRIMARY KEY,
            skill_id INT FOREIGN KEY REFERENCES Skills_Master(skill_id) ON DELETE CASCADE,
            snapshot_date DATE DEFAULT CAST(GETDATE() AS DATE),
            summary_text NVARCHAR(MAX),
            best_source_name NVARCHAR(150),
            is_credible BIT NULL,
            ai_correlation_score DECIMAL(5, 4),
            ai_sim DECIMAL(5, 4),
            infra_sim DECIMAL(5, 4),
            lang_sim DECIMAL(5, 4),
            is_override BIT DEFAULT 0,
            last_modified DATETIME DEFAULT GETDATE()
        );
    END;
    """,
    """
    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'HITL_Validation_Queue')
    BEGIN
        CREATE TABLE HITL_Validation_Queue (
            QueueID INT IDENTITY(1,1) PRIMARY KEY,
            skill_name NVARCHAR(150) NOT NULL,
            category NVARCHAR(100),
            wiki_title NVARCHAR(200),
            wiki_summary NVARCHAR(MAX),
            wiki_score DECIMAL(5, 4) NULL,
            best_source_name NVARCHAR(150) NULL,
            is_credible BIT NULL,
            gate_reason NVARCHAR(50) NULL,
            onet_code NVARCHAR(50),
            onet_title NVARCHAR(150),
            is_hot_tech BIT DEFAULT 0,
            is_approved INT NOT NULL DEFAULT 0,
            created_at DATETIME DEFAULT GETDATE()
        );
    END;
    """,
]


def _column_type(cursor: pyodbc.Cursor, table: str, column: str) -> str:
    """Returns the SQL type name of a column, or an empty string if absent."""
    cursor.execute(
        """
        SELECT t.name
        FROM sys.columns c
        JOIN sys.types t ON c.user_type_id = t.user_type_id
        WHERE c.object_id = OBJECT_ID(?) AND c.name = ?
        """,
        (table, column),
    )
    row = cursor.fetchone()
    return str(row[0]).lower() if row else ""


def _column_exists(cursor: pyodbc.Cursor, table: str, column: str) -> bool:
    return bool(_column_type(cursor, table, column))


# Columns the application requires that older databases may not have. Every entry is
# additive and nullable, so applying them cannot destroy or rewrite existing data.
REQUIRED_COLUMNS = [
    ("HITL_Validation_Queue", "best_source_name", "NVARCHAR(150) NULL"),
    ("HITL_Validation_Queue", "is_credible", "BIT NULL"),
    ("HITL_Validation_Queue", "gate_reason", "NVARCHAR(50) NULL"),
    # Free text from a dashboard viewer who reported a wrong page or definition. Only
    # ever set on rows whose gate_reason is 'reported_by_viewer'.
    ("HITL_Validation_Queue", "report_note", "NVARCHAR(500) NULL"),
    ("Skills_Historical_Metrics", "best_source_name", "NVARCHAR(150) NULL"),
    ("Skills_Historical_Metrics", "is_credible", "BIT NULL"),
    ("Skill_Occupation_Map", "is_hot_tech", "BIT NULL"),
]

# Time-series tracking columns. Declared NOT NULL for a fresh database, but added as
# nullable to an existing one: backfilling first_seen_date with today's date would
# assert a discovery date we do not actually know.
TRACKING_COLUMNS = [
    ("Skill_Occupation_Map", "first_seen_date", "DATE NULL"),
    ("Skill_Occupation_Map", "last_seen_date", "DATE NULL"),
]


def _supersede_legacy_queue_rows(cursor: pyodbc.Cursor) -> None:
    """
    Retires pending queue rows that no human could ever approve.

    An earlier three-source pipeline queued rows with no summary text at all.
    score.promote_queue_item refuses to promote a row without a summary, and the review
    template's textarea is required, so those rows were permanently stuck pending -- and
    because pipeline._has_pending_review matches on skill_name, each one shadowed its
    skill and kept it from ever being scraped or scored.

    Marking them SUPERSEDED removes them from the pending queue and from the shadow
    check while keeping every row on disk for audit.

    The 388 rows this originally caught were dumped to JSON and deleted on 2026-08-04 by
    purge_superseded.py, so this UPDATE now matches nothing and is silent. It is kept
    because it is the correct handler if another legacy generation ever turns up.

    Reverse with:
        UPDATE HITL_Validation_Queue SET is_approved = 0 WHERE is_approved = -2;

    That is a true inverse -- nothing else about the rows is touched -- but it only
    applies to rows superseded AFTER the purge above. It does re-clutter the reviewer's
    queue, but it can no longer re-shadow ingestion, because _has_pending_review now
    ignores rows with no summary.

    Idempotent with no marker table: the UPDATE is self-extinguishing, since rewritten
    rows stop matching is_approved = 0.

    Must run AFTER the is_approved BIT-to-INT widening (a BIT column cannot hold -2) and
    AFTER the add-column loop (gate_reason must exist or the batch fails to compile).
    """
    cursor.execute(
        f"""
        UPDATE HITL_Validation_Queue
        SET is_approved = {QUEUE_SUPERSEDED}
        WHERE is_approved = {QUEUE_PENDING}
          AND gate_reason IS NULL
          AND (wiki_summary IS NULL OR LTRIM(RTRIM(wiki_summary)) = '')
        """
    )
    superseded = cursor.rowcount
    if superseded and superseded > 0:
        logger.warning(
            "Superseded %d contentless pending queue rows (no gate_reason, no summary). "
            "They predate the current gate, cannot be approved, and were blocking their "
            "skills from being re-audited.",
            superseded,
        )

    # Tripwire, not a migration. A CURRENT-generation row without a summary would be a
    # real bug: the gate always supplies one, and such a row is unapprovable. Expected
    # to be zero forever.
    cursor.execute(
        f"""
        SELECT COUNT(*) FROM HITL_Validation_Queue
        WHERE is_approved = {QUEUE_PENDING}
          AND gate_reason IS NOT NULL
          AND (wiki_summary IS NULL OR LTRIM(RTRIM(wiki_summary)) = '')
        """
    )
    unapprovable = int(cursor.fetchone()[0])
    if unapprovable:
        logger.warning(
            "%d pending queue rows carry a gate_reason but no summary text. The current "
            "pipeline should never produce these and they cannot be approved.",
            unapprovable,
        )


def repair_schema() -> None:
    """
    Brings an existing database up to the current schema.

    CREATE TABLE ... IF NOT EXISTS cannot fix a table that already exists with the
    wrong definition, so these repairs run separately and idempotently. All of them
    are additive: columns are added, never dropped or retyped destructively.

      1. HITL_Validation_Queue.is_approved must be INT, not BIT. The rejection
         sentinel is -1, which BIT cannot store. Widening BIT to INT is lossless.
      2. Missing audit columns (best_source_name, is_credible, gate_reason, report_note)
         are added so a reviewer can tell a bad page pull from a failed credibility audit,
         and can read what a viewer said was wrong when they reported a skill.
      3. is_credible must allow NULL, meaning "Gemini never ran", which is distinct
         from 0 meaning "Gemini judged it not credible".
      4. Skill_Occupation_Map needs first_seen_date and last_seen_date for any
         longitudinal tracking to work at all.
      5. Skill_Occupation_Map needs is_hot_tech, because Hot Tech status belongs to
         the (skill, occupation) relationship rather than to the skill. It is added
         NULLable and NOT backfilled: rows predating the column were all written
         under hot-only ingestion, so their real status is "hot", and stamping 0
         over them would assert the opposite.

         NULL therefore means "predates per-relationship tracking, presumed hot".
         Consequence for every caller: `is_hot_tech = 0` silently omits NULL rows,
         so "not hot" must be written ISNULL(is_hot_tech, 0) = 0, and "hot" as
         COALESCE(is_hot_tech, 1) = 1 or EXISTS(... = 1).

    Legacy columns from earlier iterations (github_*, pypi_*) are left untouched.
    """
    with db_cursor(autocommit=True) as cursor:
        column_type = _column_type(cursor, "HITL_Validation_Queue", "is_approved")
        if column_type == "bit":
            logger.warning(
                "HITL_Validation_Queue.is_approved is BIT and cannot hold the -1 "
                "rejection sentinel. Widening to INT."
            )
            cursor.execute(
                "ALTER TABLE HITL_Validation_Queue ALTER COLUMN is_approved INT NOT NULL"
            )
            logger.info("Widened HITL_Validation_Queue.is_approved to INT.")
        elif column_type and column_type != "int":
            logger.error(
                "HITL_Validation_Queue.is_approved has unexpected type %r; expected INT. "
                "Not altering automatically.",
                column_type,
            )

        for table, column, definition in REQUIRED_COLUMNS + TRACKING_COLUMNS:
            if not _column_exists(cursor, table, column):
                cursor.execute(f"ALTER TABLE {table} ADD {column} {definition}")
                logger.warning("Added missing column %s.%s (%s).", table, column, definition)

        for table in ("HITL_Validation_Queue", "Skills_Historical_Metrics"):
            if _column_type(cursor, table, "is_credible") == "bit":
                cursor.execute(f"ALTER TABLE {table} ALTER COLUMN is_credible BIT NULL")

        # Pre-existing map rows cannot have a truthful discovery date. Flag rather
        # than invent one; the next ingestion run will populate last_seen_date.
        if _column_exists(cursor, "Skill_Occupation_Map", "first_seen_date"):
            cursor.execute(
                "SELECT COUNT(*) FROM Skill_Occupation_Map WHERE first_seen_date IS NULL"
            )
            undated = int(cursor.fetchone()[0])
            if undated:
                logger.warning(
                    "%d Skill_Occupation_Map rows have NULL first_seen_date (predate "
                    "time-series tracking). They are not backfilled.",
                    undated,
                )

        if _column_exists(cursor, "Skill_Occupation_Map", "is_hot_tech"):
            cursor.execute(
                "SELECT COUNT(*) FROM Skill_Occupation_Map WHERE is_hot_tech IS NULL"
            )
            unflagged = int(cursor.fetchone()[0])
            if unflagged:
                logger.warning(
                    "%d Skill_Occupation_Map rows have NULL is_hot_tech (predate "
                    "per-relationship tracking, presumed hot). Not backfilled; this "
                    "count should trend to zero as occupations are re-ingested.",
                    unflagged,
                )

        # Runs here deliberately: after the add-column loop (needs gate_reason) and
        # after the BIT-to-INT widening (a BIT column cannot hold -2).
        _supersede_legacy_queue_rows(cursor)

        # Supports the EXISTS(is_hot_tech = 1) lookup the dashboard leans on. Filtered
        # because only the hot rows are ever selected this way, and the table is about
        # to grow several-fold now that non-hot tools are retained.
        cursor.execute(
            """
            IF NOT EXISTS (
                SELECT 1 FROM sys.indexes
                WHERE name = 'IX_SOM_HotTech' AND object_id = OBJECT_ID('Skill_Occupation_Map')
            )
            CREATE INDEX IX_SOM_HotTech
                ON Skill_Occupation_Map (skill_id) WHERE is_hot_tech = 1;
            """
        )

        cursor.execute(
            """
            IF NOT EXISTS (
                SELECT 1 FROM sys.indexes
                WHERE name = 'IX_HITL_Pending' AND object_id = OBJECT_ID('HITL_Validation_Queue')
            )
            CREATE INDEX IX_HITL_Pending
                ON HITL_Validation_Queue (is_approved, skill_name, onet_code);
            """
        )


def initialize_schema() -> None:
    """Creates any missing tables, then repairs existing ones."""
    try:
        with db_cursor(autocommit=True) as cursor:
            for query in TABLES_SQL:
                cursor.execute(query)
        logger.info("Database schema present.")
    except Exception:
        logger.exception("Failed to create database schema.")
        return

    try:
        repair_schema()
        logger.info("Database schema repairs complete.")
    except Exception:
        logger.exception("Failed to repair database schema.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    if check_db_health():
        initialize_schema()
