import json
import os
import pyodbc
from dotenv import load_dotenv

load_dotenv()

OUTPUT_FILE = os.environ.get("DASHBOARD_OUTPUT_FILE", "AI_skills_dashboard_mini.json")


def get_db_connection():
    """Establishes a secure connection to AISkillsDB using environment variables."""
    driver = os.environ.get("SQL_DRIVER", "ODBC Driver 18 for SQL Server").strip("{}")
    server = os.environ.get("SQL_SERVER", "localhost")
    port = os.environ.get("SQL_PORT", "1433")
    database = os.environ.get("SQL_DATABASE", "AISkillsDB")
    username = os.environ.get("SQL_USERNAME", "sa")
    password = os.environ.get("SQL_PASSWORD")

    if not password:
        raise ValueError("SQL_PASSWORD environment variable is not set in .env file.")

    conn_str = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server},{port};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        "TrustServerCertificate=Yes;"
    )
    return pyodbc.connect(conn_str, autocommit=False)


# ============================================================================
# 1. HITL QUEUE REVIEW & AUDIT FUNCTIONS
# ============================================================================

def fetch_hitl_queue(only_flagged: bool = True) -> list[dict]:
    """
    Fetches records from HITL_Validation_Queue.
    If `only_flagged` is True, returns only items needing human review (is_approved = 0 or any source score < 0.90).
    """
    sql = (
        "SELECT QueueID, skill_name, category, "
        "wiki_title, wiki_summary, wiki_score, "
        "github_title, github_summary, github_score, "
        "pypi_title, pypi_summary, pypi_score, "
        "onet_code, onet_title, is_hot_tech, is_approved, created_at "
        "FROM HITL_Validation_Queue "
    )
    if only_flagged:
        sql += (
            "WHERE is_approved = 0 "
            "OR wiki_score < 0.90 OR github_score < 0.90 OR pypi_score < 0.90 "
        )
    sql += "ORDER BY created_at DESC"

    records = []
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()

        for r in rows:
            records.append({
                "queue_id": r[0],
                "skill_name": r[1],
                "category": r[2],
                "sources": {
                    "wiki": {"title": r[3], "summary": r[4], "score": float(r[5]) if r[5] is not None else None},
                    "github": {"title": r[6], "summary": r[7], "score": float(r[8]) if r[8] is not None else None},
                    "pypi": {"title": r[9], "summary": r[10], "score": float(r[11]) if r[11] is not None else None},
                },
                "onet_code": r[12],
                "onet_title": r[13],
                "is_hot_tech": bool(r[14]),
                "is_approved": bool(r[15]),
                "created_at": str(r[16]) if r[16] else None
            })
    return records


def approve_hitl_queue_item(queue_id: int):
    """Marks all source payloads for a skill in the HITL queue as approved (Checkmark Action)."""
    sql = "UPDATE HITL_Validation_Queue SET is_approved = 1 WHERE QueueID = ?"
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, queue_id)
        conn.commit()
    print(f"QueueID {queue_id} successfully approved.")


def update_or_delete_hitl_source(queue_id: int, source_type: str, new_summary: str | None = None, new_title: str | None = None):
    """
    Updates or deletes a specific source payload (wiki, github, pypi) for a skill.
    - If `new_summary` is provided, updates title/summary.
    - If `new_summary` is None, clears/deletes that source payload (Delete Source Action).
    """
    if source_type not in ["wiki", "github", "pypi"]:
        raise ValueError("source_type must be 'wiki', 'github', or 'pypi'.")

    title_col = f"{source_type}_title"
    summary_col = f"{source_type}_summary"
    score_col = f"{source_type}_score"

    if new_summary is None:
        # Delete / Discard this specific source
        sql = f"UPDATE HITL_Validation_Queue SET {title_col} = NULL, {summary_col} = NULL, {score_col} = NULL WHERE QueueID = ?"
        params = (queue_id,)
    else:
        # Edit / Update source summary inline
        sql = f"UPDATE HITL_Validation_Queue SET {title_col} = ?, {summary_col} = ? WHERE QueueID = ?"
        params = (new_title, new_summary, queue_id)

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, *params)
        conn.commit()
    print(f"Updated {source_type} source for QueueID {queue_id}.")


# ============================================================================
# 2. DASHBOARD EXPORT ENGINE
# ============================================================================

def process_and_flatten_dashboard_payload():
    """Pulls approved skills and formats JSON assets for dashboard rendering."""
    records = {}
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        query = (
            "SELECT QueueID, skill_name, category, wiki_title, wiki_summary, "
            "github_title, github_summary, pypi_title, pypi_summary, "
            "onet_code, onet_title, is_hot_tech, created_at "
            "FROM HITL_Validation_Queue "
            "WHERE is_approved = 1 "
            "ORDER BY skill_name, created_at DESC"
        )
        cursor.execute(query)
        rows = cursor.fetchall()

    for row in rows:
        (q_id, skill_name, category, w_title, w_summary, g_title, g_summary,
         p_title, p_summary, onet_code, onet_title, is_hot, created_at) = row

        key = skill_name
        if key not in records:
            records[key] = {
                "skill_name": skill_name,
                "category": category,
                "created_at": str(created_at) if created_at else None,
                "is_hot_tech": bool(is_hot),
                "summaries": {
                    "wikipedia": {"title": w_title, "summary": w_summary} if w_summary else None,
                    "github": {"title": g_title, "summary": g_summary} if g_summary else None,
                    "pypi": {"title": p_title, "summary": p_summary} if p_summary else None,
                },
                "onet_codes": [],
                "onet_titles": [],
                "onet_occupations": []
            }

        if onet_code and onet_title:
            label = f"{onet_code.strip()} - {onet_title.strip()}"
            if onet_code not in records[key]["onet_codes"]:
                records[key]["onet_codes"].append(onet_code)
            if onet_title not in records[key]["onet_titles"]:
                records[key]["onet_titles"].append(onet_title)
            if label not in records[key]["onet_occupations"]:
                records[key]["onet_occupations"].append(label)

    flattened_dashboard_rows = list(records.values())

    print(f"Writing optimized dashboard asset to {OUTPUT_FILE} ({len(flattened_dashboard_rows)} items)...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        json.dump(flattened_dashboard_rows, out_f, indent=4, ensure_ascii=False)
    print("Ready for dashboard rendering.")


if __name__ == "__main__":
    process_and_flatten_dashboard_payload()