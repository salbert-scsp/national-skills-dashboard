"""
DEAD -- do not run. Kept for reference only; nothing imports this.

An early Streamlit dashboard that read a precomputed JSON file (COMPUTED_OUTPUT_FILE)
and queried Microsoft SQL Server directly through pyodbc, with its own copy of the
connection logic.

Superseded twice over: the live UI is main.py (FastAPI + Jinja2, templates/ and
static/), and storage is the JSON store behind storage.py, not SQL. Running this would
need the retired database and the pandas/Streamlit stack.
"""

import os
import json
import pandas as pd
import pyodbc
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Multi-Source Skill Validation", layout="wide")

DASHBOARD_JSON = os.environ.get("COMPUTED_OUTPUT_FILE", "multisource_computed_dashboard.json")

def get_db_connection():
    server = os.environ.get("SQL_SERVER", "127.0.0.1")
    db = os.environ.get("SQL_DATABASE", "AISkillsDB")
    user = os.environ.get("SQL_USERNAME", "sa")
    pwd = os.environ.get("SQL_PASSWORD", "")
    driver = os.environ.get("SQL_DRIVER", "ODBC Driver 18 for SQL Server")
    conn_str = f"DRIVER={{{driver}}};SERVER={server};DATABASE={db};UID={user};PWD={pwd};TrustServerCertificate=yes;"
    return pyodbc.connect(conn_str, autocommit=True)

@st.cache_data(ttl=2)
def load_data_from_db():
    try:
        conn = get_db_connection()
        query = """
        SELECT 
            QueueID, skill_name, 
            wiki_title, wiki_summary, wiki_score,
            github_title, github_summary, github_score,
            pypi_title, pypi_summary, pypi_score,
            onet_code, onet_title, is_approved
        FROM HITL_Validation_Queue
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception:
        if os.path.exists(DASHBOARD_JSON):
            with open(DASHBOARD_JSON, "r") as f:
                return pd.DataFrame(json.load(f))
        return pd.DataFrame()

def save_grid_edits(edited_df, original_df):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Track status changes (Approve / Reject / Return to Pending)
    status_changed = edited_df[edited_df["approved_check"] != original_df["approved_check"]]
    for _, row in status_changed.iterrows():
        new_status = 1 if row["approved_check"] else 0
        cursor.execute("UPDATE HITL_Validation_Queue SET is_approved = ? WHERE QueueID = ?", (new_status, int(row["QueueID"])))
        
    # Track text edits in grid
    for _, row in edited_df.iterrows():
        cursor.execute("""
            UPDATE HITL_Validation_Queue 
            SET wiki_title = ?, wiki_summary = ?, 
                github_title = ?, github_summary = ?, 
                pypi_title = ?, pypi_summary = ?
            WHERE QueueID = ?
        """, (
            str(row["wiki_title"]), str(row["wiki_summary"]),
            str(row["github_title"]), str(row["github_summary"]),
            str(row["pypi_title"]), str(row["pypi_summary"]),
            int(row["QueueID"])
        ))
        
    conn.close()

def auto_approve_high_confidence(queue_ids):
    if not queue_ids:
        return
    conn = get_db_connection()
    cursor = conn.cursor()
    placeholders = ",".join(["?"] * len(queue_ids))
    cursor.execute(f"UPDATE HITL_Validation_Queue SET is_approved = 1 WHERE QueueID IN ({placeholders})", list(queue_ids))
    conn.close()

st.title("Skill Candidate Validation Dashboard")

df = load_data_from_db()

if df.empty:
    st.warning("No queue data found in database. Please run main.py and dashboardtables.py first.")
    st.stop()

# Formatting defaults
required_cols = [
    "QueueID", "skill_name", "wiki_score", "wiki_title", "wiki_summary",
    "github_score", "github_title", "github_summary",
    "pypi_score", "pypi_title", "pypi_summary",
    "onet_title", "is_approved"
]

for col in required_cols:
    if col not in df.columns:
        df[col] = 0 if ("score" in col or col in ["is_approved", "QueueID"]) else ""

for col in ["wiki_score", "github_score", "pypi_score"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

for col in ["wiki_title", "wiki_summary", "github_title", "github_summary", "pypi_title", "pypi_summary"]:
    df[col] = df[col].fillna("")

st.sidebar.header("Batch Operations")

tab_selection = st.sidebar.radio(
    "View Queue", 
    ["Pending Queue (is_approved = 0)", "Approved Skills (is_approved = 1)", "Rejected Skills (is_approved = -1)"]
)

if "Pending" in tab_selection:
    filtered_df = df[df["is_approved"] == 0].copy()
elif "Approved" in tab_selection:
    filtered_df = df[df["is_approved"] == 1].copy()
else:
    filtered_df = df[df["is_approved"] == -1].copy()

filtered_df["approved_check"] = filtered_df["is_approved"].apply(lambda x: True if x == 1 else False)

st.subheader(f"List View ({len(filtered_df)} items)")

if "Pending" in tab_selection and not filtered_df.empty:
    col_auto, col_space = st.columns([1, 3])
    with col_auto:
        if st.button("Auto-Approve All Skills with Match Score > 0.90"):
            high_conf = filtered_df[
                (filtered_df["wiki_score"] >= 0.90) | 
                (filtered_df["github_score"] >= 0.90) | 
                (filtered_df["pypi_score"] >= 0.90)
            ]
            ids = high_conf["QueueID"].tolist()
            if ids:
                auto_approve_high_confidence(ids)
                st.success(f"Approved {len(ids)} high-confidence skills.")
                st.cache_data.clear()
                st.rerun()
            else:
                st.info("No pending skills with score >= 0.90 found.")

display_cols = [
    "approved_check", "QueueID", "skill_name", 
    "wiki_score", "wiki_title", "wiki_summary", 
    "github_score", "github_title", "github_summary", 
    "pypi_score", "pypi_title", "pypi_summary", 
    "onet_title"
]

edited_df = st.data_editor(
    filtered_df[display_cols],
    width="stretch",
    column_config={
        "approved_check": st.column_config.CheckboxColumn("Approved", help="Check to approve, uncheck to send to pending"),
        "QueueID": st.column_config.NumberColumn("ID", disabled=True),
        "skill_name": st.column_config.TextColumn("Skill Name", disabled=True),
        "wiki_score": st.column_config.NumberColumn("Wiki Score", format="%.2f", disabled=True),
        "wiki_title": st.column_config.TextColumn("Wiki Title", disabled=False),
        "wiki_summary": st.column_config.TextColumn("Wiki Summary", disabled=False),
        "github_score": st.column_config.NumberColumn("GitHub Score", format="%.2f", disabled=True),
        "github_title": st.column_config.TextColumn("GitHub Title", disabled=False),
        "github_summary": st.column_config.TextColumn("GitHub Summary", disabled=False),
        "pypi_score": st.column_config.NumberColumn("PyPI Score", format="%.2f", disabled=True),
        "pypi_title": st.column_config.TextColumn("PyPI Title", disabled=False),
        "pypi_summary": st.column_config.TextColumn("PyPI Summary", disabled=False),
        "onet_title": st.column_config.TextColumn("O*NET Title", disabled=True),
    },
    hide_index=True,
    key="data_editor_grid"
)

if st.button("Submit Table Changes"):
    save_grid_edits(edited_df, filtered_df)
    st.success("Table changes and approvals saved successfully.")
    st.cache_data.clear()
    st.rerun()