"""
Streamlit frontend: dashboard, skill detail, HITL review, and trends.

    streamlit run app_interface.py

Reads skills_master.json and skills_timeseries.json through dashboardtables, which
does all the joining. This module is presentation and review actions only; it holds
no SQL, no scraping, and no scoring logic of its own beyond calling the engine when a
reviewer approves an edited definition.

Two display rules, both deliberate:

  1. No raw source URLs in the summary table. The reference page belongs on the review
     screen, where someone is judging whether it is the right page, and in the detail
     panel as a single link -- not as a column people scroll past.
  2. Category buckets are always the STORED value, computed from the raw cosine score
     at scoring time. Nothing here re-derives a bucket, so a skill's class cannot
     change as the table is filtered or sorted.

infra_sim and lang_sim are absent because they no longer exist: the enabling signal is
now the three-way split of tech base, ML pipeline, and embedded AI.
"""

import logging

import pandas as pd
import plotly.express as px
import streamlit as st

import json_store
from dashboardtables import PUBLIC_COLUMNS, load_dashboard
from json_store import STATUS_APPROVED, STATUS_PENDING, STATUS_REJECTED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI Skills Analytics", layout="wide")

# Bucket colours: blue is an AI Skill, red is AI Enabling, grey is not an AI skill.
BUCKET_COLOURS = {
    "AI Skill": "#1f5fbf",
    "AI Enabling Skill": "#c1274a",
    "Not AI Skill": "#8b949e",
}

# Reviewer-facing explanations for why the pipeline held a skill back.
GATE_REASON_HELP = {
    "below_relevance_threshold": (
        "The cross-encoder is not confident this page is about this skill, so no "
        "credibility audit was run. Confirm the text describes the right subject."
    ),
    "failed_credibility_audit": (
        "The page match looked correct, but the source audit judged it not "
        "authoritative. Verify before approving."
    ),
    "audit_unavailable": (
        "The page match was strong, but the credibility audit could not run (API "
        "quota or outage), so this text was never vetted."
    ),
    "no_candidate_found": (
        "No reference page could be resolved. The text below is O*NET taxonomy "
        "boilerplate, not a real definition. Replace it or reject."
    ),
    "discovered_not_audited": (
        "Recorded from the O*NET occupation map but never audited, normally because "
        "it is not flagged Hot Technology."
    ),
}


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

@st.cache_data(show_spinner="Reading the local JSON store...")
def get_dashboard():
    """
    Loads and flattens both store files.

    Cached because every widget interaction re-runs this script top to bottom, and
    re-reading and re-joining the files on every keystroke makes the UI crawl. Any
    write action clears the cache explicitly.
    """
    return load_dashboard()


def refresh():
    """Drops the cache and re-runs, so a review action is visible immediately."""
    get_dashboard.clear()
    st.rerun()


try:
    DATA = get_dashboard()
except json_store.StoreCorrupted as err:
    # Surfaced rather than swallowed: an empty dashboard would look like a pipeline
    # that found nothing, which is a very different problem from a damaged file.
    st.error(f"The local JSON store could not be read.\n\n{err}")
    st.stop()

SKILLS = DATA["skills"]
SUMMARY = DATA["summary"]

if not SKILLS and not DATA["pending"]:
    st.title("AI Skills Analytics")
    st.warning(
        "No skills in the local store yet. Run an ingestion first:\n\n"
        "`python3.11 main.py 15-`"
    )
    st.stop()


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.title("AI Skills Analytics")

kpi = st.columns(5)
kpi[0].metric("Skills scored", SUMMARY["skills_scored"])
kpi[1].metric("Tools discovered", SUMMARY["tools_discovered"])
kpi[2].metric("Occupations covered", SUMMARY["occupations_covered"])
kpi[3].metric("At or above 0.30 raw", SUMMARY["at_ai_threshold"])
kpi[4].metric("Awaiting review", SUMMARY["pending_review"])

overview_tab, detail_tab, review_tab, trend_tab = st.tabs(
    ["Overview", "Skill detail", f"Review ({SUMMARY['pending_review']})", "Trends"]
)


# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------

with overview_tab:
    st.subheader("Composition by AI category")
    st.caption(
        "Counted on raw cosine scores, never normalized ones: at or above 0.30 is an "
        "AI Skill, 0.15 to 0.30 is AI Enabling, below that is not an AI skill. "
        "Bucketing after a rescale would make a skill's category shift as you filter."
    )

    buckets = pd.DataFrame(DATA["buckets"])
    if not buckets.empty and buckets["count"].sum():
        composition = px.bar(
            buckets, x="count", y=[""] * len(buckets), color="category",
            orientation="h", height=140,
            color_discrete_map=BUCKET_COLOURS,
            category_orders={"category": list(BUCKET_COLOURS)},
        )
        composition.update_layout(
            barmode="stack", showlegend=True, margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title=None, yaxis_title=None, legend_title=None,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        composition.update_yaxes(showticklabels=False)
        st.plotly_chart(composition, use_container_width=True)

    st.subheader("Ranked skills")

    controls = st.columns([2, 1, 2])
    search = controls[0].text_input("Filter by name", key="overview_search")
    direction = controls[1].selectbox(
        "Sort", ["AI relevance, high to low", "AI relevance, low to high"],
        key="overview_sort",
    )
    chosen_buckets = controls[2].multiselect(
        "AI category", list(BUCKET_COLOURS), default=list(BUCKET_COLOURS),
        key="overview_buckets",
    )

    frame = pd.DataFrame(SKILLS)
    if not frame.empty:
        if search.strip():
            frame = frame[frame["skill_name"].str.contains(search.strip(), case=False, na=False)]
        if chosen_buckets:
            frame = frame[frame["category_bucket"].isin(chosen_buckets)]
        frame = frame.sort_values(
            "ai_score", ascending=direction.endswith("low to high"), kind="stable"
        )

    if frame.empty:
        st.info("No skills match the current filters.")
    else:
        st.caption(f"{len(frame)} of {len(SKILLS)} scored skills.")
        # PUBLIC_COLUMNS deliberately omits resolved_title and wikipedia_summary. The
        # reference page is review-screen material, and a full definition in a table
        # cell is unreadable anyway.
        st.dataframe(
            frame[list(PUBLIC_COLUMNS)],
            hide_index=True,
            use_container_width=True,
            column_config={
                "skill_name": st.column_config.TextColumn("Skill"),
                "category": st.column_config.TextColumn("O*NET category"),
                "ai_score": st.column_config.NumberColumn("AI score", format="%.4f"),
                "category_bucket": st.column_config.TextColumn("Class"),
                "tech_base_sim": st.column_config.NumberColumn("Tech base", format="%.4f"),
                "ml_pipeline_sim": st.column_config.NumberColumn("ML pipeline", format="%.4f"),
                "embedded_ai_sim": st.column_config.NumberColumn("Embedded AI", format="%.4f"),
                "occupation_count": st.column_config.NumberColumn("Occupations"),
            },
        )


# --------------------------------------------------------------------------
# Skill detail
# --------------------------------------------------------------------------

with detail_tab:
    if not SKILLS:
        st.info("Nothing scored yet.")
    else:
        names = [row["skill_name"] for row in SKILLS]
        selected = st.selectbox("Skill", names, key="detail_select")
        record = next(row for row in SKILLS if row["skill_name"] == selected)

        st.subheader(record["skill_name"])
        st.caption(record["category"] or "Uncategorized")

        headline = st.columns(2)
        headline[0].metric("AI score (raw cosine)", f"{record['ai_score']:.4f}")
        headline[1].metric("Class", record["category_bucket"])

        st.markdown("**Enabling breakdown**")
        st.caption(
            "How the skill is adjacent to AI, measured against three separate anchors. "
            "These are diagnostics reported alongside the AI score, not inputs to it."
        )
        enabling = st.columns(3)
        enabling[0].metric("Tech base", f"{record['tech_base_sim']:.4f}")
        enabling[1].metric("ML pipeline", f"{record['ml_pipeline_sim']:.4f}")
        enabling[2].metric("Embedded AI", f"{record['embedded_ai_sim']:.4f}")

        st.markdown("**Definition**")
        st.write(record["wikipedia_summary"] or "No definition recorded.")

        if record["onet_titles"]:
            st.markdown("**Occupations**")
            for code, title in zip(record["onet_codes"], record["onet_titles"]):
                st.write(f"`{code}`  {title}")

        st.caption(
            f"Snapshot {record['snapshot_date']} ({record['quarter']})"
            + (f" · reference: {record['resolved_title']}" if record["resolved_title"] else "")
        )


# --------------------------------------------------------------------------
# Review (HITL)
# --------------------------------------------------------------------------

def apply_review(skill_name: str, summary: str, reference: str, approve: bool) -> None:
    """
    Writes a review decision straight to the JSON store.

    Approving rescores from the text in the box, so an edited definition is the one
    that gets measured, and writes the snapshot through the same code path ingestion
    uses -- a human approval and an automatic one produce identical records.

    Both files are re-read here rather than reusing the cached copy: the cache is a
    render-time snapshot, and writing back a stale one would silently discard anything
    changed since the page loaded.
    """
    from definitions_algorithm import record_snapshot

    master = json_store.load_master()
    entry = master.get(skill_name)
    if entry is None:
        st.error(f"{skill_name} is no longer in the store.")
        return

    entry["wikipedia_summary"] = summary.strip()
    entry["resolved_title"] = reference.strip() or entry.get("resolved_title")
    entry["last_updated"] = json_store.today().isoformat()
    entry["status"] = STATUS_APPROVED if approve else STATUS_REJECTED

    if approve:
        timeseries = json_store.load_timeseries()
        metrics = record_snapshot(entry, timeseries)
        json_store.save_timeseries(timeseries)
        json_store.save_master(master)
        st.success(
            f"Approved {skill_name}: {metrics['ai_score']:+.4f} ({metrics['category_bucket']})."
        )
    else:
        json_store.save_master(master)
        st.success(f"Rejected {skill_name}. It will not appear on the dashboard.")


with review_tab:
    st.subheader("Human review")
    st.caption(
        "Skills the pipeline would not approve on its own. Editing the definition and "
        "approving rescores from the edited text. Nothing here is on the dashboard yet."
    )

    mode = st.radio(
        "Show", ["Awaiting review", "Edit an approved skill"],
        horizontal=True, key="review_mode",
    )

    if mode == "Awaiting review":
        queue = DATA["pending"]
        if not queue:
            st.success("Nothing awaiting review.")
        else:
            for item in queue:
                with st.expander(f"{item['skill_name']}  ·  {item['gate_reason'] or 'flagged'}"):
                    note = GATE_REASON_HELP.get(item["gate_reason"])
                    if note:
                        st.warning(note)

                    facts = st.columns(3)
                    facts[0].write(f"**Category**  \n{item['category'] or 'n/a'}")
                    facts[1].write(
                        "**Cross-encoder**  \n"
                        + (f"{item['cross_score']:.4f}" if item["cross_score"] is not None else "not scored")
                    )
                    facts[2].write(
                        "**Credibility**  \n"
                        + ("not run" if item["is_credible"] is None
                           else ("passed" if item["is_credible"] else "failed"))
                    )

                    reference = st.text_input(
                        "Reference page", value=item["resolved_title"],
                        key=f"ref_{item['skill_name']}",
                        help="Recorded for provenance. Editing it does not refetch the "
                             "text; the definition below is what gets scored.",
                    )
                    summary = st.text_area(
                        "Definition used for scoring", value=item["wikipedia_summary"],
                        height=160, key=f"sum_{item['skill_name']}",
                    )

                    actions = st.columns(2)
                    if actions[0].button("Approve and score", key=f"ok_{item['skill_name']}",
                                         type="primary", use_container_width=True):
                        if not summary.strip():
                            st.error("The definition cannot be empty.")
                        else:
                            apply_review(item["skill_name"], summary, reference, approve=True)
                            refresh()
                    if actions[1].button("Reject", key=f"no_{item['skill_name']}",
                                         use_container_width=True):
                        apply_review(item["skill_name"], summary, reference, approve=False)
                        refresh()

    else:
        # Post-mortem correction. Same form, applied to something already scored:
        # in a single-user local app, fixing an approved skill and reviewing a pending
        # one are the same action.
        if not SKILLS:
            st.info("Nothing approved yet.")
        else:
            target = st.selectbox(
                "Approved skill", [row["skill_name"] for row in SKILLS], key="edit_select"
            )
            record = next(row for row in SKILLS if row["skill_name"] == target)
            st.caption(
                f"Currently {record['ai_score']:+.4f} ({record['category_bucket']}), "
                f"snapshot {record['snapshot_date']}."
            )

            reference = st.text_input(
                "Reference page", value=record["resolved_title"], key="edit_ref"
            )
            summary = st.text_area(
                "Definition used for scoring", value=record["wikipedia_summary"],
                height=200, key="edit_sum",
            )
            st.caption(
                "Saving rescores and updates this quarter's snapshot in place rather "
                "than adding a second point, so the trend shows no false movement."
            )
            if st.button("Save correction", type="primary"):
                if not summary.strip():
                    st.error("The definition cannot be empty.")
                else:
                    apply_review(target, summary, reference, approve=True)
                    refresh()

    if DATA["discovered"]:
        with st.expander(f"{len(DATA['discovered'])} tools discovered but never audited"):
            st.caption(
                "Recorded from the O*NET occupation map and skipped before any network "
                "call, normally because they are not flagged Hot Technology. They have "
                "no definition, so there is nothing to review; they are listed so the "
                "gap between 'tools discovered' and 'skills scored' is inspectable."
            )
            st.dataframe(
                pd.DataFrame(DATA["discovered"])[
                    ["skill_name", "category", "is_hot_tech_anywhere"]
                ],
                hide_index=True, use_container_width=True, height=280,
            )


# --------------------------------------------------------------------------
# Trends
# --------------------------------------------------------------------------

with trend_tab:
    st.subheader("AI score over time")
    st.caption(
        "Raw cosine score at each quarterly snapshot. Unnormalized, because the point "
        "is movement in the true value."
    )

    trends = pd.DataFrame(DATA["trends"])
    if trends.empty:
        st.info("No snapshots recorded yet.")
    else:
        counts = trends.groupby("skill_name").size().sort_values(ascending=False)
        # Skills with real history first, so the default selection is one that
        # actually has a line to draw rather than a single point.
        ordered = list(counts.index)

        chosen = st.multiselect(
            "Skills", ordered, default=ordered[:1], key="trend_select",
            help="Pick several to compare them on one axis.",
        )
        if not chosen:
            st.info("Select at least one skill.")
        else:
            subset = trends[trends["skill_name"].isin(chosen)]
            figure = px.line(
                subset.sort_values(["skill_name", "quarter"]),
                x="quarter", y="ai_score", color="skill_name", markers=True,
            )
            figure.update_layout(
                xaxis_title="Quarter", yaxis_title="Raw AI score", legend_title=None,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            # Reference lines for the two absolute thresholds, so a reader can see
            # which side of a bucket boundary a skill is drifting toward.
            figure.add_hline(y=0.30, line_dash="dot", line_color=BUCKET_COLOURS["AI Skill"],
                             annotation_text="AI Skill 0.30", annotation_position="right")
            figure.add_hline(y=0.15, line_dash="dot", line_color=BUCKET_COLOURS["AI Enabling Skill"],
                             annotation_text="AI Enabling 0.15", annotation_position="right")
            st.plotly_chart(figure, use_container_width=True)

            single = [name for name in chosen if counts.get(name, 0) < 2]
            if single:
                # Said explicitly rather than drawing a flat line, which would imply a
                # stable measurement where there is only one reading.
                st.info(
                    "Only one snapshot so far for "
                    + ", ".join(single)
                    + ". A second point appears after the next quarterly run."
                )

            st.dataframe(
                subset[["skill_name", "quarter", "snapshot_date", "ai_score",
                        "category_bucket"]].sort_values(["skill_name", "quarter"]),
                hide_index=True, use_container_width=True,
            )
