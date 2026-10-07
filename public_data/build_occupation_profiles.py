"""
STAGE: Public data (Phase 2)

Builds one row per occupation (6-digit SOC code) that joins:

  1. The AI skills catalog  -- approved skills in skills_master.json, with each skill's
     latest category from skills_timeseries.json
  2. BLS OEWS national file -- employment and wages
  3. BLS Employment Projections, Table 1.2 -- projected 10-year change

Usage (from the project folder, venv active):

    python public_data/build_occupation_profiles.py \
        --oews data/raw/oesm25nat.zip \
        --projections data/raw/occupation.xlsx

Output: data/processed/occupation_profiles.csv

READ THIS BEFORE QUOTING THE NUMBERS. The skill shares count an occupation's O*NET Hot
Technology software skills, not workers and not job postings. "40% of the occupation's
listed software is AI Enabling" is the correct reading; "40% of workers use AI" is not.
Employer demand comes later, from NLx postings.

Offline: reads local files only and makes no network call. It never writes to the skill
store.
"""

import argparse
import io
import json
import os
import re
import sys
import zipfile
from collections import defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER = os.path.join(ROOT, "skills_master.json")
SERIES = os.path.join(ROOT, "skills_timeseries.json")
DEFAULT_OUT = os.path.join(ROOT, "data", "processed", "occupation_profiles.csv")

AI = "AI Skill"
ENABLING = "AI Enabling Skill"
NOT_AI = "Not AI Skill"

# Two general office tools sit in AI Enabling (as Technical Backbone, by text similarity,
# not by a verified AI feature) and are listed for 700+ occupations each. Together they
# account for about half of the worker-weighted AI / AI Enabling share, and they alone put
# occupations such as Fast Food and Counter Workers at 40%. Until the Phase 1 accuracy
# check settles their category, every profile also reports the share without them.
OFFICE_TOOLS = {"Microsoft Excel", "Microsoft Office software"}

# OEWS codes for suppressed or top-coded values.
SUPPRESSED = {"*", "**", "#", "~", "-", ""}


# --- 1. Skill catalog --------------------------------------------------------------

def load_catalog():
    """Approved skills -> (category, sub_category, SOC codes)."""
    with open(MASTER) as f:
        master = json.load(f)
    with open(SERIES) as f:
        series = json.load(f)

    latest = {}
    for row in series:
        name = row["skill_name"]
        if name not in latest or row["snapshot_date"] >= latest[name]["snapshot_date"]:
            latest[name] = row

    catalog = {}
    for name, entry in master.items():
        if entry.get("status") != "approved" or name not in latest:
            continue
        codes = entry.get("onet_codes", []) or []
        titles = entry.get("onet_titles", []) or []
        socs = sorted({code[:7] for code in codes if code})
        catalog[name] = {
            "titles": {c[:7]: t for c, t in zip(codes, titles) if c.endswith(".00")},
            "category": latest[name]["category_bucket"],
            "sub_category": latest[name].get("sub_category") or "",
            "socs": socs,
        }
    return catalog


def skill_profiles(catalog):
    """One row per SOC code with counts and shares of each skill group."""
    reach = {name: len(info["socs"]) for name, info in catalog.items()}
    by_soc = defaultdict(lambda: {AI: [], ENABLING: [], NOT_AI: []})
    titles = {}
    for name, info in catalog.items():
        titles.update(info["titles"])
        for soc in info["socs"]:
            by_soc[soc][info["category"]].append(name)

    rows = []
    for soc, groups in by_soc.items():
        total = sum(len(v) for v in groups.values())
        rest = [n for v in groups.values() for n in v if n not in OFFICE_TOOLS]
        rest_ai = [n for n in groups[AI] + groups[ENABLING] if n not in OFFICE_TOOLS]
        # Most widely used skills first: the ones most relevant to career mobility.
        top_enabling = sorted(groups[ENABLING], key=lambda n: (-reach[n], n))[:5]
        rows.append({
            "soc_code": soc,
            "onet_title": titles.get(soc, ""),
            "n_skills": total,
            "n_ai_skill": len(groups[AI]),
            "n_ai_enabling": len(groups[ENABLING]),
            "n_not_ai": len(groups[NOT_AI]),
            "share_ai_or_enabling": round((len(groups[AI]) + len(groups[ENABLING])) / total, 4),
            "share_ai_or_enabling_excl_office": (round(len(rest_ai) / len(rest), 4)
                                                 if rest else None),
            "share_ai_skill": round(len(groups[AI]) / total, 4),
            "ai_skills": "; ".join(sorted(groups[AI])),
            "top_ai_enabling_skills": "; ".join(top_enabling),
        })
    return pd.DataFrame(rows)


# --- Helpers -----------------------------------------------------------------------

def _norm(col):
    return re.sub(r"\s+", " ", str(col)).strip().lower()


def _find(columns, *patterns, required=True):
    """First column whose normalized name matches every regex in patterns."""
    for col in columns:
        name = _norm(col)
        if all(re.search(p, name) for p in patterns):
            return col
    if required:
        raise SystemExit(f"Could not find a column matching {patterns}. Columns: {list(columns)}")
    return None


def _number(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().replace(",", "")
    if text in SUPPRESSED:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _read_excel_maybe_zipped(path, name_hint):
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            members = [m for m in z.namelist()
                       if m.lower().endswith((".xlsx", ".xls")) and name_hint in m.lower()]
            if not members:
                raise SystemExit(f"No Excel file containing '{name_hint}' inside {path}: {z.namelist()}")
            with z.open(members[0]) as f:
                return pd.read_excel(io.BytesIO(f.read()), dtype=str)
    return pd.read_excel(path, dtype=str)


# --- 2. OEWS -----------------------------------------------------------------------

def load_oews(path):
    df = _read_excel_maybe_zipped(path, "national")
    code = _find(df.columns, r"^occ_code$")
    title = _find(df.columns, r"^occ_title$")
    group = _find(df.columns, r"^o_group$|^occ_group$", required=False)
    emp = _find(df.columns, r"^tot_emp$")
    median = _find(df.columns, r"^a_median$")
    mean = _find(df.columns, r"^a_mean$", required=False)

    if group is not None:
        df = df.assign(_g=df[group].str.strip().str.lower())
        df = df[df["_g"].isin(["detailed", "broad"])]
        # A code can appear as both broad and detailed; prefer the detailed row.
        df = df.sort_values("_g", key=lambda g: g.map({"detailed": 0, "broad": 1}))

    out = pd.DataFrame({
        "soc_code": df[code].str.strip(),
        "oews_title": df[title].str.strip(),
        "employment": df[emp].map(_number),
        "median_annual_wage": df[median].map(_number),
        "mean_annual_wage": df[mean].map(_number) if mean is not None else None,
        "wage_top_coded": df[median].astype(str).str.strip().eq("#"),
    })
    return out.drop_duplicates("soc_code")


# --- 3. Employment Projections -------------------------------------------------------

def load_projections(path):
    sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=str)
    name = next((s for s in sheets if "1.2" in s), next(iter(sheets)))
    raw = sheets[name]

    # The table has title rows above the header; find the header row.
    header_row = next(i for i, row in raw.iterrows()
                      if any("code" in _norm(v) for v in row if isinstance(v, str))
                      and any("employment" in _norm(v) for v in row if isinstance(v, str)))
    df = raw.iloc[header_row + 1:].copy()
    df.columns = raw.iloc[header_row]

    code = _find(df.columns, r"code")
    kind = _find(df.columns, r"occupation type", required=False)
    emp_cols = [c for c in df.columns if re.match(r"^employment, \d{4}$", _norm(c))]
    pct = _find(df.columns, r"change", r"percent")
    if len(emp_cols) < 2:
        raise SystemExit(f"Expected two 'Employment, YYYY' columns, found {emp_cols}")

    if kind is not None:
        df = df[df[kind].astype(str).str.strip().str.lower() == "line item"]

    base, proj = emp_cols[0], emp_cols[1]
    base_year, proj_year = _norm(base)[-4:], _norm(proj)[-4:]
    out = pd.DataFrame({
        "soc_code": df[code].astype(str).str.strip(),
        f"proj_employment_{base_year}_thousands": df[base].map(_number),
        f"proj_employment_{proj_year}_thousands": df[proj].map(_number),
        f"proj_change_pct_{base_year}_{proj_year}": df[pct].map(_number),
    })
    out = out[out["soc_code"].str.match(r"^\d{2}-\d{4}$")]
    return out.drop_duplicates("soc_code")


# --- Matching ------------------------------------------------------------------------

def _merge_with_fallback(profiles, table, label):
    """
    Join on the 6-digit SOC code. BLS publishes some occupations only as a combined BROAD
    group (e.g. the three buyer occupations 13-1021/22/23 appear only as 13-1020). When the
    exact code is missing, fall back to the broad code (first five digits + "0") and record
    that in <label>_match. Broad values are shared by several occupations, so never add
    them up across occupations.
    """
    lookup = table.set_index("soc_code")
    matched_code, level = [], []
    for soc in profiles["soc_code"]:
        broad = soc[:6] + "0"
        if soc in lookup.index:
            matched_code.append(soc); level.append("detailed")
        elif broad in lookup.index:
            matched_code.append(broad); level.append("broad (shared)")
        else:
            matched_code.append(None); level.append("not published")
    key = f"{label}_matched_code"
    profiles[key] = matched_code
    profiles[f"{label}_match"] = level
    joined = table.rename(columns={"soc_code": key})
    return profiles.merge(joined, on=key, how="left")


# --- Main --------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--oews", required=True, help="OEWS national zip or xlsx (oesmYYnat.zip)")
    parser.add_argument("--projections", required=True, help="Employment Projections occupation.xlsx")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    catalog = load_catalog()
    skills = skill_profiles(catalog)
    oews = load_oews(args.oews)
    proj = load_projections(args.projections)

    profiles = skills.copy()
    profiles = _merge_with_fallback(profiles, oews, "oews")
    profiles = _merge_with_fallback(profiles, proj, "proj")
    profiles["onet_title"] = profiles["onet_title"].where(
        profiles["onet_title"].astype(bool), profiles["oews_title"]).fillna("")
    profiles.insert(1, "major_group", profiles["soc_code"].str[:2])
    profiles = profiles.sort_values("soc_code")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    profiles.to_csv(args.out, index=False)

    print(f"Approved skills in catalog:     {len(catalog):,}")
    print(f"Occupations with skills:        {len(profiles):,}")
    for label, name in (("oews", "OEWS employment"), ("proj", "projections")):
        counts = profiles[f"{label}_match"].value_counts().to_dict()
        print(f"  {name + ':':<28} " + ", ".join(f"{v} {k}" for k, v in counts.items()))
    # Count each published OEWS row once. A broad group already includes its detailed
    # occupations, so drop any detailed code whose broad group is also counted.
    codes = set(profiles["oews_matched_code"].dropna())
    broad_codes = {c for c in codes if (profiles["oews_matched_code"].eq(c)
                                        & profiles["oews_match"].eq("broad (shared)")).any()}
    codes = {c for c in codes if c in broad_codes or c[:6] + "0" not in broad_codes}
    covered = oews.set_index("soc_code").loc[sorted(codes), "employment"].sum()
    print(f"Workers in those occupations:   {covered:,.0f}")
    print(f"Wrote {args.out}")

    weighted = profiles[profiles["oews_match"] == "detailed"].dropna(
        subset=["employment", "share_ai_or_enabling_excl_office"])
    def worker_weighted(col):
        return (weighted[col] * weighted["employment"]).sum() / weighted["employment"].sum()
    print(f"\nWorker-weighted AI / AI Enabling share: {worker_weighted('share_ai_or_enabling'):.1%}"
          f"  (excluding Excel and Office: {worker_weighted('share_ai_or_enabling_excl_office'):.1%})")

    # Occupations with few listed skills swing to extreme shares, so require 10+.
    outside_tech = profiles[(profiles["major_group"] != "15") & (profiles["n_skills"] >= 10)]
    top = outside_tech.nlargest(10, "share_ai_or_enabling")
    print("\nHighest AI / AI Enabling share outside computer & math occupations (10+ skills):")
    for _, r in top.iterrows():
        title = r["onet_title"] or r["soc_code"]
        print(f"  {r['share_ai_or_enabling']:.0%}  {title}  ({r['n_skills']} skills)")


if __name__ == "__main__":
    sys.exit(main())
