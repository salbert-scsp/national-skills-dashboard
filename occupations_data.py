"""
STAGE: Web App (read model)

Loads data/processed/occupation_profiles.csv -- built by
public_data/build_occupation_profiles.py -- for the /occupations page.

Read-only and standard library only (csv), so the web app gains no new dependency.
The file is re-read only when it changes on disk.
"""

import csv
import logging
import os

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
PROFILES_FILE = os.getenv(
    "OCCUPATION_PROFILES_FILE",
    os.path.join(_HERE, "data", "processed", "occupation_profiles.csv"),
)

# SOC 2018 major groups, for the filter and the table.
MAJOR_GROUPS = {
    "11": "Management", "13": "Business and Financial Operations",
    "15": "Computer and Mathematical", "17": "Architecture and Engineering",
    "19": "Life, Physical, and Social Science", "21": "Community and Social Service",
    "23": "Legal", "25": "Educational Instruction and Library",
    "27": "Arts, Design, Entertainment, Sports, and Media",
    "29": "Healthcare Practitioners and Technical", "31": "Healthcare Support",
    "33": "Protective Service", "35": "Food Preparation and Serving Related",
    "37": "Building and Grounds Cleaning and Maintenance", "39": "Personal Care and Service",
    "41": "Sales and Related", "43": "Office and Administrative Support",
    "45": "Farming, Fishing, and Forestry", "47": "Construction and Extraction",
    "49": "Installation, Maintenance, and Repair", "51": "Production",
    "53": "Transportation and Material Moving", "55": "Military Specific",
}

_NUMERIC = ("n_skills", "n_ai_skill", "n_ai_enabling", "n_not_ai", "share_ai_or_enabling",
            "share_ai_or_enabling_excl_office", "share_ai_skill", "employment",
            "median_annual_wage", "mean_annual_wage")

_cache = {"mtime": None, "data": None}


def _num(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _split(value):
    return [part.strip() for part in (value or "").split(";") if part.strip()]


def _load(path):
    with open(path, newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    if not raw:
        raise ValueError("occupation_profiles.csv has no rows")

    fields = raw[0].keys()
    change_col = next((c for c in fields if c.startswith("proj_change_pct_")), None)
    years = change_col.replace("proj_change_pct_", "").split("_") if change_col else ["", ""]

    rows = []
    for r in raw:
        row = {k: _num(r.get(k)) for k in _NUMERIC}
        row.update({
            "soc_code": r["soc_code"],
            "title": r.get("onet_title") or r.get("oews_title") or r["soc_code"],
            "major_group": r.get("major_group") or r["soc_code"][:2],
            "major_group_name": MAJOR_GROUPS.get(r.get("major_group") or r["soc_code"][:2], "Other"),
            "ai_skills": _split(r.get("ai_skills")),
            "top_ai_enabling_skills": _split(r.get("top_ai_enabling_skills")),
            "proj_change_pct": _num(r.get(change_col)) if change_col else None,
            "oews_match": r.get("oews_match") or "",
            "oews_matched_code": r.get("oews_matched_code") or "",
            "proj_match": r.get("proj_match") or "",
            "wage_top_coded": (r.get("wage_top_coded") or "").lower() == "true",
        })
        rows.append(row)
    rows.sort(key=lambda x: x["title"])
    return {"rows": rows, "summary": _summary(rows),
            "proj_years": {"base": years[0], "end": years[-1]}}


def _summary(rows):
    # Workers: count each published BLS row once; a broad group already includes its
    # detailed occupations (same rule as build_occupation_profiles.py).
    broad = {r["oews_matched_code"] for r in rows if r["oews_match"] == "broad (shared)"}
    seen, workers = set(), 0.0
    for r in rows:
        code = r["oews_matched_code"]
        if not code or code in seen or r["employment"] is None:
            continue
        if r["oews_match"] == "detailed" and code[:6] + "0" in broad:
            continue
        seen.add(code)
        workers += r["employment"]

    weighted = [r for r in rows if r["oews_match"] == "detailed" and r["employment"]
                and r["share_ai_or_enabling_excl_office"] is not None]
    total = sum(r["employment"] for r in weighted) or 1.0

    def worker_weighted(key):
        return sum(r[key] * r["employment"] for r in weighted) / total

    return {
        "occupations": len(rows),
        "workers": workers,
        "share_all": worker_weighted("share_ai_or_enabling"),
        "share_excl_office": worker_weighted("share_ai_or_enabling_excl_office"),
        "skills_listed": int(sum(r["n_skills"] or 0 for r in rows)),
    }


def load_occupations():
    """Returns {"failed": bool, "missing": bool, ...data}. Never raises."""
    try:
        mtime = os.path.getmtime(PROFILES_FILE)
    except OSError:
        return {"failed": False, "missing": True, "rows": [], "summary": None}
    if _cache["mtime"] != mtime:
        try:
            _cache["data"] = _load(PROFILES_FILE)
            _cache["mtime"] = mtime
        except Exception:
            logger.exception("Could not read %s", PROFILES_FILE)
            return {"failed": True, "missing": False, "rows": [], "summary": None}
    return {"failed": False, "missing": False, **_cache["data"]}
