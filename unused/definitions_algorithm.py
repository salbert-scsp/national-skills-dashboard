import os
import json
import re
import time
import warnings
from urllib.parse import quote, unquote

import numpy as np
import onnxruntime as ort
import pyodbc
import requests
from dotenv import load_dotenv
from tokenizers import Tokenizer

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
load_dotenv()

CAREERONESTOP_TOKEN = os.environ.get("CAREERONESTOP_TOKEN", "").strip().replace(";", "")
CAREERONESTOP_USER_ID = os.environ.get("CAREERONESTOP_USER_ID", "").strip()
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()
CACHE_FILE = os.environ.get("CACHE_FILE", "wikipedia_skills_cache.json")

NOISE_CLEANER = re.compile(
    r'(?i)\b(software|systems?|analytics|big data software|tools?|database|language|platform|framework|library)\b'
)


def get_db_connection():
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


class CrossEncoderEvaluator:
    """Evaluates individual source payloads and returns per-source confidence scores."""

    def __init__(self, model_path: str = "cross_encoder_model.onnx"):
        self.tokenizer = Tokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.tokenizer.enable_truncation(max_length=256, direction="right")
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    def evaluate_source(self, query: str, candidate_title: str, candidate_text: str) -> float:
        if not candidate_title or not candidate_text:
            return 0.0

        encoded = self.tokenizer.encode(query.lower().strip(), f"{candidate_title}. {candidate_text}".lower().strip())
        onnx_inputs = {
            "input_ids": np.array([encoded.ids], dtype=np.int64),
            "attention_mask": np.array([encoded.attention_mask], dtype=np.int64),
            "token_type_ids": np.array([encoded.type_ids], dtype=np.int64),
        }

        raw_logits = self.session.run(None, onnx_inputs)[0]
        logit_val = float(raw_logits[0][0])
        return round(float(1.0 / (1.0 + np.exp(-logit_val))), 4)


class SourceFetcher:
    def __init__(self, session: requests.Session):
        self.session = session
        self.wiki_headers = {
            "User-Agent": os.environ.get("WIKI_USER_AGENT", "AISkillsAnalyticsDashboard/1.0 (kwalker@scsp.ai)")
        }
        self.github_headers = {"Accept": "application/vnd.github.v3+json"}
        if GITHUB_TOKEN:
            self.github_headers["Authorization"] = f"token {GITHUB_TOKEN}"

    def fetch_wikipedia(self, query: str, category: str) -> dict | None:
        cleaned = NOISE_CLEANER.sub('', query).strip() or query
        params = {
            "action": "query",
            "list": "search",
            "srsearch": f"{cleaned} {category} computer technology",
            "srlimit": 1,
            "format": "json"
        }
        try:
            res = self.session.get("https://en.wikipedia.org/w/api.php", params=params, headers=self.wiki_headers, timeout=4)
            if res.status_code == 200:
                results = res.json().get("query", {}).get("search", [])
                if results:
                    title = results[0].get("title")
                    summary = self._fetch_wiki_summary(title)
                    if summary:
                        return {"title": title, "summary": summary}
        except Exception:
            pass
        return None

    def _fetch_wiki_summary(self, title: str) -> str:
        try:
            sanitized = quote(title.replace(" ", "_"))
            res = self.session.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{sanitized}", headers=self.wiki_headers, timeout=3)
            if res.status_code == 200:
                return res.json().get("extract", "").strip()
        except Exception:
            pass
        return ""

    def fetch_github(self, query: str) -> dict | None:
        cleaned = NOISE_CLEANER.sub('', query).strip() or query
        url = f"https://api.github.com/search/repositories?q={quote(cleaned)}+in:name&sort=stars&order=desc&per_page=1"
        try:
            res = self.session.get(url, headers=self.github_headers, timeout=4)
            if res.status_code == 200:
                items = res.json().get("items", [])
                if items:
                    top = items[0]
                    return {
                        "title": top.get("full_name"),
                        "summary": top.get("description") or f"GitHub repository for {cleaned}."
                    }
        except Exception:
            pass
        return None

    def fetch_pypi(self, query: str) -> dict | None:
        cleaned = re.sub(r'[^a-zA-Z0-9\-_]', '', query.lower().strip())
        url = f"https://pypi.org/pypi/{cleaned}/json"
        try:
            res = self.session.get(url, timeout=3)
            if res.status_code == 200:
                info = res.json().get("info", {})
                return {
                    "title": info.get("name"),
                    "summary": info.get("summary") or f"PyPI library for {cleaned}."
                }
        except Exception:
            pass
        return None


class DefinitionsPipeline:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {CAREERONESTOP_TOKEN}",
            "Accept": "application/json"
        })
        self.evaluator = CrossEncoderEvaluator()
        self.fetcher = SourceFetcher(self.session)

    def process_skill(self, skill_name: str, category: str) -> dict:
        query = f"{skill_name} ({category})"
        
        sources_payload = {
            "wiki": {"title": None, "summary": None, "score": None},
            "github": {"title": None, "summary": None, "score": None},
            "pypi": {"title": None, "summary": None, "score": None},
        }

        # 1. Evaluate Wikipedia
        wiki_data = self.fetcher.fetch_wikipedia(skill_name, category)
        if wiki_data:
            score = self.evaluator.evaluate_source(query, wiki_data["title"], wiki_data["summary"])
            sources_payload["wiki"] = {"title": wiki_data["title"], "summary": wiki_data["summary"], "score": score}

        # 2. Evaluate GitHub
        gh_data = self.fetcher.fetch_github(skill_name)
        if gh_data:
            score = self.evaluator.evaluate_source(query, gh_data["title"], gh_data["summary"])
            sources_payload["github"] = {"title": gh_data["title"], "summary": gh_data["summary"], "score": score}

        # 3. Evaluate PyPI
        pypi_data = self.fetcher.fetch_pypi(skill_name)
        if pypi_data:
            score = self.evaluator.evaluate_source(query, pypi_data["title"], pypi_data["summary"])
            sources_payload["pypi"] = {"title": pypi_data["title"], "summary": pypi_data["summary"], "score": score}

        # Flag skill if ANY active source score is under 0.90
        active_scores = [v["score"] for k, v in sources_payload.items() if v["score"] is not None]
        
        if active_scores and all(s >= 0.90 for s in active_scores):
            is_approved = 1  # Auto-approve
        else:
            is_approved = 0  # Pull for HITL Review

        return {
            "payload": sources_payload,
            "is_approved": is_approved
        }

    def run(self, target_codes: list[str]):
        local_cache = self._load_cache()
        queue_rows = []

        for code in target_codes:
            print(f"\nProcessing O*NET Code: {code}")
            url = f"https://api.careeronestop.org/v1/occupation/{CAREERONESTOP_USER_ID}/{code}/US?skills=false&toolsAndTechnology=true&tasks=false&alternateOnetTitles=false"

            try:
                res = self.session.get(url, timeout=6)
                if res.status_code != 200:
                    continue
                details = res.json().get("OccupationDetail", [])
                if not details:
                    continue

                onet_title = details[0].get("OnetTitle", "Unknown Title")
                tools_tech = details[0].get("ToolsAndTechOccupationDetails", {}) or {}
                tech_list = (tools_tech.get("Technology", {}).get("CategoryList", []) or []) + \
                            (tools_tech.get("Tools", {}).get("CategoryList", []) or [])

                for item in tech_list:
                    category = item.get("Title", "").strip()
                    for ex in (item.get("Examples", []) or []):
                        skill_name = ex.get("Name", "").strip()
                        if not skill_name:
                            continue

                        if skill_name in local_cache:
                            result = local_cache[skill_name]
                        else:
                            print(f" -> Cross-Encoder Evaluating 3 Sources: {skill_name}")
                            result = self.process_skill(skill_name, category)
                            local_cache[skill_name] = result

                        p = result["payload"]
                        queue_rows.append({
                            "skill_name": skill_name,
                            "category": category,
                            "wiki_title": p["wiki"]["title"],
                            "wiki_summary": p["wiki"]["summary"],
                            "wiki_score": p["wiki"]["score"],
                            "github_title": p["github"]["title"],
                            "github_summary": p["github"]["summary"],
                            "github_score": p["github"]["score"],
                            "pypi_title": p["pypi"]["title"],
                            "pypi_summary": p["pypi"]["summary"],
                            "pypi_score": p["pypi"]["score"],
                            "onet_code": code,
                            "onet_title": onet_title,
                            "is_hot_tech": ex.get("Hot_Technology") == "Y",
                            "is_approved": result["is_approved"],
                            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                        })

            except Exception as err:
                print(f"Error processing {code}: {err}")

        self._save_cache(local_cache)
        self._insert_to_db(queue_rows)
        print(f"\nCycle Complete. Processed {len(queue_rows)} skills with per-source confidence scores.")

    def _load_cache(self) -> dict:
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_cache(self, cache_data: dict):
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=4, ensure_ascii=False)
        except Exception:
            pass

    def _insert_to_db(self, rows: list[dict]):
        if not rows:
            return
        sql = (
            "INSERT INTO HITL_Validation_Queue "
            "(skill_name, category, wiki_title, wiki_summary, wiki_score, "
            "github_title, github_summary, github_score, "
            "pypi_title, pypi_summary, pypi_score, "
            "onet_code, onet_title, is_hot_tech, is_approved, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        with get_db_connection() as conn:
            cursor = conn.cursor()
            for row in rows:
                try:
                    cursor.execute(
                        sql,
                        row["skill_name"],
                        row["category"],
                        row["wiki_title"],
                        row["wiki_summary"],
                        row["wiki_score"],
                        row["github_title"],
                        row["github_summary"],
                        row["github_score"],
                        row["pypi_title"],
                        row["pypi_summary"],
                        row["pypi_score"],
                        row["onet_code"],
                        row["onet_title"],
                        int(row["is_hot_tech"]),
                        row["is_approved"],
                        row["created_at"]
                    )
                except Exception as e:
                    print(f"Failed to insert queue row: {e}")
            conn.commit()


if __name__ == "__main__":
    pipeline = DefinitionsPipeline()
    pipeline.run(target_codes=["15-1252.00", "15-2051.00"])