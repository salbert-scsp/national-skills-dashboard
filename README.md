# National Skills Dashboard

How the U.S. workforce is adapting to technological change, measured through the skills
employers ask for in every occupation, not only in jobs that build AI.

This repository contains the dashboard's data pipeline and web application. Its first
module, the **AI skills catalog**, classifies the software skills listed in O\*NET into
three groups:

| Group | Meaning |
|---|---|
| AI Skill | Tools for building, training or deploying AI models |
| AI Enabling Skill | Data, cloud and programming tools that AI systems run on, and everyday software with verified built-in AI features |
| Not AI Skill | Everything else |

Status: the catalog covers O\*NET's Hot Technologies (1,123 approved skills across 916
occupations). Public labor-market data (BLS) is being added next; job-posting demand
data (NLx Research Hub) will follow once access is approved.

## Acknowledgment

Builds on the AI Skills Dashboard developed by Keira Walker at SCSP (2026), originally at
[github.com/kwalker-scsp/SkillsDashboard](https://github.com/kwalker-scsp/SkillsDashboard).
The full commit history is preserved here.

## Setup

Requires **Python 3.11** (macOS: `brew install python@3.11`).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
sed '/Retired, kept installable/,$d' requirements.txt > requirements-local.txt
pip install -r requirements-local.txt
```

1. **Model files.** Two ONNX models are required and are not in git. See [MODELS.md](MODELS.md)
   for sources and checksums. To run without them, set `SKILLS_NO_MODELS=1` in `.env`
   (read-only mode: stored scores can be viewed; nothing new can be scored).
2. **Settings.** Create `.env` with at least `ADMIN_PASSWORD=...` and `SESSION_SECRET=...`.
   Fetching new skills also needs `CAREERONESTOP_TOKEN` and `CAREERONESTOP_USER_ID`.
3. **Tests:** `python -m pytest -q`
4. **Run:** `python main.py`, then open http://127.0.0.1:8000/dashboard. The review queue at
   http://127.0.0.1:8000/ uses `ADMIN_PASSWORD`.

How the pipeline is organized: [ARCHITECTURE.md](ARCHITECTURE.md).

## Data sources and credits

- **O\*NET** database, U.S. Department of Labor, Employment and Training Administration,
  licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Occupation codes,
  titles and technology skills.
- **Wikipedia and Wikidata.** Skill summaries stored in `skills_master.json` are derived from
  Wikipedia content, licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
- **U.S. Bureau of Labor Statistics** (Occupational Employment and Wage Statistics; Employment
  Projections). Public domain.
- **Embedding models:** sentence-transformers/all-MiniLM-L6-v2 and
  cross-encoder/ms-marco-MiniLM-L6-v2 (Apache 2.0). See MODELS.md.

## Restricted data

Job-posting data from the NLx Research Hub is licensed under a data use agreement and
**must never be committed to this repository**. `.gitignore` blocks the usual file types and
the `data/nlx/` folder; only aggregate results may be published.

## License

[To be confirmed by SCSP.] Until a license is added, the code is visible but not licensed
for reuse.
