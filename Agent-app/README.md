# Investment Bank Risk Monitor

Early-warning pipeline: an agent gathers evidence from an **IMF PDF (RAG)** and **Google News RSS**, then a second model turns that text into rows in a CSV. Each signal’s **`risk_category`** is always one of four themes (news keywords); IMF excerpts are mapped to the best-fitting theme. The **Streamlit** app reads the CSV and shows signals by day.

---

## Architecture

```
                ┌─────────────────────┐
                │   User Request      │
                │ "Find early risks"  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  AGENT LLM          │
                │  (GPT-5*)           │
                │                     │
                │ • Decides what to   │
                │   search            │
                │ • Uses tools        │
                │ • Collects evidence │
                └──────────┬──────────┘
                           │
          ┌────────────────┴────────────────┐
          │                                 │
          ▼                                 ▼

 ┌───────────────────┐          ┌───────────────────┐
 │ RAG Knowledge Base│          │ Google News (RSS) │
 │                   │          │                   │
 │ • IMF PDF chunks  │          │ • Live headlines  │
 │ • Qdrant + embed  │          │ • Four risk themes│
 └─────────┬─────────┘          └─────────┬─────────┘
           │                              │
           └──────────────┬───────────────┘
                          ▼

               ┌─────────────────────┐
               │ STRUCTURED LLM      │
               │ (GPT-4o-mini*)      │
               │                     │
               │ • Extracts signals  │
               │ • Categories + CSV  │
               └──────────┬──────────┘
                          ▼

               ┌─────────────────────┐
               │ early_warning_      │
               │ signals.csv         │
               └─────────────────────┘
                          │
                          ▼
               ┌─────────────────────┐
               │ Streamlit dashboard │
               └─────────────────────┘
```

\*Default models; override with `EARLY_WARNING_AGENT_MODEL` and `EARLY_WARNING_STRUCT_MODEL`.

---

## What you need

| Item | Notes |
|------|--------|
| Python 3.10+ | |
| `OPENAI_API_KEY` | In `.env` or environment (embeddings + both LLMs) |
| `data/imf-report.pdf` | IMF report used for RAG |
| `data/image.png` | Optional logo in Streamlit |

No paid news API: **`news_tool.py`** uses public Google News RSS (`feedparser`).

---

## Quick start

```bash
cd Agent-app    # same layout exists under agent_app/
pip install -r ../requirements.txt
```

**CLI (writes / overwrites CSV by default)**

```bash
python app.py
```

**Dashboard (loads `.env`, auto-runs `app.py` once if the CSV is empty, then shows results)**

```bash
streamlit run streamlit_app.py
```

---

## Useful environment variables

| Variable | Purpose |
|----------|---------|
| `EARLY_WARNING_APPEND_CSV=1` | Append new scan rows instead of replacing the CSV (history by day in the UI) |
| `EARLY_WARNING_AGENT_MODEL` | Agent model id (default `gpt-5`) |
| `EARLY_WARNING_STRUCT_MODEL` | Structured extraction model (default `gpt-4o-mini`) |
| `EARLY_WARNING_MAX_EVIDENCE_CHARS` | Cap evidence size for the second LLM (default `120000`) |
| `EARLY_WARNING_CHUNK_CSV_MAX` | Max characters stored in CSV `chunk` column (default `2500`) |

---

## Main files

| File | Role |
|------|------|
| `app.py` | Agent + structured extraction → `early_warning_signals.csv` |
| `rag_tool.py` | Load PDF, chunk, embed, Qdrant, `search_financial_stress_knowledge` |
| `news_tool.py` | Historical analog retrieval + `assess_impact_for_signal` / `assess_impact_for_csv_row` |
| `streamlit_app.py` | Logo, headings, expanders, risk level badges |

---

## Troubleshooting

- **Empty CSV after a run** — The structured model may have marked every item as not an early warning; try **Run new scan** in Streamlit or inspect stderr in the debug expander.
- **Import / PDF errors** — Run from `Agent-app` (or ensure `data/imf-report.pdf` sits next to `rag_tool.py` under `data/`).
- **Keys not picked up** — Put them in `Agent-app/.env`; Streamlit loads it before spawning `app.py`.
