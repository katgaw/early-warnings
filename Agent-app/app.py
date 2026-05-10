"""
===========================================================
INVESTMENT BANK EARLY WARNING SYSTEM
===========================================================

This app monitors financial risks using:

1. IMF RAG knowledge search (search_financial_stress_knowledge)
2. Agent surveillance + tool-use logging
3. Structured LLM extraction → CSV rows (matches Streamlit dashboard)
4. Historical analog impact enrichment (Qdrant + LLM)

-----------------------------------------------------------
WORKFLOW
-----------------------------------------------------------

Search IMF / gather tool evidence
        ↓
Structured warning rows (risk_category, chunk, URL, …)
        ↓
Save to early_warning_signals.csv
        ↓
Add historical_analog_impact per yes-row

-----------------------------------------------------------
RUN
-----------------------------------------------------------

  python app.py

Env: EARLY_WARNING_AGENT_MODEL, EARLY_WARNING_STRUCT_MODEL, EARLY_WARNING_MAX_YES_ROWS,
EARLY_WARNING_APPEND_CSV, EARLY_WARNING_CHUNK_CSV_MAX, EARLY_WARNING_AGENT_*_LIMIT, …

===========================================================
"""

from __future__ import annotations

import csv
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

# =========================================================
# ENVIRONMENT SETUP
# =========================================================

APP_DIR = Path(__file__).resolve().parent
os.chdir(APP_DIR)

from dotenv import load_dotenv

load_dotenv(APP_DIR / ".env")
load_dotenv()

# =========================================================
# WARNINGS (LangGraph / LangChain deprecations)
# =========================================================

try:
    from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
except ImportError:
    LangChainPendingDeprecationWarning = DeprecationWarning  # type: ignore[misc, assignment]

warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)
warnings.filterwarnings(
    "ignore",
    message=r"The default value of [`']allowed_objects[`'] will change.*",
)

# =========================================================
# IMPORTS
# =========================================================

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    after_model,
    before_model,
)
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from news_tool import assess_impact_for_csv_row, assess_impact_for_signal
from rag_tool import search_financial_stress_knowledge

# =========================================================
# PATHS & MODELS
# =========================================================

CSV_PATH = APP_DIR / "early_warning_signals.csv"
AGENT_MODEL = os.environ.get("EARLY_WARNING_AGENT_MODEL", "gpt-4o-mini")
STRUCTURED_MODEL = os.environ.get("EARLY_WARNING_STRUCT_MODEL", "gpt-4o-mini")

# =========================================================
# RAG FALLBACK (no tool messages)
# =========================================================

RAG_FALLBACK_QUERIES = [
    "IMF geopolitical conflict military escalation sanctions financial stability transmission banking",
    "IMF oil commodity energy price shock inflation financial stress emerging markets",
    "IMF bank liquidity funding stress systemic banking crisis vulnerability indicators",
    "IMF cyber attack operational resilience financial sector digital infrastructure risk",
]

# =========================================================
# RISK CATEGORIES (aligned with streamlit_app.RISK_CATEGORY_ORDER)
# =========================================================

RISK_CATEGORIES = [
    "US military escalation",
    "oil price spike",
    "bank liquidity crisis",
    "cyberattack on banks",
]

RISK_CATEGORY_LITERAL = Literal[
    "US military escalation",
    "oil price spike",
    "bank liquidity crisis",
    "cyberattack on banks",
]

# =========================================================
# CSV HEADERS (must match streamlit_app.CSV_COLUMNS)
# =========================================================

CSV_HEADERS = [
    "timestamp",
    "risk_category",
    "early warning",
    "article headlines",
    "source country",
    "potential impact (high/medium/low)",
    "chunk",
    "URL",
    "historical_analog_impact",
]

# =========================================================
# AGENT PROMPT
# =========================================================

SYSTEM_PROMPT = """You are a bank risk surveillance analyst.

Rules:
- Use tools for factual claims; never invent URLs or quotes.
- search_financial_stress_knowledge: run **four separate searches**, one per theme (parallel OK):
  (1) US military escalation — conflict, sanctions, geopolitical shocks to finance
  (2) oil price spike — commodities, energy, inflation spillovers
  (3) bank liquidity crisis — funding stress, systemic banking
  (4) cyberattack on banks — cyber risk, operational resilience
- assess_impact_for_signal: skip unless one live demo is essential; batch analog impacts run after CSV.

End with a brief risk summary."""

# =========================================================
# AGENT MIDDLEWARE (terminal logs)
# =========================================================

_agent_llm_step = 0


@before_model
def log_before_agent_model(state, runtime):
    global _agent_llm_step
    _agent_llm_step += 1
    n = len(state.get("messages", []))
    print(
        f"[agent LLM] model={AGENT_MODEL!r} | call #{_agent_llm_step} | "
        f"messages_in_thread={n}"
    )
    return None


@after_model
def log_after_agent_model(state, runtime):
    msgs = state.get("messages") or []
    last = msgs[-1] if msgs else None
    if not last:
        return None
    raw_calls = getattr(last, "tool_calls", None) or []
    if not raw_calls:
        print("[agent LLM] tools requested: (none — model returned text)")
        return None
    names: list[str] = []
    for tc in raw_calls:
        if isinstance(tc, dict):
            names.append(str(tc.get("name") or "?"))
        else:
            names.append(str(getattr(tc, "name", None) or "?"))
    print(f"[agent LLM] tools requested: {', '.join(names)}")
    return None


_agent_call_limiter = ModelCallLimitMiddleware(
    thread_limit=int(os.environ.get("EARLY_WARNING_AGENT_THREAD_LIMIT", "25")),
    run_limit=int(os.environ.get("EARLY_WARNING_AGENT_RUN_LIMIT", "25")),
    exit_behavior="end",
)

# =========================================================
# CREATE AGENT
# =========================================================

agent = create_agent(
    model=AGENT_MODEL,
    tools=[search_financial_stress_knowledge, assess_impact_for_signal],
    system_prompt=SYSTEM_PROMPT,
    middleware=[
        log_before_agent_model,
        log_after_agent_model,
        _agent_call_limiter,
    ],
)


# =========================================================
# STRUCTURED OUTPUT SCHEMA (CSV row extraction)
# =========================================================

class EarlyWarningRow(BaseModel):
    early_warning: Literal["yes", "no"] = Field(
        description='"yes" only for clearly material bank-relevant risks; prefer "no".'
    )
    risk_category: RISK_CATEGORY_LITERAL = Field()
    article_headlines: str = Field()
    source_country: str = Field()
    potential_impact: Literal["high", "medium", "low"] = Field()
    chunk: str = Field(description="Excerpt from tool evidence.")
    url: str = Field(description="Exact citation from evidence.")


class EarlyWarningAssessment(BaseModel):
    rows: list[EarlyWarningRow]


# =========================================================
# EVIDENCE HELPERS
# =========================================================

def tool_evidence(messages) -> str:
    return "\n\n=====\n\n".join(
        str(m.content)
        for m in messages
        if isinstance(m, ToolMessage) and m.content
    )


def rag_fallback_evidence() -> str:
    return "\n\n=====\n\n".join(
        search_financial_stress_knowledge.invoke({"query": q}) for q in RAG_FALLBACK_QUERIES
    )


# =========================================================
# STRUCTURED ROW EXTRACTION
# =========================================================

def _max_yes_rows() -> int | None:
    raw = os.environ.get("EARLY_WARNING_MAX_YES_ROWS", "").strip()
    if raw.isdigit():
        return max(0, int(raw))
    return None


def rows_from_evidence(evidence: str) -> EarlyWarningAssessment:
    cap_ev = int(os.environ.get("EARLY_WARNING_MAX_EVIDENCE_CHARS", "120000"))
    if len(evidence) > cap_ev:
        evidence = evidence[:cap_ev] + "\n\n[TRUNCATED]\n"
    print(
        f"[structured LLM] model={STRUCTURED_MODEL!r} | "
        f"evidence_chars={len(evidence)} (CSV row extraction)"
    )
    llm = ChatOpenAI(model=STRUCTURED_MODEL, temperature=0)
    structured = llm.with_structured_output(EarlyWarningAssessment)
    max_yes = _max_yes_rows()
    cap_line = ""
    if max_yes is not None:
        cap_line = (
            f"\n- At most {max_yes} row(s) may use early_warning=yes; "
            "all other rows early_warning=no.\n"
        )
    themes = " | ".join(RISK_CATEGORIES)
    prompt = f"""Build early-warning rows from tool evidence only.

- Distinct URLs or IMF [Source N] blocks; do not merge unrelated sources.
- For each source, consider all four themes ({themes}).
  If multiple themes are material, one row per (source + theme); reuse url; tailor chunk/headline to that theme.
- early_warning=yes only when material; chunk and url must match evidence.
{cap_line}
Evidence:
{evidence}
"""
    return structured.invoke(prompt)


# =========================================================
# WRITE CSV
# =========================================================

def write_csv(path: Path, assessment: EarlyWarningAssessment) -> int:
    yes = [r for r in assessment.rows if r.early_warning == "yes"]
    cap = _max_yes_rows()
    if cap is not None:
        yes = yes[:cap]
        print(f"[csv] EARLY_WARNING_MAX_YES_ROWS={cap} → writing {len(yes)} yes row(s)")
    append = os.environ.get("EARLY_WARNING_APPEND_CSV", "").lower() in ("1", "true", "yes")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    mode = "a" if append and path.is_file() else "w"
    write_header = mode == "w" or not path.is_file()
    chunk_cap = int(os.environ.get("EARLY_WARNING_CHUNK_CSV_MAX", "2500"))

    if not yes:
        if append and path.is_file():
            print("[csv] no yes rows; nothing appended.")
            return 0
        if mode == "w" and path.is_file():
            try:
                with path.open(newline="", encoding="utf-8") as f:
                    existing_rows = sum(1 for _ in csv.DictReader(f))
                if existing_rows > 0:
                    print(
                        "[csv] no yes rows from this run; keeping existing CSV unchanged."
                    )
                    return 0
            except Exception:
                pass

    with path.open(mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if write_header:
            w.writeheader()
        for r in yes:
            chunk = r.chunk if len(r.chunk) <= chunk_cap else r.chunk[: chunk_cap - 3] + "..."
            w.writerow({
                "timestamp": ts,
                "risk_category": r.risk_category,
                "early warning": r.early_warning,
                "article headlines": r.article_headlines,
                "source country": r.source_country,
                "potential impact (high/medium/low)": r.potential_impact,
                "chunk": chunk,
                "URL": r.url,
                "historical_analog_impact": "",
            })
    return len(yes)


# =========================================================
# HISTORICAL ANALOG IMPACTS (enrich yes-rows)
# =========================================================

def enrich_impacts(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        if "historical_analog_impact" not in fields:
            fields.append("historical_analog_impact")
        out, n = [], 0
        for raw in reader:
            row = {fn: (raw.get(fn) or "").strip() for fn in fields}
            if row.get("early warning", "").lower() == "yes":
                row["historical_analog_impact"] = assess_impact_for_csv_row(row)
                n += 1
            out.append(row)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in out:
            w.writerow({k: row.get(k, "") for k in fields})
    return n


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    global _agent_llm_step
    _agent_llm_step = 0
    print(
        f"Starting surveillance | AGENT_MODEL={AGENT_MODEL!r} | "
        f"STRUCTURED_MODEL={STRUCTURED_MODEL!r}"
    )
    response = agent.invoke({
        "messages": [{
            "role": "user",
            "content": (
                "Run surveillance: search_financial_stress_knowledge **four times** "
                "(military/geopolitical, oil/commodities, bank liquidity, cyber). "
                "Skip assess_impact_for_signal unless essential for a demo."
            ),
        }],
    })
    evidence = tool_evidence(response["messages"]).strip()
    if not evidence:
        print("No tool messages; RAG fallback.")
        evidence = rag_fallback_evidence()
    print("Structured rows…")
    assessment = rows_from_evidence(evidence)
    written = write_csv(CSV_PATH, assessment)
    print("Analog impacts…")
    impacts = enrich_impacts(CSV_PATH)
    print(f"Done. rows={written}, impacts={impacts}. {CSV_PATH.resolve()}")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
