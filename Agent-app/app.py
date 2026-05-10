"""
===========================================================
EARLY WARNING RISK MONITORING SYSTEM
===========================================================
This project monitors potential financial risk signals using:

1. RAG Knowledge Base
   - IMF reports
   - Banking stress documents
   - Financial contagion research

2. Real-Time News Search
   - Geopolitical events
   - Cyberattacks
   - Oil price shocks
   - Banking crises

-----------------------------------------------------------
TWO LLM ARCHITECTURE
-----------------------------------------------------------
LLM #1 → AGENT MODEL (GPT-5)
--------------------------------
Role:
- Thinks like a financial analyst
- Decides what to search
- Calls tools
- Collects evidence

LLM #2 → STRUCTURED MODEL (GPT-4o-mini)
----------------------------------------
Role:
- Reads collected evidence
- Extracts important signals
- Creates structured CSV rows

-----------------------------------------------------------
FINAL OUTPUT
-----------------------------------------------------------
Creates:
    early_warning_signals.csv

Containing:
- risk signal
- headline
- country
- impact level
- evidence
- URL

Prerequisites:
- pip install -r ../requirements.txt (includes feedparser for news_tool)
- OPENAI_API_KEY in environment or .env (RAG embeddings + LLMs)
- data/imf-report.pdf next to rag_tool.py (see rag_tool PDF_PATH)

Run from this directory: python app.py (working directory is set automatically).
===========================================================
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

# Cwd + .env before tool imports (rag_tool loads PDF/embeddings; needs OPENAI_API_KEY).
APP_DIR = Path(__file__).resolve().parent
os.chdir(APP_DIR)

from dotenv import load_dotenv

load_dotenv(APP_DIR / ".env")
load_dotenv()

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    after_model,
    before_model,
)
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from news_tool import RISK_NEWS_KEYWORDS, search_risk_news_articles
from rag_tool import search_financial_stress_knowledge

CSV_PATH = APP_DIR / "early_warning_signals.csv"

# =========================================================
# MODEL CONFIGURATION
# =========================================================

AGENT_MODEL = os.environ.get("EARLY_WARNING_AGENT_MODEL", "gpt-5")
STRUCTURED_MODEL = os.environ.get("EARLY_WARNING_STRUCT_MODEL", "gpt-4o-mini")

# =========================================================
# SEARCH CONFIGURATION
# =========================================================

RAG_QUERIES = [
    "systemic financial stress banking liquidity risk indicators vulnerability",
    "geopolitical shocks commodity prices contagion emerging market stress",
]
NEWS_KEYWORDS = list(RISK_NEWS_KEYWORDS)

# =========================================================
# CSV CONFIGURATION
# =========================================================

RISK_CATEGORY_LITERAL = Literal[
    "US military escalation",
    "oil price spike",
    "bank liquidity crisis",
    "cyberattack on banks",
]

CSV_HEADERS = [
    "timestamp",
    "risk_category",
    "early warning",
    "article headlines",
    "source country",
    "potential impact (high/medium/low)",
    "chunk",
    "URL",
]

# =========================================================
# LOGGING MIDDLEWARE
# =========================================================

model_call_count = 0


@before_model
def log_before_model(state, runtime):
    """Runs BEFORE every LLM call; logs call count and message depth."""
    global model_call_count
    model_call_count += 1
    message_count = len(state.get("messages", []))
    print(f"[LOG] Model Call #{model_call_count} | Messages: {message_count}")
    return None


@after_model
def log_after_model(state, runtime):
    """Runs AFTER every LLM call; logs whether tool calls were requested."""
    last_message = (
        state.get("messages", [])[-1] if state.get("messages") else None
    )
    if last_message:
        has_tool_calls = (
            hasattr(last_message, "tool_calls") and last_message.tool_calls
        )
        print(f"[LOG] Tool Calls Requested: {has_tool_calls}")
    return None


call_limiter = ModelCallLimitMiddleware(
    thread_limit=25,
    run_limit=25,
    exit_behavior="end",
)

# =========================================================
# TOOLS AVAILABLE TO THE AGENT
# =========================================================

early_warning_tools = [
    search_financial_stress_knowledge,
    search_risk_news_articles,
]

# =========================================================
# AGENT LLM (LLM #1)
# =========================================================

early_warning_agent = create_agent(
    model=AGENT_MODEL,
    tools=early_warning_tools,
    system_prompt="""
You are an early-warning analyst for US investment bank risk surveillance.

IMPORTANT:
- You MUST use tools for all factual retrieval.
- Never invent headlines or citations.

--------------------------------------------------
REQUIRED PROCEDURE
--------------------------------------------------

1. Search the RAG knowledge base multiple times
   using queries related to:
   - systemic stress
   - banking liquidity
   - contagion
   - geopolitical transmission

2. Search news articles using these keywords:
   - US military escalation
   - oil price spike
   - bank liquidity crisis
   - cyberattack on banks

3. After retrieval:
   - summarize major themes
   - identify potential risks
   - provide analyst-style commentary
""",
    middleware=[
        log_before_model,
        log_after_model,
        call_limiter,
    ],
)

# =========================================================
# STRUCTURED OUTPUT SCHEMA
# =========================================================


class EarlyWarningRow(BaseModel):
    """One CSV row representing one possible risk signal."""

    early_warning: Literal["yes", "no"] = Field(
        description=(
            "Whether this item should be monitored "
            "as a potential early-warning signal."
        )
    )
    risk_category: RISK_CATEGORY_LITERAL = Field(
        description=(
            "Exactly one of the four bank monitoring themes (same strings as the "
            "news searches). For IMF PDF excerpts (file:// URL), infer the single "
            "best-fitting theme from the passage content — never use any other label."
        )
    )
    article_headlines: str = Field(description="Headline or short title.")
    source_country: str = Field(
        description="Country or region where risk originates."
    )
    potential_impact: Literal["high", "medium", "low"] = Field(
        description="Potential impact on US markets."
    )
    chunk: str = Field(
        description=(
            "Evidence text copied from the source: prefer several contiguous "
            "sentences (roughly 400–2500 characters). Do not compress to a single "
            "short sentence unless the source truly contains only that much."
        )
    )
    url: str = Field(description="Source URL or citation.")


class EarlyWarningAssessment(BaseModel):
    """Full structured output containing all rows."""

    rows: list[EarlyWarningRow]


# =========================================================
# HELPER: EXTRACT TOOL OUTPUTS
# =========================================================


def extract_tool_outputs(messages) -> str:
    """Extracts all tool outputs from LangChain messages."""
    parts = []
    for message in messages:
        if isinstance(message, ToolMessage) and message.content:
            parts.append(str(message.content))
    return "\n\n=====\n\n".join(parts)


# =========================================================
# FALLBACK: DIRECT TOOL EXECUTION
# =========================================================


def gather_evidence_direct() -> str:
    """Runs tools directly if the agent fails to produce tool outputs."""
    parts = []
    for query in RAG_QUERIES:
        result = search_financial_stress_knowledge.invoke({"query": query})
        parts.append(result)
    for keyword in NEWS_KEYWORDS:
        result = search_risk_news_articles.invoke({"keyword": keyword})
        parts.append(result)
    return "\n\n=====\n\n".join(parts)


# =========================================================
# STRUCTURED EXTRACTION (LLM #2)
# =========================================================


def extract_assessment(evidence_text: str) -> EarlyWarningAssessment:
    """Uses a second LLM to transform raw evidence into structured CSV rows."""
    max_chars = int(os.environ.get("EARLY_WARNING_MAX_EVIDENCE_CHARS", "120000"))
    if len(evidence_text) > max_chars:
        evidence_text = evidence_text[:max_chars] + "\n\n[TRUNCATED]\n"
    llm = ChatOpenAI(model=STRUCTURED_MODEL, temperature=0)
    structured_llm = llm.with_structured_output(EarlyWarningAssessment)
    prompt = f"""
You are converting financial evidence into a structured
early-warning monitoring table.

RULES:
- One Google News article = one row (never merge multiple articles).
- One IMF PDF [Source N] block = one row (never merge multiple PDF blocks).
- You MUST emit separate rows for IMF PDF evidence and for news evidence when both qualify:
  many rows may share the same risk_category but must have different url/chunk.
- Do not drop IMF/file:// rows just because news rows exist for the same theme.
- Only mark "yes" for meaningful risks
- Estimate impact level (high / medium / low) for US investment banks
- For field "chunk": copy a substantive excerpt from THAT item only — usually several
  sentences or a full paragraph from the IMF Content block or the news Summary,
  about 400–2500 characters when available. Avoid one-sentence summaries.
- Preserve URLs / file:// citations exactly from the evidence (field "url")

risk_category — MUST be exactly one of these four strings (no other spelling):
- "US military escalation"
- "oil price spike"
- "bank liquidity crisis"
- "cyberattack on banks"

How to choose risk_category:
- If the row is from a Google News item under a keyword block, use that keyword phrase.
- If the row is from an IMF PDF chunk (file:// citation in the evidence), pick the ONE theme
  the passage best supports, for example:
  • banking runs, liquidity stress, systemic stress tests, solvency → "bank liquidity crisis"
  • oil, energy shocks, commodity spikes → "oil price spike"
  • conflict, escalation, defense, geopolitical military tension → "US military escalation"
  • cyber, ransomware, operational outages at banks → "cyberattack on banks"
  If several fit, choose the dominant risk in the excerpt.

Evidence:
{evidence_text}
"""
    return structured_llm.invoke(prompt)


# =========================================================
# WRITE CSV FILE
# =========================================================


def write_csv(path: Path, assessment: EarlyWarningAssessment) -> int:
    """Writes ONLY rows marked as early_warning = yes.

    Set EARLY_WARNING_APPEND_CSV=1 to append rows (keeps history by scan day).
    Default overwrites the file for a single latest snapshot.
    """
    yes_rows = [row for row in assessment.rows if row.early_warning == "yes"]
    append = os.environ.get("EARLY_WARNING_APPEND_CSV", "").lower() in (
        "1",
        "true",
        "yes",
    )
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    mode = "a" if append and path.is_file() else "w"
    write_header = mode == "w" or not path.is_file()
    with path.open(mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_HEADERS)
        if write_header:
            writer.writeheader()
        chunk_cap = int(os.environ.get("EARLY_WARNING_CHUNK_CSV_MAX", "2500"))
        for row in yes_rows:
            if len(row.chunk) > chunk_cap:
                chunk = row.chunk[: chunk_cap - 3] + "..."
            else:
                chunk = row.chunk
            writer.writerow({
                "timestamp": run_ts,
                "risk_category": row.risk_category,
                "early warning": row.early_warning,
                "article headlines": row.article_headlines,
                "source country": row.source_country,
                "potential impact (high/medium/low)": row.potential_impact,
                "chunk": chunk,
                "URL": row.url,
            })
    return len(yes_rows)


# =========================================================
# MAIN PIPELINE
# =========================================================


def main():
    """Agent gathers evidence → structured extraction → CSV."""
    print("\nRunning Early Warning Agent...\n")
    user_instruction = """
Execute the complete early-warning monitoring workflow:
- run all required searches
- gather evidence
- summarize key risks
"""
    response = early_warning_agent.invoke({
        "messages": [{"role": "user", "content": user_instruction}],
    })
    evidence = extract_tool_outputs(response["messages"]).strip()
    if not evidence:
        print(
            "No tool outputs found."
            " Running direct retrieval fallback..."
        )
        evidence = gather_evidence_direct()
    print("\nStructuring risk signals...\n")
    assessment = extract_assessment(evidence)
    rows_written = write_csv(CSV_PATH, assessment)
    print(
        f"\nFinished."
        f"\nRows Written: {rows_written}"
        f"\nCSV Path: {CSV_PATH.resolve()}"
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
