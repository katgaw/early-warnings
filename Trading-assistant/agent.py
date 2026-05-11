"""LLM analysis with mock-news context + structured Recommendation (no legacy langchain.agents)."""

from __future__ import annotations

import os
import re
from typing import Any
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from guardrails import Recommendation

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "mock_news.csv"

load_dotenv(BASE_DIR / ".env")
load_dotenv()

MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

_df: pd.DataFrame | None = None


def _news_df() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = pd.read_csv(CSV_PATH)
    return _df


@tool
def search_mock_financial_news(query: str) -> str:
    """Search mock headlines by ticker (e.g. TSLA) or keyword in headline/risk columns."""
    q = (query or "").strip().upper()
    df = _news_df()
    if not q:
        return df.to_string(index=False)

    tickers = set(re.findall(r"\b[A-Z]{1,5}\b", query.upper()))
    rows = df[df["ticker"].str.upper().isin(tickers)] if tickers else df[
        df.apply(
            lambda r: q.lower() in str(r["headline"]).lower()
            or q.lower() in str(r["risk"]).lower()
            or q.lower() in str(r["ticker"]).lower(),
            axis=1,
        )
    ]
    if rows.empty:
        rows = df.head(5)
    return rows.to_string(index=False)


SYSTEM_PROMPT = """You are an educational mock equity research assistant.

Rules:
- Base sentiment and risks ONLY on the mock CSV rows provided below; do not invent live prices or filings.
- Never promise returns or use words like guaranteed, risk-free, or certain profit.
- suggested_action is a coarse classroom label (HOLD/BUY/SELL/RESEARCH_FURTHER), not real advice.
- confidence reflects how well the mock data supports your view (0–1).
- Pick the primary ticker that best matches the user's question.
"""


_llm_structured: Any = None


def _structured_llm() -> Any:
    global _llm_structured
    if _llm_structured is None:
        base = ChatOpenAI(model=MODEL_NAME, temperature=0)
        _llm_structured = base.with_structured_output(Recommendation)
    return _llm_structured


def run_agent(user_query: str) -> Recommendation:
    """Retrieve mock headlines, then produce a validated structured recommendation."""
    raw_q = user_query.strip()
    table = search_mock_financial_news.invoke({"query": raw_q})
    llm = _structured_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"User question:\n{raw_q}\n\n"
                f"Mock news rows (from search_mock_financial_news):\n{table}"
            )
        ),
    ]
    result = llm.invoke(messages)
    if isinstance(result, Recommendation):
        return result
    return Recommendation.model_validate(result)
