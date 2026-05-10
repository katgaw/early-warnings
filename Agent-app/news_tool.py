"""
Historical analog impact: cosine retrieval against rag_tool's historical index,
then a small LLM summarizes likely market/macro impacts.

Exports:
  assess_impact_for_signal — LangChain tool (for agent demos; pass chunk/headline fields).
  assess_impact_for_csv_row — same logic using a full CSV row dict (app.py enrichment).
"""

from __future__ import annotations

import os

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from rag_tool import historical_vector_store

IMPACT_MODEL = os.environ.get("EARLY_WARNING_STRUCT_MODEL", "gpt-4o-mini")
TOP_K = 3


def build_query_text(row: dict[str, str]) -> str:
    """Text sent through the embedding model (chunk preferred, else headline)."""
    return (row.get("chunk") or row.get("article headlines") or "").strip()


def retrieve_analogs(
    query: str,
    top_k: int = TOP_K,
    *,
    k: int | None = None,
) -> list[tuple[Document, float]]:
    """Optional keyword k is an alias for top_k (LangChain-style)."""
    n = k if k is not None else top_k
    return historical_vector_store.similarity_search_with_score(query, k=n)


def build_llm_context(query: str, results: list[tuple[Document, float]]) -> str:
    lines = ["CURRENT EVENT:", query, "HISTORICAL ANALOGS:"]
    for i, (doc, score) in enumerate(results, start=1):
        meta = doc.metadata
        lines.append(
            f"Analog {i} | {meta.get('historical_event', '')} | "
            f"distance={score:.4f}\n"
            f"  Transmission: {meta.get('transmission_mechanism', '')}\n"
            f"  Market: {meta.get('market_impact', '')}\n"
            f"  Economic: {meta.get('economic_impact', '')}"
        )
    lines.append(
        "LLM TASK: From these analogs and mechanisms, outline likely market "
        "and macro impacts for the current event."
    )
    return "\n".join(lines)


def format_analog_source_footer(results: list[tuple[Document, float]]) -> str:
    """Maps Analog 1..N to historical row title + URL so readers need not guess."""
    lines = ["---", "**Analog sources** (same order as Analog 1, 2, … above):"]
    for i, (doc, _) in enumerate(results, start=1):
        meta = doc.metadata or {}
        title = (meta.get("historical_event") or "").strip() or "(unnamed historical row)"
        url = (meta.get("url") or "").strip()
        if url:
            lines.append(f"- **Analog {i}:** {title} — [{url}]({url})")
        else:
            lines.append(f"- **Analog {i}:** {title}")
    return "\n".join(lines)


def _run_impact_assessment(row: dict[str, str], top_k: int) -> str:
    query = build_query_text(row)
    if not query:
        return ""
    results = retrieve_analogs(query, top_k)
    if not results:
        return "No historical analogs retrieved."
    context = build_llm_context(query, results)
    llm = ChatOpenAI(model=IMPACT_MODEL, temperature=0)
    prompt = (
        "You are a US investment bank early-warning analyst.\n"
        "Use ONLY the historical analogs and mechanisms below. "
        "Summarize the most likely market and macroeconomic impacts if the current "
        "situation parallels those episodes.\n"
        "Write 5–8 concise bullet points. Whenever you compare to an analog, name the "
        "**full historical event title** shown next to that analog (do not rely on "
        "readers knowing what 'Analog 1' means by number alone).\n\n"
        f"{context}"
    )
    message = llm.invoke(prompt)
    body = (message.content or "").strip()
    footer = format_analog_source_footer(results)
    return f"{body}\n\n{footer}".strip()


@tool
def assess_impact_for_signal(
    chunk: str,
    article_headlines: str = "",
    risk_category: str = "",
    source_url: str = "",
    top_k: int = TOP_K,
) -> str:
    """
    Embed the surveillance text, find the closest historical shocks in Qdrant (cosine),
    then return bullet-point impact analysis grounded in those analogs.

    Prefer a substantive **chunk** (evidence passage). If empty, **article_headlines** is used.
    Optional **risk_category** and **source_url** are only for context in debugging;
    search uses chunk/headline text only.
    """
    row: dict[str, str] = {
        "chunk": chunk.strip(),
        "article headlines": (article_headlines or "").strip(),
        "risk_category": (risk_category or "").strip(),
        "URL": (source_url or "").strip(),
    }
    k = max(1, min(int(top_k), 25))
    return _run_impact_assessment(row, k)


def assess_impact_for_csv_row(row: dict[str, str], top_k: int = TOP_K) -> str:
    """Same as the tool, but accepts one full CSV row dict (used after writing the CSV)."""
    return _run_impact_assessment(row, top_k)


if __name__ == "__main__":
    print(
        assess_impact_for_signal.invoke({
            "chunk": "",
            "article_headlines": (
                "Oil prices surge after military tensions in the Middle East "
                "increase inflation concerns."
            ),
            "top_k": 3,
        })
    )
