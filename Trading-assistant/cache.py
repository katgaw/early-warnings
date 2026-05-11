"""Semantic cache: OpenAI embeddings + cosine similarity (JSON file, no ChromaDB)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from langchain_openai import OpenAIEmbeddings

from guardrails import Recommendation

BASE_DIR = Path(__file__).resolve().parent
CACHE_PATH = BASE_DIR / "data" / "semantic_cache.json"

_embeddings_singleton: OpenAIEmbeddings | None = None
_entries_singleton: list[dict[str, Any]] | None = None

# Minimum cosine similarity between query and cached query (same rough scale as typical semantic matching).
DEFAULT_SIMILARITY_THRESHOLD = float(
    os.environ.get("SEMANTIC_CACHE_MIN_SCORE", "0.82"),
)


def _get_embeddings() -> OpenAIEmbeddings:
    global _embeddings_singleton
    if _embeddings_singleton is None:
        _embeddings_singleton = OpenAIEmbeddings(model="text-embedding-3-small")
    return _embeddings_singleton


def _load_from_disk() -> list[dict[str, Any]]:
    if not CACHE_PATH.is_file():
        return []
    try:
        raw = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        entries = raw.get("entries")
        return entries if isinstance(entries, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_to_disk(entries: list[dict[str, Any]]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps({"entries": entries}, indent=2),
        encoding="utf-8",
    )


def _get_entries() -> list[dict[str, Any]]:
    global _entries_singleton
    if _entries_singleton is None:
        _entries_singleton = _load_from_disk()
    return _entries_singleton


def _cosine(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def lookup_similar(
    query: str,
    *,
    threshold: float | None = None,
) -> tuple[bool, Recommendation | None, float]:
    """
    Return (cache_hit, recommendation_or_none, score).

    Score is cosine similarity of the best cached query vs this query; 0.0 when miss.
    """
    th = DEFAULT_SIMILARITY_THRESHOLD if threshold is None else threshold
    q = query.strip()
    if not q:
        return False, None, 0.0

    entries = _get_entries()
    if not entries:
        return False, None, 0.0

    emb = _get_embeddings().embed_query(q)
    best_score = -1.0
    best_payload: str | None = None
    for row in entries:
        cached_emb = row.get("embedding")
        payload = row.get("payload")
        if not isinstance(cached_emb, list) or not payload:
            continue
        sim = _cosine(emb, cached_emb)
        if sim > best_score:
            best_score = sim
            best_payload = str(payload)

    if best_payload is None or best_score < th:
        return False, None, float(max(best_score, 0.0))

    try:
        data: dict[str, Any] = json.loads(best_payload)
        return True, Recommendation.model_validate(data), float(best_score)
    except Exception:
        return False, None, float(best_score)


def remember(query: str, rec: Recommendation) -> None:
    """Store a successful structured response for future semantic reuse."""
    q = query.strip()
    if not q:
        return
    emb = _get_embeddings().embed_query(q)
    payload = rec.model_dump_json()
    entries = _get_entries()
    entries.append({
        "query": q,
        "embedding": emb,
        "payload": payload,
    })
    _save_to_disk(entries)
