"""
IMF PDF RAG + historical-events index (shared Qdrant under data/qdrant/).

Exports:
  search_financial_stress_knowledge — LangChain tool for the agent
  historical_vector_store — used by news_tool for cosine similarity

If ./data/qdrant already contains points for a collection, that collection is reused:
no PDF reload/split, no CSV→document build, and no re-embedding into Qdrant for that index.
"""

from __future__ import annotations

import csv
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "imf-report.pdf"
HISTORICAL_CSV = BASE_DIR / "data" / "historical-events.csv"
QDRANT_PATH = BASE_DIR / "data" / "qdrant"

IMF_COLLECTION = "imf_knowledge_base"
HISTORICAL_COLLECTION = "historical_event_analogs"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "text-embedding-3-small"

load_dotenv(BASE_DIR / ".env")
load_dotenv()

QDRANT_PATH.mkdir(parents=True, exist_ok=True)
_qdrant_client = QdrantClient(path=str(QDRANT_PATH))
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
_embed_dim = len(embeddings.embed_query("financial stress"))
print(f"Embedding dim: {_embed_dim}")


def _ensure_collection(name: str) -> None:
    try:
        _qdrant_client.get_collection(name)
    except Exception:
        _qdrant_client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=_embed_dim, distance=Distance.COSINE),
        )


def _points(name: str) -> int:
    try:
        return int(_qdrant_client.get_collection(name).points_count)
    except Exception:
        return 0


_ensure_collection(IMF_COLLECTION)
_ensure_collection(HISTORICAL_COLLECTION)

vector_store = QdrantVectorStore(
    client=_qdrant_client,
    collection_name=IMF_COLLECTION,
    embedding=embeddings,
)

if _points(IMF_COLLECTION) == 0:
    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"Missing PDF (required for first-time IMF index): {PDF_PATH}")
    print("\nLoading IMF PDF (first-time index)…")
    loader = PyPDFLoader(str(PDF_PATH))
    pdf_pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    imf_chunks = splitter.split_documents(pdf_pages)
    print(f"Indexing {len(imf_chunks)} IMF chunks into Qdrant…")
    vector_store.add_documents(imf_chunks)
    print("IMF index ready.")
else:
    print(
        f"Reusing IMF Qdrant index ({_points(IMF_COLLECTION)} points); "
        "skipped PDF load, split, and re-index."
    )

retriever = vector_store.as_retriever(search_kwargs={"k": 5})


def _strip(row: dict[str, str]) -> dict[str, str]:
    return {(k or "").strip(): (v or "").strip() for k, v in row.items()}


def _historical_documents() -> list[Document]:
    docs: list[Document] = []
    with HISTORICAL_CSV.open(newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            row = _strip(raw)
            if not any(row.values()):
                continue
            event = row.get("Historical Event", "")
            shock = row.get("Shock/Event Description", "")
            quant = row.get("Quantitative Relationship Extracted", "")
            market = row.get("Market Impact", "")
            economic = row.get("Economic Impact", "")
            transmission = row.get("Transmission Mechanism", "")
            url = row.get("URL", "")
            display = "\n".join(
                [
                    f"Historical Event: {event}",
                    f"Headline: {row.get('Headline', '')}",
                    f"URL: {url}",
                    f"Shock/Event Description: {shock}",
                    f"Quantitative Relationship Extracted: {quant}",
                    f"Market Impact: {market}",
                    f"Economic Impact: {economic}",
                    f"Transmission Mechanism: {transmission}",
                    f"Estimated Confidence: {row.get('Estimated Confidence', '')}",
                ]
            )
            embed_text = "\n".join(
                [
                    f"Economic Impact: {economic}",
                    f"Market Impact: {market}",
                    f"Transmission Mechanism: {transmission}",
                    f"Quantitative Relationship Extracted: {quant}",
                    f"Shock / event context: {shock}",
                ]
            )
            docs.append(
                Document(
                    page_content=embed_text,
                    metadata={
                        "historical_event": event,
                        "url": url,
                        "transmission_mechanism": transmission,
                        "market_impact": market,
                        "economic_impact": economic,
                        "display_content": display,
                    },
                )
            )
    return docs


historical_vector_store = QdrantVectorStore(
    client=_qdrant_client,
    collection_name=HISTORICAL_COLLECTION,
    embedding=embeddings,
)

if _points(HISTORICAL_COLLECTION) == 0:
    if not HISTORICAL_CSV.is_file():
        raise FileNotFoundError(
            f"Missing historical CSV (required for first-time historical index): {HISTORICAL_CSV}"
        )
    hist_docs = _historical_documents()
    print(f"Indexing {len(hist_docs)} historical rows into Qdrant…")
    historical_vector_store.add_documents(hist_docs)
    print("Historical index ready.")
else:
    print(
        f"Reusing historical Qdrant index ({_points(HISTORICAL_COLLECTION)} points); "
        "skipped CSV document build and re-index."
    )

print("RAG + historical index ready.\n")


@tool
def search_financial_stress_knowledge(query: str) -> str:
    """Search the IMF financial-stress knowledge base (PDF chunks)."""
    results = retriever.invoke(query)
    if not results:
        return "No relevant information found."
    blocks = []
    for index, doc in enumerate(results, start=1):
        page = doc.metadata.get("page")
        src = doc.metadata.get("source") or str(PDF_PATH.resolve())
        loc = ""
        cite = f"file://{Path(src).resolve()}"
        if page is not None:
            try:
                pn = int(page) + 1
                loc = f" (page {pn})"
                cite += f"#page={pn}"
            except (TypeError, ValueError):
                pass
        blocks.append(
            f"[Source {index}{loc}]\nCitation: {cite}\n\n{doc.page_content.strip()}"
        )
    return "\n\n".join(blocks)
