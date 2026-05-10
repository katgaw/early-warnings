"""
===========================================================
RAG KNOWLEDGE BASE FOR FINANCIAL STRESS ANALYSIS
===========================================================

This script:

1. Loads an IMF PDF report
2. Splits the text into chunks
3. Converts chunks into embeddings
4. Stores embeddings inside Qdrant
5. Creates a retriever for semantic search
6. Exposes a LangChain tool for agents

-----------------------------------------------------------
WHAT IS RAG?
-----------------------------------------------------------

RAG = Retrieval-Augmented Generation

Instead of relying only on LLM memory, we:
- retrieve relevant documents
- provide them to the LLM
- generate grounded answers

-----------------------------------------------------------
PIPELINE
-----------------------------------------------------------

PDF Report
    ↓
Document Loading
    ↓
Text Chunking
    ↓
Embeddings
    ↓
Qdrant Vector Database
    ↓
Retriever
    ↓
LangChain Tool

===========================================================
"""

# =========================================================
# IMPORTS
# =========================================================

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# =========================================================
# PDF FILE CONFIGURATION (relative to this package directory)
# =========================================================

_RAG_DIR = Path(__file__).resolve().parent
PDF_PATH = _RAG_DIR / "data" / "imf-report.pdf"
if not PDF_PATH.is_file():
    raise FileNotFoundError(
        "Missing IMF corpus PDF. Place imf-report.pdf at:\n"
        f"  {PDF_PATH}\n"
        "(Paths are resolved from rag_tool.py location, not the shell cwd.)"
    )

print("\nLoading IMF report...\n")

# =========================================================
# STEP 1: LOAD PDF DOCUMENTS
# =========================================================

loader = PyPDFLoader(str(PDF_PATH))
documents = loader.load()
print(f"Loaded {len(documents)} PDF pages")
total_characters = sum(len(doc.page_content) for doc in documents)
print(f"Total characters: {total_characters:,}")

# =========================================================
# STEP 2: SPLIT DOCUMENTS INTO CHUNKS
# =========================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1400,
    chunk_overlap=200,
)
chunk_docs = text_splitter.split_documents(documents)
print(f"\nCreated {len(chunk_docs)} chunks")
print("\nSample Chunk")
print("-" * 60)
print(chunk_docs[0].page_content[:300] + "...")

# =========================================================
# STEP 3–4: EMBEDDINGS AND VECTOR DIMENSION
# =========================================================

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
sample_embedding = embedding_model.embed_query("financial stress")
embedding_dimension = len(sample_embedding)
print(f"\nEmbedding Dimension: {embedding_dimension}")

# =========================================================
# STEP 5–7: QDRANT, VECTOR STORE, INDEX CHUNKS
# =========================================================

qdrant_client = QdrantClient(":memory:")
collection_name = "financial_stress_knowledge_base"
qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=embedding_dimension,
        distance=Distance.COSINE,
    ),
)
print(f"\nCreated Qdrant Collection:\n{collection_name}")

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=collection_name,
    embedding=embedding_model,
)
vector_store.add_documents(chunk_docs)
print(f"\nAdded {len(chunk_docs)} chunks to vector database")

# =========================================================
# STEP 8: RETRIEVER (k=5)
# =========================================================

retriever = vector_store.as_retriever(search_kwargs={"k": 6})
print("\nRetriever created")

# =========================================================
# LANGCHAIN TOOL
# =========================================================


@tool
def search_financial_stress_knowledge(query: str) -> str:
    """
    Search the financial stress knowledge base.

    Use this tool for:
    - banking crises
    - financial instability
    - liquidity risk
    - systemic risk
    - recession signals
    - economic vulnerability
    - market fragility

    Args:
        query: Search query related to financial risk

    Returns:
        Formatted retrieval results with citations
    """
    results = retriever.invoke(query)
    if not results:
        return "No relevant information found in the knowledge base."
    formatted_results = []
    for index, doc in enumerate(results, start=1):
        page = doc.metadata.get("page")
        source = doc.metadata.get("source") or str(PDF_PATH.resolve())
        location = f" (Page {page + 1})" if page is not None else ""
        page_fragment = f"#page={page + 1}" if page is not None else ""
        citation_url = f"file://{Path(source).resolve()}{page_fragment}"
        formatted_text = f"""
[Source {index}{location}]

Citation:
{citation_url}

Content:
{doc.page_content}
"""
        formatted_results.append(formatted_text.strip())
    return "\n\n".join(formatted_results)


# =========================================================
# FINAL LOGGING
# =========================================================

print("\nTool Successfully Created")
print(f"\nTool Name:\n{search_financial_stress_knowledge.name}")
print(f"\nTool Description:\n{search_financial_stress_knowledge.description[:200]}...")
print("\nRAG Knowledge Base Ready\n")
