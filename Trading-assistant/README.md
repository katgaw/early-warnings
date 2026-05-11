# AI Financial Research Assistant with Guardrails

Classroom demo: LangChain agents, input/output guardrails, semantic caching (vector similarity), structured outputs (Pydantic), mock financial data, and Streamlit.

**Educational only.** This is **not** a trading system and **does not** provide real financial advice.

---

## Step 0 — Read this first (2 minutes)

1. **Goal:** See how a finance-style assistant can layer **safety checks** and **caching** around an LLM agent.
2. **You will:** Install dependencies, set `OPENAI_API_KEY`, run Streamlit, try a normal question, then try blocked prompts and cache behavior.
3. **Mock data:** Analysis uses **`mock_news.csv`** (not live markets).

---

## Step 1 — What the app does (pipeline)

Follow one user request through the system **in order**:

| Step | Stage | What happens |
|------|--------|----------------|
| 1 | **You** | Type a question in Streamlit (e.g. “Analyze Tesla after earnings.”). |
| 2 | **Input guardrails** | Text is checked for injection / unsafe phrases; bad requests are **blocked** before the LLM runs. |
| 3 | **Semantic cache** | Query is embedded and matched against past queries; if **similar enough**, a cached answer is returned (skip LLM). |
| 4 | **LangChain agent** | If not cached, the agent runs with tools / context from the **mock news** dataset. |
| 5 | **LLM analysis** | Model produces analysis (sentiment, risks, summary, etc.). |
| 6 | **Output guardrails** | Response is validated (structure, forbidden wording like “guaranteed return”). |
| 7 | **Human approval** (if enabled in your build) | Final action may require **Y/N** before showing “BUY/SELL” style text. |
| 8 | **Response** | Structured result shown in the UI. |

**ASCII flow**

```text
User → Input guardrails → Semantic cache check → LangChain agent → Mock CSV DB
        → LLM → Output validation → Human approval (optional) → Final response
```

---

## Step 2 — Project layout

Expected files (adjust names if your repo differs):

```text
project/
├── app.py              # Streamlit entrypoint
├── agent.py            # LangChain agent wiring
├── guardrails.py       # Input / output checks
├── cache.py            # Semantic cache (embeddings + JSON file, cosine similarity)
├── mock_news.csv       # Mock headlines & labels
├── requirements.txt
└── README.md
```

---

## Step 3 — Mock dataset

Columns typically include **`ticker`**, **`headline`**, **`sentiment`**, **`risk`**.

| ticker | headline | sentiment | risk |
|--------|----------|-----------|------|
| TSLA | Tesla beats delivery expectations | Bullish | Margin pressure |
| AAPL | Apple services revenue slows | Neutral | China demand |
| NVDA | Nvidia AI chip demand surges | Bullish | Valuation risk |
| JPM | JPMorgan reports strong earnings | Bullish | Credit exposure |

---

## Step 4 — Install and run (commands)

### Step 4a — Clone and enter the project

```bash
git clone <repo-url>
cd project
```

### Step 4b — Create and activate a virtual environment

```bash
python -m venv venv
```

Activate:

- **Windows:** `venv\Scripts\activate`
- **macOS / Linux:** `source venv/bin/activate`

### Step 4c — Install Python packages

```bash
pip install -r requirements.txt
```

Example `requirements.txt` (your course version may pin versions):

```text
python-dotenv
langchain-core
langchain-openai
streamlit
pandas
pydantic
numpy
```

Optional: **LangSmith** for tracing (`langsmith`).

### Step 4d — Environment variables

Create a **`.env`** file in the project root:

```bash
OPENAI_API_KEY=your_api_key
```

### Step 4e — Start the UI

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

---

## Step 5 — Try it in the UI (hands-on sequence)

Do these **in order** so each concept clicks:

1. **Normal query** — Example: `Analyze Nvidia after earnings.`  
   - Expect: sentiment, confidence, risks, short summary (exact fields depend on your schema).

2. **Unsafe / compliance prompt** — Example: `Give me guaranteed profitable trades.`  
   - Expect: message that **input or output guardrails** blocked the request.

3. **Semantic cache** — Run a query, then a **paraphrase**:  
   - First: `Analyze Tesla earnings.`  
   - Second: `Tesla earnings sentiment?`  
   - Expect: second answer **much faster** if cache hits (timing shown in slides/demo notes).

4. **Human-in-the-loop** (if implemented) — When the UI asks **Approve recommendation? [Y/N]**, try both paths.

---

## Step 6 — Guardrails (what to teach)

### Input guardrails

- Block **prompt injection**, **illegal / irresponsible finance asks**, and **prohibited wording**.
- Example blocked themes: “guaranteed profit”, “insider trading”, “risk-free”.

### Output guardrails

- Enforce **schema** (Pydantic), block **guaranteed returns**, require sensible **risk disclosure** language.

Example forbidden output phrases (conceptual):

```python
FORBIDDEN_OUTPUTS = [
    "certain profit",
    "guaranteed return",
]
```

---

## Step 7 — Structured outputs (Pydantic)

Responses should match a strict model, e.g.:

```python
class Recommendation(BaseModel):
    ticker: str
    sentiment: str
    confidence: float
    risks: list[str]
    summary: str
```

Example JSON-shaped result:

```json
{
  "ticker": "TSLA",
  "sentiment": "Bullish",
  "confidence": 0.81,
  "risks": ["Margin pressure", "Demand volatility"],
  "summary": "Strong earnings growth offset by weaker margins."
}
```

---

## Step 8 — Semantic caching (short theory)

1. Embed the user query.
2. Compare embeddings to **prior queries** (local JSON cache + cosine similarity).
3. If similarity passes a threshold, **reuse** the stored answer instead of calling the LLM again.

**Why it matters:** lower cost, lower latency, good classroom analogy for **production optimization** (with caveats: stale answers, threshold tuning).

---

## Step 9 — Tech stack reference

| Component | Technology |
|-----------|------------|
| Agent | LangChain |
| UI | Streamlit |
| Validation | Pydantic |
| Vector cache | OpenAI embeddings + JSON file (`data/semantic_cache.json`) |
| Embeddings | OpenAI embeddings |
| Data | Pandas + CSV |
| Optional observability | LangSmith |

---

## Step 10 — Suggested classroom demos

1. Show a **clean** financial analysis on mock data.
2. Demonstrate a **prompt injection** or “ignore rules” style attack → **blocked**.
3. Show **validation failure** when output breaks the schema or uses forbidden phrases.
4. Show **cache hit** latency vs **cold** LLM call.
5. Walk through **human approval** before any “action” text.

---

## Step 11 — Extensions (homework / capstone ideas)

- Swap mock CSV for a **read-only** news API (with keys and rate limits).
- Add **LangGraph** for branching workflows.
- Stronger **audit logs**, roles, and retention policies for prompts/responses.

---

## Disclaimer

- **Educational / simulated / non-production.**
- Does **not** execute trades, does **not** provide personalized investment advice, does **not** guarantee outcomes.
- Mock data is for demonstration only.

---

## Key takeaway

LLMs are powerful but unsafe by default for regulated domains. **Guardrails**, **structured validation**, **semantic caching**, **monitoring**, and **human oversight** are what move a demo toward **enterprise-style** AI design—still not automatic compliance, but the right vocabulary for class.
