"""
Streamlit UI: input guardrails → semantic cache → agent → output guardrails → optional human approval.

  streamlit run app.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv()

from agent import run_agent  # noqa: E402
from cache import DEFAULT_SIMILARITY_THRESHOLD, lookup_similar, remember  # noqa: E402
from guardrails import Recommendation, validate_input, validate_output  # noqa: E402

NEEDS_APPROVAL = {"BUY", "SELL"}


def main() -> None:
    st.set_page_config(page_title="Finance Assistant Demo", layout="wide")
    st.title("AI Financial Research Assistant (demo)")
    st.caption(
        "Guardrails · semantic cache · LangChain agent · structured outputs · "
        "**Not financial advice.**"
    )

    if not os.environ.get("OPENAI_API_KEY"):
        st.error("Set **OPENAI_API_KEY** in `.env` or your environment.")
        return

    with st.sidebar:
        st.markdown("### Pipeline options")
        human_gate = st.checkbox(
            "Require approval for BUY/SELL labels",
            value=True,
            help="Human-in-the-loop before showing trades-style labels.",
        )
        st.caption(
            f"Semantic cache threshold: **{DEFAULT_SIMILARITY_THRESHOLD:.2f}** "
            "(env `SEMANTIC_CACHE_MIN_SCORE`)"
        )

    default_q = "Analyze NVDA after strong AI chip demand headlines."
    query = st.text_area("Your question", value=default_q, height=100)
    run = st.button("Run pipeline", type="primary")

    if not run:
        st.info("Enter a question and click **Run pipeline**.")
        return

    steps = st.container()

    # --- Step 1: Input guardrails ---
    with steps.expander("1 · Input guardrails", expanded=True):
        ok_in, reason_in = validate_input(query)
        if not ok_in:
            st.error(f"**Blocked:** {reason_in}")
            return
        st.success("Input passed guardrails.")

    # --- Step 2: Semantic cache ---
    with steps.expander("2 · Semantic cache", expanded=True):
        t0 = time.perf_counter()
        hit, cached_rec, score = lookup_similar(query)
        dt_ms = (time.perf_counter() - t0) * 1000
        if hit and cached_rec is not None:
            st.success(
                f"**Cache hit** (similarity score **{score:.3f}**) — "
                f"latency **{dt_ms:.1f} ms** (no LLM call)."
            )
            rec = cached_rec
            used_cache = True
        else:
            st.warning(
                f"**Cache miss** (best score **{score:.3f}** vs threshold "
                f"{DEFAULT_SIMILARITY_THRESHOLD:.3f}) — calling agent."
            )
            used_cache = False

    # --- Step 3: Agent (only on miss) ---
    if not used_cache:
        with steps.expander("3 · LangChain agent + mock CSV", expanded=True):
            t1 = time.perf_counter()
            try:
                rec = run_agent(query)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Agent failed: {exc}")
                return
            llm_ms = (time.perf_counter() - t1) * 1000
            st.success(f"Agent finished in **{llm_ms:.0f} ms**.")

    # --- Output guardrails ---
    with steps.expander("4 · Output guardrails", expanded=True):
        ok_out, reason_out = validate_output(rec)
        if not ok_out:
            st.error(f"**Blocked:** {reason_out}")
            return
        st.success("Structured output passed compliance text checks.")

    # --- Persist only vetted responses (cache misses) ---
    if not used_cache:
        with steps.expander("5 · Semantic cache write", expanded=True):
            remember(query, rec)
            st.success("Stored this query/answer pair for future semantic reuse.")

    # --- Human approval ---
    with steps.expander("6 · Human approval (optional)", expanded=True):
        if human_gate and rec.suggested_action in NEEDS_APPROVAL:
            st.warning(
                f"Suggested action **`{rec.suggested_action}`** requires approval "
                "(classroom gate)."
            )
            approved = st.radio(
                "Approve displaying this recommendation?",
                ("Yes", "No"),
                horizontal=True,
                key="approve_radio",
                index=0,
            )
            if approved != "Yes":
                st.warning("Approval withheld — final JSON hidden.")
                st.stop()
        else:
            st.info("No BUY/SELL approval required for this response.")

    # --- Final ---
    st.subheader("Structured response")
    st.json(json.loads(rec.model_dump_json()))

    st.markdown(
        "---\n*Educational mock only. Mock CSV data; no trades executed; "
        "not personalized investment advice.*"
    )


if __name__ == "__main__":
    main()
