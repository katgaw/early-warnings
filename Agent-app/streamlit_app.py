"""
Streamlit dashboard for early-warning CSV output from app.py.

Run from this directory:
  streamlit run streamlit_app.py

On first load, if the CSV is missing or has no rows, this app runs
`python app.py` once automatically (same as the CLI pipeline).

Tip: set EARLY_WARNING_APPEND_CSV=1 before runs so history accumulates by day.
"""

from __future__ import annotations

import html
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")
load_dotenv()

CSV_PATH = APP_DIR / "early_warning_signals.csv"
LOGO_PATH = APP_DIR / "data" / "image.png"

ALLOWED_RISK_CATEGORIES = frozenset({
    "US military escalation",
    "oil price spike",
    "bank liquidity crisis",
    "cyberattack on banks",
})

RISK_CATEGORY_ORDER = (
    "US military escalation",
    "oil price spike",
    "bank liquidity crisis",
    "cyberattack on banks",
)


def _category_sort_key(cat: str) -> int:
    try:
        return RISK_CATEGORY_ORDER.index(cat)
    except ValueError:
        return len(RISK_CATEGORY_ORDER)


def _max_impact_level(levels: pd.Series) -> str:
    rank = {"high": 3, "medium": 2, "low": 1}
    best = "low"
    best_r = 0
    for raw in levels.dropna().astype(str):
        L = raw.strip().lower()
        r = rank.get(L, 0)
        if r > best_r:
            best_r = r
            best = L if r > 0 else best
    return best

CSV_COLUMNS = [
    "timestamp",
    "risk_category",
    "early warning",
    "article headlines",
    "source country",
    "potential impact (high/medium/low)",
    "chunk",
    "URL",
]

IMPACT_STYLES = {
    "high": ("#ffcdd2", "#b71c1c", "#b71c1c"),
    "medium": ("#ffe082", "#f57f17", "#f57f17"),
    "low": ("#c8e6c9", "#1b5e20", "#2e7d32"),
}


def _csv_has_signal_rows(path: Path) -> bool:
    """True if CSV exists and has at least one data row."""
    if not path.is_file():
        return False
    try:
        df = pd.read_csv(path, encoding="utf-8")
        return len(df) > 0
    except Exception:
        return False


def run_risk_scan() -> subprocess.CompletedProcess:
    """Run the same pipeline as `python app.py` (inherits env including .env above)."""
    return subprocess.run(
        [sys.executable, str(APP_DIR / "app.py")],
        cwd=str(APP_DIR),
        capture_output=True,
        text=True,
        timeout=3600,
        env=None,
    )


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure expected columns exist (supports legacy CSV files)."""
    out = df.copy()
    for col in CSV_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    return out[CSV_COLUMNS]


@st.cache_data(ttl=15)
def load_signals(path_str: str, _mtime: float) -> pd.DataFrame:
    path = Path(path_str)
    if not path.is_file():
        return pd.DataFrame(columns=CSV_COLUMNS)
    df = pd.read_csv(path, encoding="utf-8")
    return _normalize_dataframe(df)


def risk_oval_markdown(level: str) -> str:
    raw = (level or "").strip().lower()
    bg, fg, border = IMPACT_STYLES.get(raw, ("#eceff1", "#455a64", "#455a64"))
    label = raw.upper() if raw else "N/A"
    return (
        f'<div style="display:flex;align-items:center;justify-content:flex-end;'
        f'min-height:3rem;">'
        f'<span style="background:{bg};color:{fg};border:2px solid {border};'
        f'padding:0.4rem 1.1rem;border-radius:999px;font-weight:700;'
        f'font-size:0.8rem;letter-spacing:0.04em;">{html.escape(label)}</span>'
        f"</div>"
    )


def format_ts(ts_val) -> str:
    if pd.isna(ts_val) or str(ts_val).strip() == "":
        return "—"
    try:
        dt = pd.to_datetime(ts_val, utc=True, errors="coerce")
        if pd.isna(dt):
            return str(ts_val)
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(ts_val)


def main() -> None:
    st.set_page_config(
        page_title="Investment Bank Risk Monitor",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    BOOTSTRAP_KEY = "_streamlit_pipeline_bootstrapped"
    if BOOTSTRAP_KEY not in st.session_state:
        st.session_state[BOOTSTRAP_KEY] = False

    if not st.session_state[BOOTSTRAP_KEY]:
        st.session_state[BOOTSTRAP_KEY] = True
        if not _csv_has_signal_rows(CSV_PATH):
            with st.spinner(
                "Running initial pipeline (**python app.py**) — first load only…"
            ):
                proc = run_risk_scan()
            load_signals.clear()
            st.session_state["_last_scan_returncode"] = proc.returncode
            st.session_state["_last_scan_log"] = (
                (proc.stderr or "").strip() + "\n" + (proc.stdout or "").strip()
            ).strip()
            st.rerun()

    logo_col, title_col = st.columns([1, 6])
    with logo_col:
        if LOGO_PATH.is_file():
            st.image(str(LOGO_PATH), width=110)
        else:
            st.caption("Add **data/image.png** for logo.")
    with title_col:
        st.markdown(
            "# Investment Bank Risk Monitor",
            help="Early-warning signals from RAG + news pipeline (CSV)",
        )

    st.divider()

    c1, c2 = st.columns([3, 1])
    with c1:
        st.caption(
            f"Reading **`{CSV_PATH.name}`** · `.env` is loaded for this process "
            f"(and child **`app.py`** runs). Append history: "
            "`EARLY_WARNING_APPEND_CSV=1`."
        )
    with c2:
        if st.button("Run new scan", help="Executes app.py in this folder"):
            with st.spinner("Running scan…"):
                proc = run_risk_scan()
            load_signals.clear()
            st.session_state["_last_scan_returncode"] = proc.returncode
            st.session_state["_last_scan_log"] = (
                (proc.stderr or "").strip() + "\n" + (proc.stdout or "").strip()
            ).strip()
            if proc.returncode != 0:
                st.error("Scan exited with an error — expand details below.")
            else:
                st.success("Scan finished.")
            st.rerun()

    last_rc = st.session_state.get("_last_scan_returncode")
    last_log = st.session_state.get("_last_scan_log", "")
    if last_rc is not None and last_rc != 0 and last_log:
        with st.expander("Last scan stderr / stdout (debug)"):
            st.code(last_log[-12000:] or "(empty)", language="text")

    mtime = CSV_PATH.stat().st_mtime if CSV_PATH.is_file() else 0.0
    df = load_signals(str(CSV_PATH), mtime)

    if df.empty:
        if last_rc is not None and last_rc != 0:
            st.error(
                "The automatic or last manual scan **failed**. "
                "Check **OPENAI_API_KEY**, **`data/imf-report.pdf`**, and the log above."
            )
        elif last_rc == 0:
            st.warning(
                "Scan completed, but **no early-warning rows** were written "
                "(everything may have been classified as not a signal). "
                "Try **Run new scan** or adjust prompts/data."
            )
        else:
            st.info("No CSV rows yet.")
        return

    df = df.copy()
    df["_dt"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["_day"] = df["_dt"].dt.date
    missing_day = df["_day"].isna()
    if missing_day.any():
        today = datetime.now(timezone.utc).date()
        df.loc[missing_day, "_day"] = today

    for day in sorted(df["_day"].unique(), reverse=True):
        day_rows = df[df["_day"] == day].sort_values(
            ["risk_category", "timestamp"],
            ascending=[True, False],
        )
        n_cat = day_rows["risk_category"].nunique()
        st.subheader(
            f"📅 {day.isoformat()} · {len(day_rows)} source row(s) · "
            f"{n_cat} risk categor{'y' if n_cat == 1 else 'ies'}"
        )

        cats_today = sorted(
            day_rows["risk_category"].dropna().unique().tolist(),
            key=_category_sort_key,
        )

        for cat in cats_today:
            grp = day_rows[day_rows["risk_category"] == cat]
            grp_impact = _max_impact_level(
                grp["potential impact (high/medium/low)"]
            )
            ts_latest = format_ts(grp["timestamp"].max())
            n_src = len(grp)

            hl0 = str(grp.iloc[0].get("article headlines", "") or "").strip()
            title_tail = (
                hl0[:90] + "…"
                if len(hl0) > 90
                else (hl0 or "(see sources below)")
            )
            if n_src > 1:
                title_tail = f"{n_src} sources — {title_tail}"

            box_col, oval_col = st.columns([5, 1])
            with box_col:
                exp_title = f"**{cat}** — {title_tail} — _{ts_latest}_"
                if cat not in ALLOWED_RISK_CATEGORIES:
                    exp_title = (
                        f"**{cat}** — {title_tail} — _{ts_latest}_ "
                        "_(rerun scan: use one of the four themes)_"
                    )
                with st.expander(exp_title):
                    st.markdown(f"**Risk category:** `{cat}`")
                    st.caption(
                        f"{n_src} separate citation(s) in this category "
                        "(news URLs and IMF PDF passages each get their own row "
                        "from the scan)."
                    )
                    if cat not in ALLOWED_RISK_CATEGORIES:
                        st.caption(
                            "Run **Run new scan** so rows map to one of the "
                            "four standard themes."
                        )

                    for i, (_, row) in enumerate(grp.iterrows(), start=1):
                        url = str(row.get("URL", "") or "").strip()
                        is_news = url.startswith("http")
                        src_kind = (
                            "News article"
                            if is_news
                            else "IMF report (PDF) — data/imf-report.pdf"
                        )
                        st.markdown(f"##### Source {i} · {src_kind}")
                        headline = str(
                            row.get("article headlines", "") or "(no headline)"
                        )
                        st.markdown(f"**Headline / label:** {headline}")
                        if is_news:
                            st.markdown(f"**URL:** [{url}]({url})")
                        else:
                            st.markdown(f"**Citation:** `{url}`")
                        st.markdown(
                            "**Source country:** "
                            f"{row.get('source country', '') or '—'}"
                        )
                        st.markdown(
                            "**Impact (this source):** "
                            f"`{row.get('potential impact (high/medium/low)', '') or '—'}` · "
                            f"_scan time {format_ts(row.get('timestamp'))}_"
                        )
                        st.markdown("**Evidence chunk**")
                        st.text_area(
                            f"chunk_{i}",
                            value=str(row.get("chunk", "") or ""),
                            height=min(320, 120 + len(str(row.get('chunk', ''))) // 4),
                            disabled=True,
                            label_visibility="collapsed",
                        )
                        if i < n_src:
                            st.divider()

            with oval_col:
                st.markdown(
                    risk_oval_markdown(grp_impact),
                    unsafe_allow_html=True,
                )
                if n_src > 1:
                    st.caption("Oval = **highest** impact among sources in this group.")


if __name__ == "__main__":
    main()
