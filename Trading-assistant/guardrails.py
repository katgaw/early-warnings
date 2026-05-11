"""Input and output guardrails for the classroom finance assistant demo."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

FORBIDDEN_INPUT_PHRASES = (
    "guaranteed profit",
    "guaranteed return",
    "risk-free",
    "risk free",
    "insider trading",
    "ignore previous",
    "ignore all previous",
    "disregard instructions",
    "jailbreak",
    "system prompt",
)

PROMPT_INJECTION_PATTERNS = (
    r"ignore\s+(the\s+)?(above|prior|previous)",
    r"you\s+are\s+now\s+",
    r"new\s+instructions\s*:",
)

FORBIDDEN_OUTPUT_PHRASES = (
    "guaranteed return",
    "guaranteed profit",
    "risk-free return",
    "certain profit",
    "cannot lose",
    "sure thing",
)


class Recommendation(BaseModel):
    """Structured analyst-style response (educational mock)."""

    ticker: str = Field(description="Primary equity ticker discussed, e.g. NVDA.")
    sentiment: Literal["Bullish", "Bearish", "Neutral"] = Field()
    confidence: float = Field(ge=0.0, le=1.0)
    risks: list[str] = Field(min_length=1, max_length=8)
    summary: str = Field(max_length=2500)
    suggested_action: Literal["HOLD", "BUY", "SELL", "RESEARCH_FURTHER"] = Field(
        description="Educational label only; not real advice."
    )


def validate_input(text: str) -> tuple[bool, str]:
    """
    Return (ok, reason). If ok is False, block before LLM / cache write.
    """
    raw = (text or "").strip()
    if len(raw) < 3:
        return False, "Question too short."
    if len(raw) > 4000:
        return False, "Question too long (max ~4000 characters)."

    lower = raw.lower()
    for phrase in FORBIDDEN_INPUT_PHRASES:
        if phrase in lower:
            return False, f"Blocked phrase detected in input: {phrase!r}"

    for pat in PROMPT_INJECTION_PATTERNS:
        if re.search(pat, lower, re.IGNORECASE):
            return False, "Possible prompt-injection pattern detected."

    return True, ""


def validate_output(rec: Recommendation) -> tuple[bool, str]:
    """Block disallowed wording in model-generated text."""
    blob = f"{rec.summary} {' '.join(rec.risks)}".lower()
    for phrase in FORBIDDEN_OUTPUT_PHRASES:
        if phrase in blob:
            return False, f"Output blocked: contains forbidden wording {phrase!r}"
    return True, ""
