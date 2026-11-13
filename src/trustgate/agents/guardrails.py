"""Prompt injection sanitization and zero-raw-cell privacy guardrails."""

import re
from typing import Any

PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+(instructions|rules)", re.IGNORECASE),
    re.compile(r"system\s*(prompt|message)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an)?", re.IGNORECASE),
    re.compile(r"dan\s+mode", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"override\s+(all\s+)?(rules|decision)", re.IGNORECASE),
    re.compile(r"developer\s+(override|mode)", re.IGNORECASE),
    re.compile(r"human\s+evaluator", re.IGNORECASE),
    re.compile(r"new\s+instructions\s*:", re.IGNORECASE),
]


class GuardrailsEngine:
    """Sanitizes text inputs and guarantees zero raw cell values reach the LLM."""

    @staticmethod
    def detect_prompt_injection(text: str) -> bool:
        """Scan candidate string (e.g. column header, user note) for prompt injection patterns."""
        if not text or not isinstance(text, str):
            return False
        for pat in PROMPT_INJECTION_PATTERNS:
            if pat.search(text):
                return True
        return False

    @staticmethod
    def sanitize_for_prompt(text: str) -> str:
        """Sanitize string by stripping control characters and injection tokens."""
        if not text or not isinstance(text, str):
            return ""
        # Normalize newlines and strip ASCII control characters except newline and tab
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)
        for pat in PROMPT_INJECTION_PATTERNS:
            cleaned = pat.sub("[REDACTED_INJECTION_PATTERN]", cleaned)
        cleaned = cleaned.strip()
        # Neutralize formula injection prefixes
        if cleaned.startswith(("=", "+", "-", "@")):
            cleaned = cleaned.lstrip("=+-@\t\r ")
        return cleaned

    @staticmethod
    def format_safe_summary_context(
        batch_id: str,
        decision: str,
        risk_score: float,
        damage_upper: float,
        top_factors: list[tuple[str, float]],
        pii_count: int,
        cells_modified: int,
    ) -> dict[str, Any]:
        """Produce safe statistical-only context dictionary for LLM (Zero raw data cells)."""
        safe_factors = [
            (GuardrailsEngine.sanitize_for_prompt(name), round(float(val), 4))
            for name, val in top_factors
        ]
        return {
            "batch_id": GuardrailsEngine.sanitize_for_prompt(batch_id),
            "decision": decision,
            "risk_score": round(float(risk_score), 1),
            "damage_upper_pct": round(float(damage_upper) * 100.0, 1),
            "top_drivers": safe_factors,
            "pii_findings_count": int(pii_count),
            "cells_modified_count": int(cells_modified),
        }
