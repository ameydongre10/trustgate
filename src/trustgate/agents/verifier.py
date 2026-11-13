"""Numeric claim verifier guarding against LLM hallucinations in governance narratives."""

import re
from typing import Any

from trustgate.agents.llm_clients import TemplateLLMClient


class NumericClaimVerifier:
    """Verifies that every numerical claim in generated text matches ground-truth decision records."""

    def __init__(self, tolerance_pct: float = 0.5) -> None:
        self.tolerance_pct = tolerance_pct
        self.fallback = TemplateLLMClient()

    def extract_numbers(self, text: str) -> list[float]:
        """Extract all floating point and integer numbers from text."""
        # Match percentages and standalone numbers, e.g. "12.5%", "100", "0.05"
        matches = re.findall(r"\b\d+(?:\.\d+)?%?\b", text)
        numbers = []
        for m in matches:
            clean_m = m.replace("%", "").strip()
            try:
                numbers.append(float(clean_m))
            except ValueError:
                pass
        return numbers

    def verify_and_correct(
        self,
        generated_text: str,
        ground_truth_context: dict[str, Any],
    ) -> tuple[bool, str, list[str]]:
        """Cross-check numbers in narrative against ground truth; fall back to template on mismatch.

        Returns
        -------
        (is_valid, final_narrative, list_of_violations)
        """
        extracted_numbers = self.extract_numbers(generated_text)
        if not extracted_numbers:
            # No numbers claimed; valid
            return True, generated_text, []

        # Collect ground truth valid numbers
        valid_numbers: set[float] = set()

        def add_val(val: Any) -> None:
            if isinstance(val, (int, float)):
                v = float(val)
                valid_numbers.add(v)
                valid_numbers.add(round(v, 1))
                valid_numbers.add(round(v, 2))
                # Add percentage representation (e.g. 0.05 -> 5.0)
                if 0.0 <= v <= 1.0:
                    pct = round(v * 100.0, 1)
                    valid_numbers.add(pct)

        add_val(ground_truth_context.get("risk_score"))
        add_val(ground_truth_context.get("damage_upper_pct"))
        add_val(ground_truth_context.get("pii_findings_count"))
        add_val(ground_truth_context.get("cells_modified_count"))

        for _, imp_val in ground_truth_context.get("top_drivers", []):
            add_val(imp_val)

        # Allow benign list indices and baseline standard constants (1, 2, 3, 100)
        benign_constants = {1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 100.0}
        valid_numbers.update(benign_constants)

        violations: list[str] = []
        for num in extracted_numbers:
            matched = False
            for v in valid_numbers:
                if abs(num - v) <= self.tolerance_pct or (v != 0 and abs(num - v) / v <= 0.05):
                    matched = True
                    break
            if not matched:
                violations.append(f"Unverified or hallucinated numerical claim: '{num}'")

        if violations:
            # Numeric hallucination detected! Replace with verified deterministic template.
            fallback_text = self.fallback.generate_explanation(ground_truth_context)
            return False, fallback_text, violations

        return True, generated_text, []
