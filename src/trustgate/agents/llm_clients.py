"""LLM provider abstraction with deterministic offline fallback and remote clients."""

import os
from typing import Any, Protocol


class LLMClient(Protocol):
    """Protocol for LLM explanation providers."""

    def generate_explanation(self, context: dict[str, Any]) -> str:
        """Generate human-readable narrative explanation from verified decision context."""
        ...


class TemplateLLMClient:
    """Deterministic, 100% offline rule-based explanation generator (Zero API keys required)."""

    def generate_explanation(self, context: dict[str, Any]) -> str:
        """Generate precise, verified narrative template from mathematical context."""
        decision = context.get("decision", "QUARANTINE")
        batch_id = context.get("batch_id", "UNKNOWN")
        risk_score = context.get("risk_score", 0.0)
        damage_upper_pct = context.get("damage_upper_pct", 0.0)
        top_drivers = context.get("top_drivers", [])
        pii_count = context.get("pii_findings_count", 0)
        cells_modified = context.get("cells_modified_count", 0)

        # Build clean narrative
        lines = [
            f"### TrustGate Governance Decision: {decision}",
            f"**Batch ID**: `{batch_id}` | **Risk Score**: {risk_score:.1f}/100 | **Predicted Damage Bound**: {damage_upper_pct:.1f}%",
            "",
        ]

        if decision == "APPROVE":
            lines.append(
                f"Batch `{batch_id}` has been approved for downstream ingestion. "
                f"Predicted model metric degradation ({damage_upper_pct:.1f}%) is within the approved tolerance threshold. "
                "Data distributions, schema integrity, and privacy standards have been validated."
            )
        elif decision == "REPAIR_THEN_APPROVE":
            lines.append(
                f"Batch `{batch_id}` required automated data remediation before approval. "
                f"TrustGate successfully applied closed-loop repair across {cells_modified} cells, "
                f"reducing predicted metric damage to an acceptable {damage_upper_pct:.1f}%. "
                "Reversible lineage patches have been recorded in the audit log."
            )
        elif decision == "ESCALATE_TO_HUMAN":
            lines.append(
                f"Batch `{batch_id}` has been routed to human review. "
                "The batch issue profile exhibits an out-of-distribution anomaly or falls within a high-sensitivity risk boundary. "
                "A human governance operator must inspect and approve this batch before pipeline execution."
            )
        else:  # QUARANTINE
            lines.append(
                f"Batch `{batch_id}` has been quarantined to prevent model degradation or privacy violations. "
                f"Predicted downstream metric degradation is {damage_upper_pct:.1f}% (Risk Score: {risk_score:.1f}/100)."
            )

        if pii_count > 0:
            lines.append(
                f"- **Privacy Alert**: {pii_count} sensitive PII entities were detected in this batch."
            )

        if top_drivers:
            lines.append("- **Primary Contributing Factors**:")
            for name, val in top_drivers[:3]:
                lines.append(f"  * {name}: impact contribution +{val * 100:.1f}%")

        return "\n".join(lines)


class AnthropicLLMClient:
    """Anthropic Claude client with automatic fallback to TemplateLLMClient on failure."""

    def __init__(self, api_key: str | None = None, model: str = "claude-3-haiku-20240307") -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model
        self.fallback = TemplateLLMClient()

    def generate_explanation(self, context: dict[str, Any]) -> str:
        if not self.api_key:
            return self.fallback.generate_explanation(context)

        try:
            import httpx

            prompt = (
                "You are TrustGate Explainer, an AI data governance agent. Summarize the following "
                "governance decision concisely in plain language. ONLY cite numbers that appear directly "
                f"in this context:\n{context}"
            )
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": 400,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=10.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                return str(data["content"][0]["text"]).strip()
        except Exception:
            pass

        return self.fallback.generate_explanation(context)


class OpenAILLMClient:
    """OpenAI-compatible client with automatic fallback to TemplateLLMClient on failure."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url
        self.model = model
        self.fallback = TemplateLLMClient()

    def generate_explanation(self, context: dict[str, Any]) -> str:
        if not self.api_key:
            return self.fallback.generate_explanation(context)

        try:
            import httpx

            prompt = (
                "You are TrustGate Explainer, an AI data governance agent. Summarize the following "
                "governance decision concisely. ONLY cite numbers that appear directly in this context:\n"
                f"{context}"
            )
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 400,
                },
                timeout=10.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                return str(data["choices"][0]["message"]["content"]).strip()
        except Exception:
            pass

        return self.fallback.generate_explanation(context)


def get_llm_client(provider: str | None = None) -> LLMClient:
    """Factory selecting the requested LLM client, defaulting to deterministic template."""
    prov = (provider or os.getenv("TRUSTGATE_LLM_PROVIDER") or "template").lower().strip()
    if prov == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
        return AnthropicLLMClient()
    elif prov in ("openai", "ollama") and os.getenv("OPENAI_API_KEY"):
        return OpenAILLMClient()
    return TemplateLLMClient()
