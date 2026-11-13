"""Model registry with SHA-256 hash pinning, versioning, and Model Card generation."""

import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any

from trustgate.damage.predictor import DamagePredictor
from trustgate.damage.validation import ProtocolMetrics


class ModelRegistry:
    """Manages versioned, cryptographic hash-pinned model artifacts and Model Cards."""

    def __init__(self, registry_dir: str = "models/damage") -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.registry_dir / "registry_manifest.json"
        self._ensure_manifest()

    def _ensure_manifest(self) -> None:
        if not self.manifest_path.exists():
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump({"models": {}}, f, indent=2)

    def _read_manifest(self) -> dict[str, Any]:
        with open(self.manifest_path, encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return {"models": {}}
            return data

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def register(
        self,
        predictor: DamagePredictor,
        version: str = "1.0.0",
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Serialize predictor, compute SHA-256 checksum, and record entry in registry manifest."""
        file_name = f"damage_predictor_v{version}.pkl"
        artifact_path = self.registry_dir / file_name

        # Serialize
        with open(artifact_path, "wb") as f:
            pickle.dump(predictor, f)

        # Compute SHA-256
        hasher = hashlib.sha256()
        with open(artifact_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        sha256_hash = hasher.hexdigest()

        # Update manifest
        manifest = self._read_manifest()
        manifest["models"][version] = {
            "version": version,
            "filename": file_name,
            "sha256_hash": sha256_hash,
            "registered_at": time.time(),
            "metadata": metadata or {},
        }
        self._write_manifest(manifest)

        return artifact_path

    def load(self, version: str = "1.0.0", verify_hash: bool = True) -> DamagePredictor:
        """Load and optionally verify integrity of a versioned damage predictor."""
        manifest = self._read_manifest()
        if version not in manifest["models"]:
            raise KeyError(f"Model version '{version}' not found in registry.")

        entry = manifest["models"][version]
        artifact_path = self.registry_dir / entry["filename"]

        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact file not found: {artifact_path}")

        if verify_hash:
            hasher = hashlib.sha256()
            with open(artifact_path, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            current_hash = hasher.hexdigest()
            if current_hash != entry["sha256_hash"]:
                raise ValueError(
                    f"Integrity check failed for {entry['filename']}! "
                    f"Expected SHA-256: {entry['sha256_hash']}, actual: {current_hash}"
                )

        with open(artifact_path, "rb") as f:
            obj = pickle.load(f)
            if not isinstance(obj, DamagePredictor):
                raise TypeError(f"Loaded object is not a DamagePredictor: {type(obj)}")
            return obj

    def generate_model_card(
        self,
        random_split_metrics: ProtocolMetrics,
        loco_metrics: dict[str, ProtocolMetrics],
        lodo_metrics: dict[str, ProtocolMetrics],
        output_path: str = "docs/model_card.md",
    ) -> Path:
        """Generate comprehensive Model Card following Mitchell et al. format."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        card_text = f"""# TrustGate Model Card: Damage Predictor

## 1. Model Details
- **Architecture**: Gradient Boosted Trees (`LightGBM`) with strictly monotonic increasing constraints on risk features.
- **Uncertainty Quantification**: Split-conformal prediction intervals with finite-sample $(1 - \\alpha)$ coverage guarantees.
- **Safety Mechanism**: Integrated Out-of-Distribution (OOD) Guard combining Isolation Forest and regularized Mahalanobis distance.
- **Interpretability**: Native TreeSHAP computation (`pred_contrib=True`) mapped to human-readable degradation drivers.
- **Fit-on-Train Discipline**: All imputers, reference statistics, and conformal non-conformity calibration sets fitted exclusively on clean baseline splits.

---

## 2. Intended Use & Policy Integration
- **Primary Task**: Predict the relative metric degradation ($Damage$) that an incoming batch of data will cause on a specified downstream machine learning model.
- **Decision Engine (The Judge)**: The deterministic Judge evaluates $Damage_{{upper}}$ (the upper bound of the split-conformal interval) against critical risk thresholds.
- **Out-of-Scope Use**: Not intended for direct human surveillance, non-tabular modalities without feature extraction, or models without registered baseline metrics.

---

## 3. Generalization Performance Across Validation Protocols

### 3.1 Random Split (In-Distribution Baseline)
- **RMSE**: `{random_split_metrics.rmse:.4f}` vs Linear Baseline `{random_split_metrics.baseline_linear_rmse:.4f}`
- **MAE**: `{random_split_metrics.mae:.4f}`
- **Spearman Rank Correlation**: `{random_split_metrics.spearman_corr:.3f}`
- **Conformal Coverage (Target 90%)**: `{random_split_metrics.conformal_coverage * 100:.1f}%`

### 3.2 Leave-One-Corruption-Out (LOCO) - Generalization to Unseen Corruptions
| Held-out Corruption | Predictor RMSE | Linear RMSE | Conformal Coverage |
| :--- | :--- | :--- | :--- |
"""
        for c_type, m in loco_metrics.items():
            card_text += f"| **{c_type}** | `{m.rmse:.4f}` | `{m.baseline_linear_rmse:.4f}` | `{m.conformal_coverage * 100:.1f}%` |\n"

        card_text += """
### 3.3 Leave-One-Dataset-Out (LODO) - Cross-Domain Generalization
| Held-out Dataset | Predictor RMSE | Linear RMSE | Conformal Coverage |
| :--- | :--- | :--- | :--- |
"""
        for d_name, m in lodo_metrics.items():
            card_text += f"| **{d_name}** | `{m.rmse:.4f}` | `{m.baseline_linear_rmse:.4f}` | `{m.conformal_coverage * 100:.1f}%` |\n"

        card_text += """
---

## 4. Ethical Considerations & Limitations
- **Fail-Safe Principle**: When OOD Guard detects high anomaly distances, confidence drops to $< 0.50$, forcing the Judge to quarantine or escalate to a human operator.
- **Zero Fabrication**: All metrics reported above are computed directly from the validated test suite without synthetic inflation.
- **Deterministic Separation**: Explanations and narratives generated by LLMs are strictly verified against these mathematical outputs before display.
"""

        with open(path, "w", encoding="utf-8") as f:
            f.write(card_text)

        return path
