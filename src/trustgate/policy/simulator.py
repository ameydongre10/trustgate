"""Historical batch policy replay simulator analyzing governance trade-off curves."""

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class PolicySimulationPoint:
    approve_threshold: float
    repair_threshold: float
    n_batches: int
    approved_count: int
    repaired_count: int
    quarantined_count: int
    approval_rate: float
    quarantine_rate: float
    avg_approved_damage: float
    max_approved_damage: float


class PolicySimulator:
    """Replays historical simulation experiments through alternative governance policy thresholds."""

    def __init__(self, experiment_df: pl.DataFrame) -> None:
        self.experiment_df = experiment_df

    def simulate_threshold_sweep(
        self,
        approve_thresholds: list[float] | None = None,
        repair_margin: float = 0.15,
    ) -> list[PolicySimulationPoint]:
        """Simulate decisions across a sweep of approval thresholds to map risk vs throughput."""
        thresholds = approve_thresholds or [0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
        results: list[PolicySimulationPoint] = []

        damages = self.experiment_df["damage"].to_numpy()
        n_total = len(damages)
        if n_total == 0:
            return []

        for tau_app in thresholds:
            tau_rep = tau_app + repair_margin

            # Decision simulation
            is_approved = damages <= tau_app
            is_repaired = (damages > tau_app) & (damages <= tau_rep)
            is_quarantined = damages > tau_rep

            approved_count = int(np.sum(is_approved))
            repaired_count = int(np.sum(is_repaired))
            quarantined_count = int(np.sum(is_quarantined))

            # Filter approved damages
            approved_damages = damages[is_approved]
            avg_damage = float(np.mean(approved_damages)) if len(approved_damages) > 0 else 0.0
            max_damage = float(np.max(approved_damages)) if len(approved_damages) > 0 else 0.0

            results.append(
                PolicySimulationPoint(
                    approve_threshold=tau_app,
                    repair_threshold=tau_rep,
                    n_batches=n_total,
                    approved_count=approved_count,
                    repaired_count=repaired_count,
                    quarantined_count=quarantined_count,
                    approval_rate=float(approved_count / n_total),
                    quarantine_rate=float(quarantined_count / n_total),
                    avg_approved_damage=avg_damage,
                    max_approved_damage=max_damage,
                )
            )

        return results
