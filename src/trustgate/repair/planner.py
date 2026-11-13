"""Closed-loop repair planner with cell-budget enforcement and rollback verification."""

from dataclasses import dataclass
from typing import Any

import polars as pl

from trustgate.damage.predictor import DamagePredictor
from trustgate.repair.strategies import (
    RepairPatch,
    coerce_types,
    impute_missing_values,
    map_unseen_categories,
    realign_schema,
    repair_sentinels,
    winsorize_outliers,
)
from trustgate.simulator.datasets import DatasetSpec
from trustgate.simulator.models import TargetModel
from trustgate.simulator.runner import extract_fast_issue_features


@dataclass
class ClosedLoopRepairResult:
    success: bool
    repaired_df: pl.DataFrame
    pre_damage: float
    post_damage: float
    damage_reduction: float
    budget_exceeded: bool
    rolled_back: bool
    cells_modified: int
    fraction_modified: float
    patches: list[RepairPatch]
    verdict: str
    message: str


class RepairPlanner:
    """Orchestrates closed-loop repair planning, cell budget gating, and rollback validation."""

    def __init__(
        self,
        max_budget_pct: float = 0.15,
        target_damage_threshold: float = 0.05,
    ) -> None:
        self.max_budget_pct = max_budget_pct
        self.target_damage_threshold = target_damage_threshold

    def execute_closed_loop_repair(
        self,
        df: pl.DataFrame,
        spec: DatasetSpec,
        target_model: TargetModel,
        ref_df: pl.DataFrame,
        ref_stats: dict[str, Any],
        damage_predictor: DamagePredictor,
    ) -> ClosedLoopRepairResult:
        """Execute multi-strategy repair, measure post-repair damage, and rollback if degraded."""
        total_cells = df.height * max(1, len(df.columns))
        max_allowed_modifications = int(total_cells * self.max_budget_pct)

        # 1. Measure Pre-Repair Damage
        pre_profile = extract_fast_issue_features(df, ref_df, target_model, spec)
        pre_pred = damage_predictor.predict(pre_profile)
        pre_damage = pre_pred.predicted_damage

        # 2. Sequential execution of repair strategies
        current_df = df.clone()
        all_patches: list[RepairPatch] = []

        # Strategy 1: Schema realignment
        current_df, s_patches = realign_schema(current_df, ref_stats, spec.all_features)
        all_patches.extend(s_patches)

        # Strategy 2: Type coercion
        current_df, t_patches = coerce_types(current_df, ref_stats)
        all_patches.extend(t_patches)

        # Strategy 3: Sentinel repair
        current_df, sen_patches = repair_sentinels(current_df, ref_stats)
        all_patches.extend(sen_patches)

        # Strategy 4: Unseen category mapping
        current_df, cat_patches = map_unseen_categories(current_df, ref_stats)
        all_patches.extend(cat_patches)

        # Strategy 5: Reference-fitted median/mode imputation
        current_df, imp_patches = impute_missing_values(current_df, ref_stats)
        all_patches.extend(imp_patches)

        # Strategy 6: Outlier winsorization
        current_df, win_patches = winsorize_outliers(current_df, ref_stats)
        all_patches.extend(win_patches)

        cells_modified = len(all_patches)
        fraction_modified = float(cells_modified / max(1, total_cells))

        # 3. Budget Check Gate
        if cells_modified > max_allowed_modifications:
            return ClosedLoopRepairResult(
                success=False,
                repaired_df=df,  # Rollback to original raw batch
                pre_damage=pre_damage,
                post_damage=pre_damage,
                damage_reduction=0.0,
                budget_exceeded=True,
                rolled_back=True,
                cells_modified=cells_modified,
                fraction_modified=fraction_modified,
                patches=[],
                verdict="BUDGET_EXCEEDED",
                message=(
                    f"Repair rejected: Required modifications ({cells_modified} cells, {fraction_modified * 100:.1f}%) "
                    f"exceeded maximum allowed budget ({self.max_budget_pct * 100:.1f}%)."
                ),
            )

        if cells_modified == 0:
            return ClosedLoopRepairResult(
                success=True,
                repaired_df=df,
                pre_damage=pre_damage,
                post_damage=pre_damage,
                damage_reduction=0.0,
                budget_exceeded=False,
                rolled_back=False,
                cells_modified=0,
                fraction_modified=0.0,
                patches=[],
                verdict="NO_REPAIR_NEEDED",
                message="Data batch already healthy; zero repair modifications required.",
            )

        # 4. Closed-Loop Re-Profiling and Damage Re-evaluation
        post_profile = extract_fast_issue_features(current_df, ref_df, target_model, spec)
        post_pred = damage_predictor.predict(post_profile)
        post_damage = post_pred.predicted_damage
        damage_reduction = pre_damage - post_damage

        # 5. Degradation Rollback Guard (ADR-004)
        if post_damage > pre_damage + 0.005:  # Tolerance for prediction noise
            # Repair degraded model performance! Rollback immediately!
            return ClosedLoopRepairResult(
                success=False,
                repaired_df=df,  # Rollback to raw original
                pre_damage=pre_damage,
                post_damage=post_damage,
                damage_reduction=damage_reduction,
                budget_exceeded=False,
                rolled_back=True,
                cells_modified=cells_modified,
                fraction_modified=fraction_modified,
                patches=all_patches,
                verdict="ROLLBACK_DEGRADATION",
                message=(
                    f"Repair worsened predicted damage from {pre_damage * 100:.1f}% to {post_damage * 100:.1f}%. "
                    "All patches rolled back to preserve original data."
                ),
            )

        # 6. Successful Repair
        return ClosedLoopRepairResult(
            success=True,
            repaired_df=current_df,
            pre_damage=pre_damage,
            post_damage=post_damage,
            damage_reduction=damage_reduction,
            budget_exceeded=False,
            rolled_back=False,
            cells_modified=cells_modified,
            fraction_modified=fraction_modified,
            patches=all_patches,
            verdict="REPAIRED",
            message=(
                f"Successfully repaired {cells_modified} cells ({fraction_modified * 100:.2f}% of batch). "
                f"Reduced predicted damage from {pre_damage * 100:.1f}% to {post_damage * 100:.1f}%."
            ),
        )
