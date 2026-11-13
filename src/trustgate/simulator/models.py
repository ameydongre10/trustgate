"""Target model abstractions, pre-training, evaluation, and feature importance weighting."""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from trustgate.simulator.datasets import DatasetSpec


@dataclass
class TargetModel:
    """Wrapped downstream model with pipeline, reference statistics, and metric evaluation."""

    model_id: str
    dataset_name: str
    task_type: Literal["classification", "regression"]
    metric_name: str
    feature_names: list[str]
    categorical_features: list[str]
    feature_importances: dict[str, float]
    training_reference_stats: dict[str, Any]
    pipeline: Any = field(repr=False)

    def evaluate(self, df: pl.DataFrame | np.ndarray, y_true: np.ndarray) -> float:
        """Evaluate model performance metric on provided test batch against true labels."""
        if isinstance(df, pl.DataFrame):
            # Select required features
            X = df.select([c for c in self.feature_names if c in df.columns]).to_pandas()
        else:
            X = df

        if self.task_type == "classification":
            if self.metric_name == "roc_auc":
                y_pred_proba = self.pipeline.predict_proba(X)[:, 1]
                # Guard against single-class evaluation batches
                if len(np.unique(y_true)) < 2:
                    return 0.5
                return float(roc_auc_score(y_true, y_pred_proba))
            elif self.metric_name == "f1":
                y_pred = self.pipeline.predict(X)
                return float(f1_score(y_true, y_pred, zero_division=0))
            else:
                y_pred = self.pipeline.predict(X)
                return float(f1_score(y_true, y_pred, zero_division=0))
        else:
            y_pred = self.pipeline.predict(X)
            if self.metric_name == "rmse":
                return float(np.sqrt(mean_squared_error(y_true, y_pred)))
            elif self.metric_name == "mae":
                return float(mean_absolute_error(y_true, y_pred))
            else:
                return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    def compute_damage(self, clean_metric: float, corrupted_metric: float) -> float:
        """Compute relative metric degradation.

        Classification (higher is better):
            damage = (clean_metric - corrupted_metric) / clean_metric
        Regression (lower is better):
            damage = (corrupted_metric - clean_metric) / clean_metric
        """
        if self.task_type == "classification":
            if clean_metric <= 0:
                return 0.0
            return float((clean_metric - corrupted_metric) / clean_metric)
        else:
            if clean_metric <= 0:
                return 0.0
            return float((corrupted_metric - clean_metric) / clean_metric)

    def get_importance_weighted_vector(self, feature_metrics: dict[str, float]) -> float:
        """Compute importance-weighted average for a per-feature metric (e.g. null rate, PSI)."""
        total_weight = 0.0
        weighted_sum = 0.0
        for feat, val in feature_metrics.items():
            imp = self.feature_importances.get(feat, 0.01)
            weighted_sum += imp * val
            total_weight += imp
        return float(weighted_sum / total_weight) if total_weight > 0 else 0.0

    def save(self, file_path: str | Path) -> None:
        """Serialize TargetModel to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path: str | Path) -> "TargetModel":
        """Load TargetModel from disk."""
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
            if not isinstance(obj, cls):
                raise TypeError(f"Loaded object is not a TargetModel: {type(obj)}")
            return obj


def train_target_model(
    spec: DatasetSpec,
    train_df: pl.DataFrame,
    ref_stats: dict[str, Any],
    model_type: Literal["lightgbm", "random_forest", "linear"],
    seed: int = 42,
) -> TargetModel:
    """Train and calibrate a TargetModel pipeline on clean training data."""
    X_train = train_df.select(spec.all_features).to_pandas()
    y_train = train_df[spec.target_column].to_numpy()

    # Preprocessing pipelines
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, spec.numeric_features),
            ("cat", categorical_transformer, spec.categorical_features),
        ]
    )

    # Estimator selection
    if spec.task_type == "classification":
        if model_type == "lightgbm":
            estimator = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                random_state=seed,
                verbose=-1,
            )
        elif model_type == "random_forest":
            estimator = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=seed,
            )
        else:
            estimator = LogisticRegression(max_iter=1000, random_state=seed)
    else:
        if model_type == "lightgbm":
            estimator = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                random_state=seed,
                verbose=-1,
            )
        elif model_type == "random_forest":
            estimator = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=seed,
            )
        else:
            estimator = Ridge(alpha=1.0, random_state=seed)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )

    # Fit exclusively on clean training data (Fit-on-train discipline)
    pipeline.fit(X_train, y_train)

    # Extract feature importances mapped back to input features
    fitted_model = pipeline.named_steps["model"]
    raw_importances: dict[str, float] = {}

    if hasattr(fitted_model, "feature_importances_"):
        imps = fitted_model.feature_importances_
        # Map one-hot encoded and numeric features back to original features
        feature_names_out = pipeline.named_steps["preprocessor"].get_feature_names_out()
        for out_name, imp_val in zip(feature_names_out, imps, strict=False):
            # Parse original name from prefix like 'num__age' or 'cat__home_ownership_RENT'
            orig_col = out_name.split("__")[-1].split("_")[0]
            # Match against original features
            matched = False
            for feat in spec.all_features:
                if out_name.startswith(f"num__{feat}") or out_name.startswith(f"cat__{feat}"):
                    raw_importances[feat] = raw_importances.get(feat, 0.0) + float(imp_val)
                    matched = True
                    break
            if not matched:
                raw_importances[orig_col] = raw_importances.get(orig_col, 0.0) + float(imp_val)
    elif hasattr(fitted_model, "coef_"):
        coefs = np.abs(fitted_model.coef_).ravel()
        feature_names_out = pipeline.named_steps["preprocessor"].get_feature_names_out()
        for out_name, c_val in zip(feature_names_out, coefs, strict=False):
            matched = False
            for feat in spec.all_features:
                if out_name.startswith(f"num__{feat}") or out_name.startswith(f"cat__{feat}"):
                    raw_importances[feat] = raw_importances.get(feat, 0.0) + float(c_val)
                    matched = True
                    break
            if not matched:
                orig_col = out_name.split("__")[-1]
                raw_importances[orig_col] = raw_importances.get(orig_col, 0.0) + float(c_val)

    # Ensure all original features have a non-zero importance and normalize to sum to 1.0
    for feat in spec.all_features:
        if feat not in raw_importances or raw_importances[feat] <= 0:
            raw_importances[feat] = 0.01

    total_imp = sum(raw_importances.values())
    normalized_importances = {k: float(v / total_imp) for k, v in raw_importances.items()}

    model_id = f"{spec.name}_{model_type}"

    return TargetModel(
        model_id=model_id,
        dataset_name=spec.name,
        task_type=spec.task_type,
        metric_name=spec.metric_name,
        feature_names=spec.all_features,
        categorical_features=spec.categorical_features,
        feature_importances=normalized_importances,
        training_reference_stats=ref_stats,
        pipeline=pipeline,
    )
