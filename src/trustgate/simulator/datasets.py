"""Benchmark datasets generation and reference distribution extraction."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl

from trustgate.utils.seeding import get_rng


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    task_type: Literal["classification", "regression"]
    target_column: str
    numeric_features: list[str]
    categorical_features: list[str]
    metric_name: str
    description: str

    @property
    def all_features(self) -> list[str]:
        return self.numeric_features + self.categorical_features


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "credit_default": DatasetSpec(
        name="credit_default",
        task_type="classification",
        target_column="default",
        numeric_features=[
            "age",
            "income",
            "credit_score",
            "debt_to_income",
            "loan_amount",
            "employment_years",
        ],
        categorical_features=["home_ownership", "loan_purpose"],
        metric_name="roc_auc",
        description="Consumer credit default risk classification benchmark.",
    ),
    "telco_churn": DatasetSpec(
        name="telco_churn",
        task_type="classification",
        target_column="churn",
        numeric_features=["tenure", "monthly_charges", "total_charges"],
        categorical_features=[
            "contract_type",
            "internet_service",
            "payment_method",
            "tech_support",
            "paperless_billing",
        ],
        metric_name="f1",
        description="Subscription telecom customer churn classification benchmark.",
    ),
    "housing_prices": DatasetSpec(
        name="housing_prices",
        task_type="regression",
        target_column="median_house_value",
        numeric_features=[
            "median_income",
            "house_age",
            "avg_rooms",
            "avg_bedrooms",
            "population",
            "avg_occupancy",
            "latitude",
            "longitude",
        ],
        categorical_features=["ocean_proximity"],
        metric_name="rmse",
        description="California housing price valuation regression benchmark.",
    ),
}


def generate_credit_default(n_samples: int = 5000, seed: int = 42) -> pl.DataFrame:
    """Generate realistic synthetic credit default dataset."""
    rng = get_rng(seed)

    age = rng.integers(21, 70, size=n_samples)
    income = np.exp(rng.normal(10.5, 0.6, size=n_samples))  # Log-normal income (~$35k - $120k)
    credit_score = np.clip(rng.normal(670, 75, size=n_samples), 350, 850).astype(int)
    debt_to_income = np.clip(rng.beta(2, 5, size=n_samples) * 0.7, 0.02, 0.65)
    loan_amount = np.clip(income * rng.uniform(0.1, 0.5, size=n_samples), 1000, 50000)
    employment_years = np.clip((age - 18) * rng.beta(2, 3, size=n_samples), 0.0, 40.0).round(1)

    home_ownership_choices = np.array(["RENT", "OWN", "MORTGAGE"])
    home_ownership = rng.choice(home_ownership_choices, size=n_samples, p=[0.45, 0.15, 0.40])

    loan_purpose_choices = np.array(
        ["DEBT_CONSOLIDATION", "HOME_IMPROVEMENT", "BUSINESS", "EDUCATION"]
    )
    loan_purpose = rng.choice(loan_purpose_choices, size=n_samples, p=[0.55, 0.20, 0.15, 0.10])

    # Log-odds of default driven by realistic credit risk formula
    z = (
        -1.5
        - 0.006 * (credit_score - 650)
        + 3.5 * debt_to_income
        - 0.000015 * income
        + 0.00003 * loan_amount
        - 0.04 * employment_years
        + (home_ownership == "RENT") * 0.4
        + rng.normal(0, 0.4, size=n_samples)
    )
    prob_default = 1.0 / (1.0 + np.exp(-z))
    default = (rng.uniform(0, 1, size=n_samples) < prob_default).astype(int)

    return pl.DataFrame(
        {
            "age": age,
            "income": income.round(2),
            "credit_score": credit_score,
            "debt_to_income": debt_to_income.round(4),
            "loan_amount": loan_amount.round(2),
            "employment_years": employment_years,
            "home_ownership": home_ownership,
            "loan_purpose": loan_purpose,
            "default": default,
        }
    )


def generate_telco_churn(n_samples: int = 5000, seed: int = 42) -> pl.DataFrame:
    """Generate realistic telecom customer churn dataset."""
    rng = get_rng(seed)

    tenure = rng.integers(1, 73, size=n_samples)
    contract_type_choices = np.array(["MONTH_TO_MONTH", "ONE_YEAR", "TWO_YEAR"])
    contract_type = rng.choice(contract_type_choices, size=n_samples, p=[0.55, 0.25, 0.20])

    internet_choices = np.array(["DSL", "FIBER_OPTIC", "NO"])
    internet_service = rng.choice(internet_choices, size=n_samples, p=[0.35, 0.45, 0.20])

    payment_choices = np.array(["ELECTRONIC_CHECK", "MAILED_CHECK", "BANK_TRANSFER", "CREDIT_CARD"])
    payment_method = rng.choice(payment_choices, size=n_samples, p=[0.35, 0.20, 0.25, 0.20])

    tech_support_choices = np.array(["YES", "NO"])
    tech_support = rng.choice(tech_support_choices, size=n_samples, p=[0.30, 0.70])

    paperless_choices = np.array(["YES", "NO"])
    paperless_billing = rng.choice(paperless_choices, size=n_samples, p=[0.60, 0.40])

    base_charge = (
        20.0 + (internet_service == "DSL") * 25.0 + (internet_service == "FIBER_OPTIC") * 50.0
    )
    monthly_charges = np.clip(
        base_charge + (tech_support == "YES") * 10.0 + rng.normal(0, 5, size=n_samples),
        18.0,
        120.0,
    ).round(2)
    total_charges = (monthly_charges * tenure * rng.uniform(0.95, 1.05, size=n_samples)).round(2)

    # Churn probability calculation
    z = (
        -0.8
        - 0.04 * tenure
        + 0.02 * (monthly_charges - 60)
        + (contract_type == "MONTH_TO_MONTH") * 1.2
        - (contract_type == "TWO_YEAR") * 1.1
        + (payment_method == "ELECTRONIC_CHECK") * 0.5
        - (tech_support == "YES") * 0.6
        + rng.normal(0, 0.3, size=n_samples)
    )
    prob_churn = 1.0 / (1.0 + np.exp(-z))
    churn = (rng.uniform(0, 1, size=n_samples) < prob_churn).astype(int)

    return pl.DataFrame(
        {
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "payment_method": payment_method,
            "tech_support": tech_support,
            "paperless_billing": paperless_billing,
            "churn": churn,
        }
    )


def generate_housing_prices(n_samples: int = 5000, seed: int = 42) -> pl.DataFrame:
    """Generate realistic California housing price dataset."""
    rng = get_rng(seed)

    median_income = np.clip(rng.gamma(4, 0.8, size=n_samples), 0.5, 15.0).round(4)
    house_age = rng.integers(1, 52, size=n_samples)
    avg_rooms = np.clip(rng.normal(5.4, 1.2, size=n_samples), 1.5, 14.0).round(2)
    avg_bedrooms = np.clip(avg_rooms * rng.uniform(0.18, 0.30, size=n_samples), 0.8, 5.0).round(2)
    population = np.clip(rng.lognormal(7.0, 0.8, size=n_samples), 100, 35000).astype(int)
    avg_occupancy = np.clip(rng.normal(3.0, 0.6, size=n_samples), 1.0, 10.0).round(2)

    latitude = np.clip(rng.uniform(32.5, 42.0, size=n_samples), 32.5, 42.0).round(4)
    longitude = np.clip(rng.uniform(-124.3, -114.3, size=n_samples), -124.3, -114.3).round(4)

    ocean_proximity_choices = np.array(["NEAR_BAY", "INLAND", "NEAR_OCEAN", "ISLAND"])
    ocean_proximity = rng.choice(
        ocean_proximity_choices, size=n_samples, p=[0.20, 0.50, 0.28, 0.02]
    )

    # Housing valuation formula in dollars ($50k to $500k)
    base_val = (
        45000.0
        + 40000.0 * median_income
        + 800.0 * house_age
        + 8000.0 * (avg_rooms - avg_bedrooms)
        - 15000.0 * (ocean_proximity == "INLAND")
        + 30000.0 * (ocean_proximity == "NEAR_OCEAN")
        + 40000.0 * (ocean_proximity == "ISLAND")
        + rng.normal(0, 25000, size=n_samples)
    )
    median_house_value = np.clip(base_val, 45000.0, 500000.0).round(0)

    return pl.DataFrame(
        {
            "median_income": median_income,
            "house_age": house_age,
            "avg_rooms": avg_rooms,
            "avg_bedrooms": avg_bedrooms,
            "population": population,
            "avg_occupancy": avg_occupancy,
            "latitude": latitude,
            "longitude": longitude,
            "ocean_proximity": ocean_proximity,
            "median_house_value": median_house_value,
        }
    )


def extract_reference_stats(df: pl.DataFrame, spec: DatasetSpec) -> dict[str, Any]:
    """Compute training reference statistics (quantiles, mean, std, category distributions)."""
    stats: dict[str, Any] = {
        "dataset_name": spec.name,
        "n_samples": df.height,
        "numeric": {},
        "categorical": {},
    }

    quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    for col in spec.numeric_features:
        if col not in df.columns:
            continue
        vals = df[col].drop_nulls().to_numpy()
        if len(vals) == 0:
            continue
        q_vals = {f"q_{int(q * 100):02d}": float(np.quantile(vals, q)) for q in quantiles}
        stats["numeric"][col] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)) if len(vals) > 1 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "quantiles": q_vals,
            "null_rate": float(df[col].null_count() / df.height),
        }

    for col in spec.categorical_features:
        if col not in df.columns:
            continue
        counts = df[col].drop_nulls().value_counts()
        total = df[col].drop_nulls().len()
        freqs = {}
        if total > 0:
            for row in counts.iter_rows():
                freqs[str(row[0])] = float(row[1] / total)
        stats["categorical"][col] = {
            "frequencies": freqs,
            "unique_count": df[col].n_unique(),
            "null_rate": float(df[col].null_count() / df.height),
        }

    return stats


def get_dataset(
    name: str,
    n_samples: int = 5000,
    seed: int = 42,
    split_ratio: float = 0.8,
) -> tuple[pl.DataFrame, pl.DataFrame, DatasetSpec, dict[str, Any]]:
    """Get train and test partitions for a registered dataset with reference stats.

    Returns
    -------
    (train_df, test_df, spec, reference_stats)
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY.keys())}")

    spec = DATASET_REGISTRY[name]

    if name == "credit_default":
        df = generate_credit_default(n_samples=n_samples, seed=seed)
    elif name == "telco_churn":
        df = generate_telco_churn(n_samples=n_samples, seed=seed)
    elif name == "housing_prices":
        df = generate_housing_prices(n_samples=n_samples, seed=seed)
    else:
        raise ValueError(f"Unhandled dataset: {name}")

    # Deterministic train/test split
    n_train = int(df.height * split_ratio)
    train_df = df.slice(0, n_train)
    test_df = df.slice(n_train, df.height - n_train)

    ref_stats = extract_reference_stats(train_df, spec)
    return train_df, test_df, spec, ref_stats
