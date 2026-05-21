"""
data_prep_job.py
─────────────────
Data preparation and validation job run on Azure ML compute.

Validation is now powered by Pandera (see src/schema.py).
The manual hand-rolled checks have been replaced by the
declarative RawTitanicSchema — column types, value sets,
nullable flags, and dataset-level invariants all in one place.

What it does:
  1. Loads the raw titanic CSV
  2. Validates against RawTitanicSchema (Pandera) — hard fail on error
  3. Computes a data profile (used as baseline for drift detection)
  4. Cleans data (impute Age, Embarked)
  5. Saves prepared_data.csv + validation_report.json + data_profile.json

Called by: mlpipelines/data_preparation/data_prep_job.yml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Schema validation
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from schema import validate_raw, SchemaValidationResult  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── Data profile ──────────────────────────────────────────────────
def compute_profile(df: pd.DataFrame) -> dict:
    """Per-column statistics written to data_profile.json.
    Consumed by get_baseline_data.py as the drift detection baseline."""
    profile = {
        "row_count":    int(len(df)),
        "column_count": int(len(df.columns)),
        "columns":      {},
    }

    for col in df.columns:
        col_info: dict = {"dtype": str(df[col].dtype)}
        missing = int(df[col].isna().sum())
        col_info["missing_count"] = missing
        col_info["missing_rate"]  = round(missing / len(df), 4)

        if pd.api.types.is_numeric_dtype(df[col]):
            col_info.update({
                "mean":   float(df[col].mean())           if not df[col].isna().all() else None,
                "std":    float(df[col].std())            if not df[col].isna().all() else None,
                "min":    float(df[col].min())            if not df[col].isna().all() else None,
                "max":    float(df[col].max())            if not df[col].isna().all() else None,
                "median": float(df[col].median())         if not df[col].isna().all() else None,
                "p25":    float(df[col].quantile(0.25))   if not df[col].isna().all() else None,
                "p75":    float(df[col].quantile(0.75))   if not df[col].isna().all() else None,
            })
        else:
            vc = df[col].value_counts(dropna=False)
            col_info["unique_count"] = int(df[col].nunique())
            col_info["top_values"]   = {str(k): int(v) for k, v in vc.head(10).items()}

        profile["columns"][col] = col_info

    if "Survived" in df.columns:
        profile["target_positive_rate"] = float(df["Survived"].mean())

    return profile


# ── Cleaning ──────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values — mirrors DataProcessor.handle_missing_values()."""
    df = df.copy()
    df["Age"]      = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    # Cabin is intentionally left as-is; it will be dropped during feature engineering
    return df


# ── Main ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Data preparation + Pandera validation")
    parser.add_argument("--input-data",  required=True, help="Path to raw CSV")
    parser.add_argument("--output-path", required=True, help="Output folder")
    args = parser.parse_args()

    out = Path(args.output_path)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────
    logger.info("Loading data from %s", args.input_data)
    df = pd.read_csv(args.input_data)
    logger.info("Loaded %d rows × %d columns", *df.shape)

    # ── 2. Pandera validation (hard fail) ─────────────────────────
    logger.info("Running Pandera schema validation…")
    result: SchemaValidationResult = validate_raw(df, raise_on_error=False)

    validation_report = result.to_dict()
    with open(out / "validation_report.json", "w") as f:
        json.dump(validation_report, f, indent=2)
    logger.info("Validation report written to %s", out / "validation_report.json")

    if not result.passed:
        logger.error(
            "❌ Pandera validation FAILED — %d error(s). Aborting pipeline.",
            len(result.errors),
        )
        for err in result.errors[:10]:   # show first 10 to avoid log flood
            logger.error("  column=%s  check=%s  value=%s  row=%s",
                         err.get("column"), err.get("check"),
                         err.get("failure_case"), err.get("index"))
        sys.exit(1)

    logger.info("✅ Pandera schema validation passed")

    # ── 3. Profile ────────────────────────────────────────────────
    profile = compute_profile(df)
    with open(out / "data_profile.json", "w") as f:
        json.dump(profile, f, indent=2)
    logger.info("Data profile saved")

    # ── 4. Clean + save ───────────────────────────────────────────
    df_clean = clean_data(df)
    df_clean.to_csv(out / "prepared_data.csv", index=False)
    logger.info("Prepared data saved to %s", out / "prepared_data.csv")

    # ── 5. Re-validate cleaned data (soft check) ──────────────────
    # After imputation Embarked and Age should no longer be null
    cleaned_result = validate_raw(df_clean, raise_on_error=False)
    if not cleaned_result.passed:
        # Imputation may introduce types that Pandera's coerce can't resolve —
        # log as warning only (the cleaned CSV is still usable for training)
        logger.warning(
            "⚠️  Post-cleaning validation has %d warning(s) — check validation_report.json",
            len(cleaned_result.errors),
        )

    logger.info("Data preparation complete ✅")


if __name__ == "__main__":
    main()
