"""
schema.py
──────────
Pandera schema definitions for the Titanic MLOps pipeline.

Two schemas are defined:

  RawTitanicSchema    — validates the raw CSV before any processing.
                        Used in data_prep_job.py and DataProcessor.load_data().

  ProcessedFeatureSchema — validates the engineered feature DataFrame
                           after DataProcessor has run all transformations.
                           Used in DataProcessor.select_features().

Why Pandera over manual checks?
  - Declarative — the schema IS the documentation
  - Per-column type + range + value-set enforcement in one place
  - SchemaErrors carry column-level detail (not just pass/fail)
  - Integrates cleanly with DataFrames (no extra boilerplate)
  - Generates a JSON-serialisable error report for pipeline artifacts
"""

import json
import logging
from typing import Optional

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
# 1. Raw Titanic CSV Schema
#    Validates the dataset exactly as it comes off disk.
#    Nullable columns are explicitly marked — Cabin and Age both
#    have significant missingness in the real dataset.
# ════════════════════════════════════════════════════════════════════

RawTitanicSchema = DataFrameSchema(
    columns={
        "PassengerId": Column(
            int,
            checks=Check.greater_than(0),
            nullable=False,
            unique=True,
            description="Unique passenger identifier — must be positive integer",
        ),
        "Survived": Column(
            int,
            checks=Check.isin([0, 1]),
            nullable=False,
            description="Survival label: 0 = did not survive, 1 = survived",
        ),
        "Pclass": Column(
            int,
            checks=Check.isin([1, 2, 3]),
            nullable=False,
            description="Ticket class: 1st, 2nd, or 3rd",
        ),
        "Name": Column(
            str,
            checks=Check(lambda s: s.str.len() > 0, element_wise=False),
            nullable=False,
            description="Passenger full name — used for title extraction",
        ),
        "Sex": Column(
            str,
            checks=Check.isin(["male", "female"]),
            nullable=False,
            description="Passenger sex",
        ),
        "Age": Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(120),
            ],
            nullable=True,   # ~20% missing in real dataset
            description="Passenger age in years — nullable, imputed with median",
        ),
        "SibSp": Column(
            int,
            checks=Check.greater_than_or_equal_to(0),
            nullable=False,
            description="Number of siblings/spouses aboard",
        ),
        "Parch": Column(
            int,
            checks=Check.greater_than_or_equal_to(0),
            nullable=False,
            description="Number of parents/children aboard",
        ),
        "Ticket": Column(
            str,
            nullable=False,
            description="Ticket number (not used as feature, kept for traceability)",
        ),
        "Fare": Column(
            float,
            checks=Check.greater_than_or_equal_to(0),
            nullable=False,
            description="Passenger fare — must be non-negative",
        ),
        "Cabin": Column(
            str,
            nullable=True,   # ~77% missing — dropped in feature engineering
            description="Cabin number — nullable, dropped during processing",
            required=False,  # Some datasets omit this column entirely
        ),
        "Embarked": Column(
            str,
            checks=Check.isin(["S", "C", "Q"]),
            nullable=True,   # 2 rows missing in Titanic dataset
            description="Port of embarkation: S=Southampton, C=Cherbourg, Q=Queenstown",
        ),
    },
    checks=[
        # Dataset-level checks (applied to the whole DataFrame)
        Check(
            lambda df: len(df) >= 500,
            error="Dataset has fewer than 500 rows — likely a truncated file",
        ),
        Check(
            lambda df: df["Survived"].mean() >= 0.25,
            error="Survival rate below 25% — possible data corruption or wrong file",
        ),
    ],
    coerce=True,        # Attempt type coercion before raising errors
    strict=False,       # Allow extra columns (e.g. index columns from exports)
    name="RawTitanicSchema",
)


# ════════════════════════════════════════════════════════════════════
# 2. Processed Feature Schema
#    Validates the DataFrame produced by DataProcessor.select_features()
#    — after missing value imputation, encoding, and feature engineering.
#    All columns are numeric at this stage (LabelEncoder + StandardScaler
#    has NOT run yet — this validates the pre-scale feature frame).
# ════════════════════════════════════════════════════════════════════

ProcessedFeatureSchema = DataFrameSchema(
    columns={
        "Pclass": Column(
            int,
            checks=Check.isin([1, 2, 3]),
            nullable=False,
        ),
        "Sex": Column(
            int,
            checks=Check.isin([0, 1]),  # LabelEncoded: female=0, male=1
            nullable=False,
        ),
        "Age": Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(120),
            ],
            nullable=False,   # imputed — no longer nullable
        ),
        "SibSp": Column(
            int,
            checks=Check.greater_than_or_equal_to(0),
            nullable=False,
        ),
        "Parch": Column(
            int,
            checks=Check.greater_than_or_equal_to(0),
            nullable=False,
        ),
        "Fare": Column(
            float,
            checks=Check.greater_than_or_equal_to(0),
            nullable=False,
        ),
        "Title": Column(
            int,
            checks=Check.greater_than_or_equal_to(0),
            nullable=False,
            description="LabelEncoded title (Mr, Mrs, Miss, Master, Rare)",
        ),
        "FamilySize": Column(
            int,
            checks=[
                Check.greater_than_or_equal_to(1),
                Check.less_than_or_equal_to(20),
            ],
            nullable=False,
        ),
        "IsAlone": Column(
            int,
            checks=Check.isin([0, 1]),
            nullable=False,
        ),
        "AgeBand": Column(
            int,
            checks=Check.greater_than_or_equal_to(0),
            nullable=False,
            description="LabelEncoded age band: Child/Teenager/Adult/Middle-aged/Senior",
        ),
        "FareBand": Column(
            int,
            checks=Check.greater_than_or_equal_to(0),
            nullable=False,
            description="LabelEncoded fare quartile: Q1/Q2/Q3/Q4",
        ),
        "Embarked": Column(
            int,
            checks=Check.isin([0, 1, 2]),  # LabelEncoded: C=0, Q=1, S=2
            nullable=False,
        ),
    },
    coerce=True,
    strict=False,
    name="ProcessedFeatureSchema",
)


# ════════════════════════════════════════════════════════════════════
# 3. Inference Raw Titanic Schema
#    Validates inference data — same as RawTitanicSchema but with
#    "Survived" marked as not required since test/inference datasets
#    don't include the target variable.
# ════════════════════════════════════════════════════════════════════

InferenceRawTitanicSchema = DataFrameSchema(
    columns={
        "PassengerId": Column(
            int,
            checks=Check.greater_than(0),
            nullable=False,
            unique=True,
            description="Unique passenger identifier — must be positive integer",
        ),
        "Survived": Column(
            int,
            checks=Check.isin([0, 1]),
            nullable=False,
            description="Survival label: 0 = did not survive, 1 = survived",
            required=False,  # Inference data doesn't include the target variable
        ),
        "Pclass": Column(
            int,
            checks=Check.isin([1, 2, 3]),
            nullable=False,
            description="Ticket class: 1st, 2nd, or 3rd",
        ),
        "Name": Column(
            str,
            checks=Check(lambda s: s.str.len() > 0, element_wise=False),
            nullable=False,
            description="Passenger full name — used for title extraction",
        ),
        "Sex": Column(
            str,
            checks=Check.isin(["male", "female"]),
            nullable=False,
            description="Passenger sex",
        ),
        "Age": Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(120),
            ],
            nullable=True,   # ~20% missing in real dataset
            description="Passenger age in years — nullable, imputed with median",
        ),
        "SibSp": Column(
            int,
            checks=Check.greater_than_or_equal_to(0),
            nullable=False,
            description="Number of siblings/spouses aboard",
        ),
        "Parch": Column(
            int,
            checks=Check.greater_than_or_equal_to(0),
            nullable=False,
            description="Number of parents/children aboard",
        ),
        "Ticket": Column(
            str,
            nullable=False,
            description="Ticket number (not used as feature, kept for traceability)",
        ),
        "Fare": Column(
            float,
            checks=Check.greater_than_or_equal_to(0),
            nullable=False,
            description="Passenger fare — must be non-negative",
        ),
        "Cabin": Column(
            str,
            nullable=True,   # ~77% missing — dropped in feature engineering
            description="Cabin number — nullable, dropped during processing",
            required=False,  # Some datasets omit this column entirely
        ),
        "Embarked": Column(
            str,
            checks=Check.isin(["S", "C", "Q"]),
            nullable=True,   # 2 rows missing in Titanic dataset
            description="Port of embarkation: S=Southampton, C=Cherbourg, Q=Queenstown",
        ),
    },
    checks=[
        # Dataset-level checks (applied to the whole DataFrame)
        Check(
            lambda df: len(df) >= 1,   # At least 1 row for inference
            error="Dataset has no rows — empty inference dataset",
        ),
    ],
    coerce=True,        # Attempt type coercion before raising errors
    strict=False,       # Allow extra columns (e.g. index columns from exports)
    name="InferenceRawTitanicSchema",
)


# ════════════════════════════════════════════════════════════════════
# Validation helpers
# ════════════════════════════════════════════════════════════════════

class SchemaValidationResult:
    """Structured result of a Pandera validation — JSON-serialisable."""

    def __init__(self, passed: bool, schema_name: str, errors: list[dict] = None):
        self.passed      = passed
        self.schema_name = schema_name
        self.errors      = errors or []

    def to_dict(self) -> dict:
        return {
            "passed":      self.passed,
            "schema_name": self.schema_name,
            "error_count": len(self.errors),
            "errors":      self.errors,
        }

    def raise_if_failed(self):
        if not self.passed:
            msg = f"Schema '{self.schema_name}' validation failed with {len(self.errors)} error(s)"
            raise ValueError(msg)


def validate_raw(df: pd.DataFrame, raise_on_error: bool = True) -> SchemaValidationResult:
    """
    Validate a raw Titanic DataFrame against RawTitanicSchema.

    Parameters
    ----------
    df              : DataFrame loaded directly from CSV
    raise_on_error  : If True, raises ValueError on failure (fails the pipeline).
                      If False, returns the result for the caller to handle.

    Returns
    -------
    SchemaValidationResult with pass/fail + per-column error details
    """
    try:
        RawTitanicSchema.validate(df, lazy=True)   # lazy=True collects ALL errors
        logger.info("✅ RawTitanicSchema validation passed (%d rows)", len(df))
        return SchemaValidationResult(passed=True, schema_name="RawTitanicSchema")

    except pa.errors.SchemaErrors as exc:
        errors = _parse_pandera_errors(exc)
        logger.error(
            "❌ RawTitanicSchema validation FAILED — %d error(s):\n%s",
            len(errors),
            json.dumps(errors, indent=2),
        )
        result = SchemaValidationResult(
            passed=False,
            schema_name="RawTitanicSchema",
            errors=errors,
        )
        if raise_on_error:
            result.raise_if_failed()
        return result


def validate_raw_inference(df: pd.DataFrame, raise_on_error: bool = True) -> SchemaValidationResult:
    """
    Validate raw inference/test data against InferenceRawTitanicSchema.
    
    Identical to validate_raw() but uses InferenceRawTitanicSchema which does NOT
    require the "Survived" column (the target variable is absent in test datasets).

    Parameters
    ----------
    df              : DataFrame loaded directly from CSV (inference/test data)
    raise_on_error  : If True, raises ValueError on failure (fails the pipeline).
                      If False, returns the result for the caller to handle.

    Returns
    -------
    SchemaValidationResult with pass/fail + per-column error details
    """
    try:
        InferenceRawTitanicSchema.validate(df, lazy=True)
        logger.info("✅ InferenceRawTitanicSchema validation passed (%d rows)", len(df))
        return SchemaValidationResult(passed=True, schema_name="InferenceRawTitanicSchema")

    except pa.errors.SchemaErrors as exc:
        errors = _parse_pandera_errors(exc)
        logger.error(
            "❌ InferenceRawTitanicSchema validation FAILED — %d error(s):\n%s",
            len(errors),
            json.dumps(errors, indent=2),
        )
        result = SchemaValidationResult(
            passed=False,
            schema_name="InferenceRawTitanicSchema",
            errors=errors,
        )
        if raise_on_error:
            result.raise_if_failed()
        return result


def validate_processed(df: pd.DataFrame, raise_on_error: bool = False) -> SchemaValidationResult:
    """
    Validate the engineered feature DataFrame against ProcessedFeatureSchema.

    raise_on_error defaults to False here — a schema mismatch in the
    processed frame usually points to a DataProcessor bug and is logged
    as a warning rather than a hard pipeline failure.
    """
    try:
        ProcessedFeatureSchema.validate(df, lazy=True)
        logger.info("✅ ProcessedFeatureSchema validation passed")
        return SchemaValidationResult(passed=True, schema_name="ProcessedFeatureSchema")

    except pa.errors.SchemaErrors as exc:
        errors = _parse_pandera_errors(exc)
        logger.warning(
            "⚠️  ProcessedFeatureSchema validation found %d issue(s):\n%s",
            len(errors),
            json.dumps(errors, indent=2),
        )
        result = SchemaValidationResult(
            passed=False,
            schema_name="ProcessedFeatureSchema",
            errors=errors,
        )
        if raise_on_error:
            result.raise_if_failed()
        return result


def _parse_pandera_errors(exc: "pa.errors.SchemaErrors") -> list[dict]:
    """Convert Pandera's SchemaErrors into a clean list of dicts."""
    errors = []
    try:
        # failure_cases is a DataFrame with schema_context, column, check, etc.
        for _, row in exc.failure_cases.iterrows():
            errors.append({
                "schema_context": str(row.get("schema_context", "")),
                "column":         str(row.get("column", "")),
                "check":          str(row.get("check", "")),
                "check_number":   str(row.get("check_number", "")),
                "failure_case":   str(row.get("failure_case", "")),
                "index":          str(row.get("index", "")),
            })
    except Exception:
        errors = [{"raw_error": str(exc)}]
    return errors
