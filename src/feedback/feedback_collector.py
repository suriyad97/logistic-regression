"""
feedback_collector.py
──────────────────────
Ground Truth Feedback Loop — The Missing Link in MLOps

WHY THIS EXISTS:
  In concept drift detection we compare model predictions against
  actual outcomes (ground truth). For Titanic this is obvious —
  we know who survived. In real production ML this is the hard part:
  outcomes only become available AFTER time passes (e.g., loan default
  after 6 months, churn after 30 days).

  This module simulates a realistic feedback loop by:
  1. Storing predictions with a unique ID + timestamp when inference runs
  2. Accepting ground truth labels when they arrive (via API call, file drop, etc.)
  3. Joining predictions ↔ ground truth → the matched set feeds concept drift detection
  4. Tracking feedback coverage so you know how much of your inference is validated

HOW IT FITS IN YOUR PIPELINE:
  CD Pipeline  → inference runs → predictions logged with FeedbackStore.log_predictions()
  Ground Truth → arrives later  → FeedbackStore.submit_ground_truth() is called
  CM Pipeline  → FeedbackStore.get_matched_data() feeds detect_concept_drift.py

For Titanic specifically:
  We can simulate ground truth by withholding part of the dataset.
  In production you'd connect this to your labeling tool, CRM, or database.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


class FeedbackStore:
    """
    File-based feedback store for tracking predictions and ground truth.

    In production, replace the CSV files with:
      - Azure SQL / Cosmos DB for low-latency joins
      - Azure Data Lake Storage for large-scale batch feedback
      - Azure ML Data Assets for version control
    """

    PREDICTIONS_FILE = "ground_truth.csv"
    MATCHED_FILE     = "matched_ground_truth.csv"
    COVERAGE_FILE    = "feedback_coverage.json"

    def __init__(self, store_dir: str = "data/feedback"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_path = self.store_dir / self.PREDICTIONS_FILE
        self.matched_path     = self.store_dir / self.MATCHED_FILE
        self.coverage_path    = self.store_dir / self.COVERAGE_FILE

    # ── Logging predictions ───────────────────────────────────────
    def log_predictions(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Attach a unique inference_id to each prediction row and persist.
        Call this immediately after batch inference.

        Returns the DataFrame augmented with 'inference_id'.
        """
        df = predictions_df.copy()
        df["inference_id"]   = [str(uuid.uuid4()) for _ in range(len(df))]
        df["logged_at"]      = datetime.now(timezone.utc).isoformat()
        df["ground_truth"]   = None  # will be filled later
        df["feedback_received"] = False

        if self.predictions_path.exists():
            existing = pd.read_csv(self.predictions_path)
            df = pd.concat([existing, df], ignore_index=True)

        df.to_csv(self.predictions_path, index=False)
        logger.info(
            "Logged %d predictions to %s (total=%d)",
            len(predictions_df), self.predictions_path, len(df)
        )
        return df

    # ── Submitting ground truth ───────────────────────────────────
    def submit_ground_truth(
        self,
        labels: dict[str, int],
        match_column: str = "PassengerId",
    ) -> int:
        """
        Accept ground truth labels and join them to stored predictions.

        Args:
            labels: dict mapping match_column value → actual label (0 or 1)
            match_column: column to join on (e.g. PassengerId)

        Returns:
            Number of predictions updated with ground truth
        """
        if not self.predictions_path.exists():
            logger.warning("No predictions file found at %s", self.predictions_path)
            return 0

        df = pd.read_csv(self.predictions_path)
        updated = 0

        for key, label in labels.items():
            mask = df[match_column].astype(str) == str(key)
            if mask.any():
                df.loc[mask, "ground_truth"]      = int(label)
                df.loc[mask, "feedback_received"]  = True
                df.loc[mask, "feedback_received_at"] = datetime.now(timezone.utc).isoformat()
                updated += int(mask.sum())

        df.to_csv(self.predictions_path, index=False)
        logger.info("Updated ground truth for %d rows", updated)
        self._update_coverage(df)
        return updated

    def submit_ground_truth_from_csv(
        self,
        csv_path: str,
        id_column: str = "PassengerId",
        label_column: str = "Survived",
    ) -> int:
        """
        Convenience method: load a CSV with IDs + labels and submit them all.
        Useful for batch feedback ingestion from a labeling tool or data export.
        """
        gt_df = pd.read_csv(csv_path)
        labels = dict(zip(gt_df[id_column].astype(str), gt_df[label_column]))
        logger.info("Submitting %d ground truth labels from %s", len(labels), csv_path)
        return self.submit_ground_truth(labels)

    # ── Retrieving matched data ───────────────────────────────────
    def get_matched_data(
        self,
        min_coverage: float = 0.0,
        save: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Return rows where ground truth has been received.
        This is the input to detect_concept_drift.py.

        Args:
            min_coverage: If coverage is below this fraction, return None
                          (not enough data to detect drift reliably)
        """
        if not self.predictions_path.exists():
            logger.warning("No predictions file — cannot get matched data")
            return None

        df = pd.read_csv(self.predictions_path)
        matched = df[df["feedback_received"] == True].copy()  # noqa: E712

        coverage = len(matched) / max(len(df), 1)
        logger.info(
            "Feedback coverage: %d / %d (%.1f%%)",
            len(matched), len(df), coverage * 100,
        )

        if coverage < min_coverage:
            logger.warning(
                "Coverage %.1f%% is below minimum %.1f%%. Concept drift not computed.",
                coverage * 100, min_coverage * 100,
            )
            return None

        # Rename for downstream compatibility with detect_concept_drift.py
        matched = matched.rename(columns={"ground_truth": "actual_label"})

        if save:
            matched.to_csv(self.matched_path, index=False)
            logger.info("Matched data saved to %s", self.matched_path)

        return matched

    # ── Coverage reporting ────────────────────────────────────────
    def _update_coverage(self, df: pd.DataFrame) -> None:
        total     = len(df)
        received  = int(df["feedback_received"].sum())
        coverage  = received / max(total, 1)

        report = {
            "timestamp":        datetime.now(timezone.utc).isoformat(),
            "total_predictions": total,
            "feedback_received": received,
            "coverage_fraction": round(coverage, 4),
            "coverage_percent":  round(coverage * 100, 2),
        }
        with open(self.coverage_path, "w") as f:
            json.dump(report, f, indent=2)

    def get_coverage_report(self) -> dict:
        if self.coverage_path.exists():
            with open(self.coverage_path) as f:
                return json.load(f)
        return {"coverage_fraction": 0, "message": "No predictions logged yet"}


# ── Titanic-specific simulation ───────────────────────────────────
def simulate_ground_truth_titanic(
    predictions_csv: str,
    original_data_csv: str,
    store: FeedbackStore,
    fraction: float = 0.8,
) -> int:
    """
    Simulate a feedback loop for Titanic by using the known Survived labels.

    In production this is replaced by your actual labeling system.
    Here it lets you test the full pipeline end-to-end:
      - fraction=0.8 → 80% of predictions get ground truth immediately
      - The remaining 20% simulate "not yet labelled"

    Args:
        predictions_csv: Path to predictions.csv from inference
        original_data_csv: Path to titanic.csv (has Survived column)
        store: FeedbackStore instance
        fraction: Fraction of predictions to label (simulate delay)
    """
    import numpy as np

    preds    = pd.read_csv(predictions_csv)
    original = pd.read_csv(original_data_csv)

    if "PassengerId" not in preds.columns or "PassengerId" not in original.columns:
        logger.warning("PassengerId column missing — cannot simulate feedback")
        return 0

    # Log the predictions first
    store.log_predictions(preds)

    # Submit ground truth for a fraction of rows (simulate delayed labeling)
    original_with_gt = original[["PassengerId", "Survived"]].dropna()
    sample = original_with_gt.sample(frac=fraction, random_state=42)
    labels = dict(zip(sample["PassengerId"].astype(str), sample["Survived"].astype(int)))

    updated = store.submit_ground_truth(labels)
    logger.info(
        "Simulated feedback for %.0f%% of predictions (%d rows updated)",
        fraction * 100, updated,
    )
    return updated


# ── CLI ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Feedback loop manager — submit or retrieve ground truth"
    )
    sub = parser.add_subparsers(dest="command")

    # simulate command
    sim = sub.add_parser("simulate", help="Simulate ground truth for Titanic")
    sim.add_argument("--predictions", required=True, help="predictions.csv path")
    sim.add_argument("--original-data", required=True, help="titanic.csv path")
    sim.add_argument("--store-dir", default="data/feedback")
    sim.add_argument("--fraction", type=float, default=0.8)

    # submit command
    sub_cmd = sub.add_parser("submit", help="Submit ground truth from a CSV file")
    sub_cmd.add_argument("--labels-csv", required=True)
    sub_cmd.add_argument("--id-column", default="PassengerId")
    sub_cmd.add_argument("--label-column", default="Survived")
    sub_cmd.add_argument("--store-dir", default="data/feedback")

    # status command
    status = sub.add_parser("status", help="Show feedback coverage")
    status.add_argument("--store-dir", default="data/feedback")

    args = parser.parse_args()

    if args.command == "simulate":
        store = FeedbackStore(args.store_dir)
        simulate_ground_truth_titanic(
            args.predictions, args.original_data, store, args.fraction
        )
        print(json.dumps(store.get_coverage_report(), indent=2))

    elif args.command == "submit":
        store = FeedbackStore(args.store_dir)
        n = store.submit_ground_truth_from_csv(
            args.labels_csv, args.id_column, args.label_column
        )
        print(f"Updated {n} rows with ground truth.")

    elif args.command == "status":
        store = FeedbackStore(args.store_dir)
        print(json.dumps(store.get_coverage_report(), indent=2))

    else:
        parser.print_help()
