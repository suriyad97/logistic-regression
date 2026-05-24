"""
train.py  [Optuna edition]
───────────────────────────
Training script for the Titanic Logistic Regression model.
Supports optional Optuna hyperparameter tuning with full MLflow integration.

Two modes:
  ① Default (--tune not set)
    Uses fixed hyperparameters (or those passed via CLI).
    Fast — suitable for quick retraining triggered by concept drift.

  ② Tuning (--tune flag)
    Runs an Optuna TPE study across the full hyperparameter search space.
    Each trial is logged as a nested MLflow child run.
    Best parameters are used to train the final model.
    Full study summary saved to model/optuna_study.json.

Search space:
  C             — regularisation strength [0.001 → 10, log-scale]
  penalty       — l2 / l1 (with appropriate solver)
  solver        — lbfgs (l2) / liblinear (l1 + l2) / saga (l1 + l2)
  class_weight  — balanced / None
  max_iter      — [200, 2000]
"""

import os
import json
import logging
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)

# ── MLflow ────────────────────────────────────────────────────────
try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

# ── Optuna ────────────────────────────────────────────────────────
try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
    # Suppress the per-trial INFO logs from Optuna — keep the console clean
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    HAS_OPTUNA = False

from data_processing import DataProcessor  # noqa: E402

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
# Metric helpers
# ════════════════════════════════════════════════════════════════════

def calculate_metrics(y_true, y_pred, y_pred_proba=None) -> dict:
    """Compute a full suite of classification metrics."""
    metrics = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score":  float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_pred_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba))
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return metrics


# ════════════════════════════════════════════════════════════════════
# Optuna hyperparameter tuning
# ════════════════════════════════════════════════════════════════════

# Compatible (penalty, solver) pairs for LogisticRegression
_SOLVER_PENALTY_MAP = {
    "l2":   ["lbfgs", "liblinear", "saga"],
    "l1":   ["liblinear", "saga"],
    "none": ["lbfgs", "saga"],
}


def _suggest_params(trial: "optuna.Trial") -> dict:
    """Define the Optuna hyperparameter search space."""
    combo = trial.suggest_categorical(
        "solver_penalty",
        [
            "lbfgs_l2", "lbfgs_none",
            "liblinear_l1", "liblinear_l2",
            "saga_l1", "saga_l2", "saga_none"
        ]
    )
    solver, penalty = combo.split("_")
    return {
        "C":            trial.suggest_float("C", 1e-3, 10.0, log=True),
        "solver":       solver,
        "penalty":      penalty if penalty != "none" else None,
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
        "max_iter":     trial.suggest_int("max_iter", 200, 2000, step=200),
        "random_state": 42,
    }


def _make_objective(X_train, y_train, X_val, y_val, parent_run_id: str | None):
    """Factory: returns an Optuna objective closure over the given data splits."""

    def objective(trial: "optuna.Trial") -> float:
        params = _suggest_params(trial)

        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred  = model.predict(X_val)

        roc_auc  = float(roc_auc_score(y_val, y_proba))
        f1       = float(f1_score(y_val, y_pred, zero_division=0))

        # Store both for later inspection
        trial.set_user_attr("val_roc_auc", roc_auc)
        trial.set_user_attr("val_f1_score", f1)

        # ── Log each trial as a nested MLflow child run ───────────
        if HAS_MLFLOW and parent_run_id:
            with mlflow.start_run(
                run_name=f"trial_{trial.number}",
                nested=True,
                tags={"optuna_trial": str(trial.number)},
            ):
                mlflow.log_params(params)
                mlflow.log_metric("val_roc_auc", roc_auc)
                mlflow.log_metric("val_f1_score", f1)

        logger.debug(
            "Trial %3d — ROC-AUC: %.4f  F1: %.4f  params: %s",
            trial.number, roc_auc, f1, params,
        )
        return roc_auc   # Maximise ROC-AUC

    return objective


def tune_hyperparameters(
    X_train, y_train,
    X_val,   y_val,
    n_trials: int = 50,
    parent_run_id: str | None = None,
    random_state: int = 42,
) -> dict:
    """
    Run an Optuna TPE study to find the best LogisticRegression hyperparameters.

    Returns
    -------
    dict  Best hyperparameters found by the study.
    """
    if not HAS_OPTUNA:
        logger.warning(
            "Optuna not installed. Using default hyperparameters. "
            "Install with: pip install optuna"
        )
        return {"C": 1.0, "solver": "lbfgs", "penalty": "l2",
                "class_weight": "balanced", "max_iter": 1000}

    logger.info("Starting Optuna study — %d trials, optimising ROC-AUC", n_trials)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=random_state),
        study_name="titanic-logistic-regression",
    )

    objective = _make_objective(X_train, y_train, X_val, y_val, parent_run_id)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    logger.info(
        "Optuna best — trial #%d  ROC-AUC: %.4f  F1: %.4f",
        best.number,
        best.value,
        best.user_attrs.get("val_f1_score", 0.0),
    )
    logger.info("Best params: %s", best.params)

    return best.params


def _save_study_summary(study: "optuna.Study", output_dir: str) -> None:
    """Persist the full Optuna study results as JSON for audit / visualisation."""
    if not HAS_OPTUNA:
        return

    trials_data = []
    for t in study.trials:
        trials_data.append({
            "number":    t.number,
            "value":     t.value,         # val_roc_auc
            "params":    t.params,
            "state":     str(t.state),
            "user_attrs": t.user_attrs,
        })

    summary = {
        "best_trial":   study.best_trial.number,
        "best_value":   study.best_trial.value,
        "best_params":  study.best_params,
        "n_trials":     len(study.trials),
        "trials":       trials_data,
    }

    path = Path(output_dir) / "optuna_study.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Optuna study summary saved to %s", path)


# ════════════════════════════════════════════════════════════════════
# Main training function
# ════════════════════════════════════════════════════════════════════

def train_model(
    data_path:     str,
    output_dir:    str,
    test_size:     float = 0.2,
    max_iter:      int   = 1000,
    random_state:  int   = 42,
    tune:          bool  = False,
    n_trials:      int   = 50,
    explain:       bool  = True,
):
    """
    Full training pipeline.

    Parameters
    ----------
    data_path    : Path to training CSV
    output_dir   : Directory to save model + artifacts
    test_size    : Fraction held out for test set
    max_iter     : Max iterations for solver (ignored when tune=True)
    random_state : Reproducibility seed
    tune         : Run Optuna hyperparameter search before training
    n_trials     : Number of Optuna trials (used only when tune=True)
    """
    logger.info("Starting training  (tune=%s)", tune)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Start outer MLflow run ────────────────────────────────────
    parent_run_id = None
    if HAS_MLFLOW:
        mlflow.set_experiment("titanic-logistic-regression")
        run = mlflow.start_run(run_name="training-run" + ("-tuned" if tune else ""))
        parent_run_id = run.info.run_id
        logger.info("MLflow run started: %s", parent_run_id)

    try:
        # ── 1. Data processing ────────────────────────────────────
        logger.info("Processing data from %s", data_path)
        processor = DataProcessor(random_state=random_state)
        X_train, X_test, y_train, y_test = processor.process_data(
            data_path, is_training=True, test_size=test_size
        )
        logger.info("Train: %s  |  Test: %s", X_train.shape, X_test.shape)

        if HAS_MLFLOW:
            mlflow.log_params({
                "test_size":        test_size,
                "random_state":     random_state,
                "training_samples": X_train.shape[0],
                "test_samples":     X_test.shape[0],
                "feature_count":    X_train.shape[1],
                "tuning_enabled":   tune,
                "n_trials":         n_trials if tune else 0,
            })

        # ── 2. Hyperparameter selection ───────────────────────────
        if tune:
            logger.info("── Optuna hyperparameter tuning ─────────────────")
            # Use a 20% validation split inside the training fold
            from sklearn.model_selection import train_test_split
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train,
                test_size=0.2,
                random_state=random_state,
                stratify=y_train,
            )

            # Run the Optuna study
            study = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(seed=random_state),
                study_name="titanic-logistic-regression",
            ) if HAS_OPTUNA else None

            if study is not None:
                objective = _make_objective(X_tr, y_tr, X_val, y_val, parent_run_id)
                study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
                best_params = study.best_params
                _save_study_summary(study, output_dir)
            else:
                best_params = {}

            final_params = {
                "C":            best_params.get("C", 1.0),
                "penalty":      best_params.get("penalty", "l2"),
                "solver":       best_params.get("solver", "lbfgs"),
                "class_weight": best_params.get("class_weight", "balanced"),
                "max_iter":     best_params.get("max_iter", max_iter),
                "random_state": random_state,
            }

            if HAS_MLFLOW:
                mlflow.log_params({f"best_{k}": v for k, v in final_params.items()})
                mlflow.log_metric("optuna_best_val_roc_auc", study.best_value if study else 0)
                if study:
                    mlflow.log_artifact(str(Path(output_dir) / "optuna_study.json"))

        else:
            final_params = {
                "C":            1.0,
                "penalty":      "l2",
                "solver":       "lbfgs",
                "class_weight": "balanced",
                "max_iter":     max_iter,
                "random_state": random_state,
            }
            if HAS_MLFLOW:
                mlflow.log_params(final_params)

        logger.info("Final model params: %s", final_params)

        # ── 3. Train final model ──────────────────────────────────
        logger.info("Training final LogisticRegression model…")
        model = LogisticRegression(**final_params)
        model.fit(X_train, y_train)
        logger.info("Training complete")

        # ── 4. Evaluate ───────────────────────────────────────────
        y_train_pred  = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)[:, 1]
        train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba)

        y_test_pred  = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]
        test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)

        logger.info("Train metrics: acc=%.4f  roc_auc=%.4f", train_metrics["accuracy"], train_metrics.get("roc_auc", 0))
        logger.info("Test  metrics: acc=%.4f  roc_auc=%.4f", test_metrics["accuracy"],  test_metrics.get("roc_auc", 0))

        if HAS_MLFLOW:
            for k, v in train_metrics.items():
                if k != "confusion_matrix":
                    mlflow.log_metric(f"train_{k}", v)
            for k, v in test_metrics.items():
                if k != "confusion_matrix":
                    mlflow.log_metric(f"test_{k}", v)
            mlflow.sklearn.log_model(model, "model")

        # ── 5. Save artefacts ─────────────────────────────────────
        joblib.dump(model, Path(output_dir) / "model.pkl")
        processor.save_preprocessor(str(Path(output_dir) / "preprocessor.pkl"))

        metrics_data = {
            "training_metrics": train_metrics,
            "test_metrics":     test_metrics,
            "model_params":     final_params,
            "data_info": {
                "training_samples": int(X_train.shape[0]),
                "test_samples":     int(X_test.shape[0]),
                "features":         processor.feature_names,
                "feature_count":    int(X_train.shape[1]),
            },
            "tuning": {
                "enabled":    tune,
                "n_trials":   n_trials if tune else 0,
                "tool":       "optuna" if (tune and HAS_OPTUNA) else "none",
            },
        }
        with open(Path(output_dir) / "metrics.json", "w") as f:
            json.dump(metrics_data, f, indent=2)

        # ── 6. Reports ────────────────────────────────────────────
        logger.info("\nClassification Report (Test Set):")
        print(classification_report(
            y_test, y_test_pred,
            target_names=["Did not survive", "Survived"],
        ))

        fi_df = pd.DataFrame({
            "feature":     processor.feature_names,
            "coefficient": model.coef_[0],
        }).sort_values("coefficient", key=abs, ascending=False)
        logger.info("\nFeature Importance (abs coefficient):\n%s", fi_df.to_string(index=False))

        # ── 7. SHAP + LIME Explainability ─────────────────────────
        if explain:
            try:
                from explain import ModelExplainer
                reports_dir = Path(output_dir) / "reports"
                X_train_sc = processor.scale_features(
                    pd.DataFrame(X_train, columns=processor.feature_names), is_training=False
                )
                X_test_sc = processor.scale_features(
                    pd.DataFrame(X_test, columns=processor.feature_names), is_training=False
                )
                model_explainer = ModelExplainer(model, processor, processor.feature_names)
                model_explainer.run_all(
                    X_train_scaled=X_train_sc,
                    X_test_scaled=X_test_sc,
                    X_test_df=pd.DataFrame(X_test, columns=processor.feature_names),
                    output_dir=reports_dir,
                    local_indices=list(range(min(5, len(X_test_sc)))),
                )
                logger.info("Explainability reports saved to %s", reports_dir)
                if HAS_MLFLOW:
                    mlflow.log_artifacts(str(reports_dir), artifact_path="explainability")
            except Exception as exc:
                logger.warning("Explainability failed (non-fatal): %s", exc)

        if HAS_MLFLOW:
            mlflow.end_run()
            logger.info("MLflow run ended")

        logger.info("Training completed ✅")
        return model, processor, metrics_data

    except Exception as exc:
        logger.error("Training failed: %s", exc, exc_info=True)
        if HAS_MLFLOW:
            mlflow.end_run(status="FAILED")
        raise


# ════════════════════════════════════════════════════════════════════
# CLI entrypoint
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train Titanic LogisticRegression — with optional Optuna tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data",         default="../data/titanic.csv",  help="Path to training CSV")
    parser.add_argument("--output",       default="../model",             help="Output directory")
    parser.add_argument("--test-size",    type=float, default=0.2,        help="Test set fraction")
    parser.add_argument("--max-iter",     type=int,   default=1000,       help="Max solver iterations (non-tuning mode)")
    parser.add_argument("--random-state", type=int,   default=42,         help="Random seed")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna hyperparameter search before training",
    )
    parser.add_argument(
        "--n-trials",
        type=int, default=50,
        help="Number of Optuna trials (used only with --tune)",
    )
    parser.add_argument(
        "--explain",
        action="store_true", default=True,
        help="Run SHAP + LIME explainability after training (default: on)",
    )
    parser.add_argument(
        "--no-explain",
        dest="explain", action="store_false",
        help="Skip SHAP + LIME explainability (faster)",
    )
    args = parser.parse_args()

    train_model(
        data_path=os.path.abspath(args.data),
        output_dir=os.path.abspath(args.output),
        test_size=args.test_size,
        max_iter=args.max_iter,
        random_state=args.random_state,
        tune=args.tune,
        n_trials=args.n_trials,
        explain=args.explain,
    )


if __name__ == "__main__":
    main()
