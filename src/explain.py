"""
explain.py
───────────
Model explainability using SHAP and LIME.

Two complementary approaches:

SHAP (SHapley Additive exPlanations)
  ─ Global: ranks which features drive survival predictions across ALL passengers
  ─ Local:  waterfall chart showing why a SPECIFIC passenger was predicted the way they were
  ─ Uses LinearExplainer (exact, fast for logistic regression)
  ─ SHAP values are mathematically grounded in game theory

LIME (Local Interpretable Model-agnostic Explanations)
  ─ Local only: builds a simple linear model around a single prediction
  ─ Answers: "What would change this passenger's prediction?"
  ─ Complements SHAP — gives a human-friendly "if-then" rule explanation

Outputs (per run):
  reports/shap_summary.png         ─ beeswarm plot of feature importance
  reports/shap_waterfall_<n>.png   ─ per-passenger waterfall chart
  reports/shap_values.json         ─ raw SHAP values for all test samples
  reports/lime_explanation_<n>.html ─ per-passenger LIME report
  reports/feature_importance.json  ─ ranked feature importance from SHAP

Usage (standalone):
  python src/explain.py \\
    --model model/model.pkl \\
    --preprocessor model/preprocessor.pkl \\
    --data data/raw_data/titanic.csv \\
    --output reports/

Usage (import):
  from explain import ModelExplainer
  explainer = ModelExplainer(model, processor, feature_names)
  explainer.explain_global(X_test)
  explainer.explain_local(X_test, index=0)
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server/AML compute
import matplotlib.pyplot as plt

# ── Optional MLflow ───────────────────────────────────────────────
try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

# ── SHAP ──────────────────────────────────────────────────────────
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# ── LIME ──────────────────────────────────────────────────────────
try:
    import lime
    import lime.lime_tabular
    HAS_LIME = True
except Exception:
    HAS_LIME = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


class ModelExplainer:
    """
    Wrapper that runs SHAP and LIME explanations on a trained
    LogisticRegression + DataProcessor pipeline.
    """

    def __init__(self, model, processor, feature_names: list[str]):
        self.model         = model
        self.processor     = processor
        self.feature_names = feature_names

    # ── SHAP ──────────────────────────────────────────────────────

    def explain_global_shap(
        self,
        X_train_scaled: np.ndarray,
        X_test_scaled:  np.ndarray,
        output_dir: Path,
        top_n: int = 10,
    ) -> dict:
        """
        Global SHAP explanation — ranks features by their average absolute
        contribution to model predictions across the entire test set.

        For LogisticRegression we use LinearExplainer which is exact (not sampled)
        and very fast even for thousands of rows.

        Returns dict of feature → mean |SHAP value| for MLflow logging.
        """
        if not HAS_SHAP:
            logger.warning("SHAP not installed. Run: pip install shap")
            return {}

        logger.info("Computing SHAP values (LinearExplainer)…")

        # LinearExplainer takes a background dataset (training) + explains test
        explainer   = shap.LinearExplainer(self.model, X_train_scaled)
        shap_values = explainer.shap_values(X_test_scaled)

        # shap_values shape: (n_samples, n_features)
        # For binary classification LinearExplainer returns values for class 1 (survived)
        if isinstance(shap_values, list):
            sv = shap_values[1]   # class 1 = survived
        else:
            sv = shap_values

        mean_abs_shap = np.abs(sv).mean(axis=0)
        feature_importance = dict(zip(self.feature_names, mean_abs_shap.tolist()))
        ranked = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

        # ── Beeswarm summary plot ─────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            sv,
            X_test_scaled,
            feature_names=self.feature_names,
            show=False,
            plot_type="dot",
            max_display=top_n,
        )
        plt.title("SHAP Feature Importance — Titanic Survival Prediction", pad=15)
        plt.tight_layout()
        summary_path = output_dir / "shap_summary.png"
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("SHAP summary plot saved to %s", summary_path)

        # ── Bar chart (cleaner for reports) ──────────────────────
        fig, ax = plt.subplots(figsize=(9, 6))
        features = list(ranked.keys())[:top_n]
        values   = list(ranked.values())[:top_n]
        colors   = ["#ef4444" if v > 0 else "#3b82f6" for v in values]
        ax.barh(features[::-1], values[::-1], color=colors[::-1])
        ax.set_xlabel("Mean |SHAP Value| — average impact on model output", fontsize=11)
        ax.set_title("Global Feature Importance (SHAP)", fontsize=13, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        bar_path = output_dir / "shap_importance_bar.png"
        plt.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("SHAP bar chart saved to %s", bar_path)

        # ── Save raw values ───────────────────────────────────────
        shap_json = {
            "feature_importance_ranked": ranked,
            "top_feature":               features[0] if features else "",
            "n_samples_explained":       len(X_test_scaled),
        }
        with open(output_dir / "feature_importance.json", "w") as f:
            json.dump(shap_json, f, indent=2)

        # ── Log to MLflow ─────────────────────────────────────────
        if HAS_MLFLOW:
            for feat, val in ranked.items():
                mlflow.log_metric(f"shap_{feat}", round(val, 6))
            mlflow.log_artifact(str(summary_path))
            mlflow.log_artifact(str(bar_path))

        logger.info("Top features by SHAP: %s", list(ranked.items())[:5])
        return ranked

    def explain_local_shap(
        self,
        X_train_scaled: np.ndarray,
        X_test_scaled:  np.ndarray,
        X_test_df:      pd.DataFrame,
        output_dir:     Path,
        indices:        list[int] = None,
    ) -> None:
        """
        Local SHAP explanation — waterfall chart for individual passengers.
        Shows exactly which features pushed this passenger toward/away from survival.
        """
        if not HAS_SHAP:
            return

        if indices is None:
            # Explain first 3 test passengers by default
            indices = list(range(min(3, len(X_test_scaled))))

        explainer   = shap.LinearExplainer(self.model, X_train_scaled)
        shap_values = explainer.shap_values(X_test_scaled)

        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

        for idx in indices:
            pred       = self.model.predict(X_test_scaled[idx:idx+1])[0]
            prob       = self.model.predict_proba(X_test_scaled[idx:idx+1])[0][1]
            label      = "Survived" if pred == 1 else "Did not survive"

            fig, ax = plt.subplots(figsize=(10, 6))

            # Waterfall: positive = pushes toward survival, negative = against
            feat_shap  = sv[idx]
            sorted_idx = np.argsort(np.abs(feat_shap))[::-1][:8]
            feat_names = [self.feature_names[i] for i in sorted_idx]
            feat_vals  = [feat_shap[i] for i in sorted_idx]

            colors = ["#22c55e" if v > 0 else "#ef4444" for v in feat_vals]
            ax.barh(feat_names[::-1], feat_vals[::-1], color=colors[::-1])
            ax.axvline(0, color="#374151", linewidth=0.8, linestyle="--")
            ax.set_xlabel("SHAP Value  (positive = pushes toward Survived)", fontsize=10)
            ax.set_title(
                f"Passenger #{idx} — Predicted: {label} ({prob:.1%} survival probability)",
                fontsize=12, fontweight="bold",
            )
            ax.spines[["top", "right"]].set_visible(False)

            # Annotate feature values from the original DataFrame
            if idx < len(X_test_df):
                row = X_test_df.iloc[idx]
                for i, (fname, fval) in enumerate(zip(feat_names[::-1], feat_vals[::-1])):
                    raw_val = row.get(fname, "")
                    ax.text(
                        fval + (0.002 if fval >= 0 else -0.002),
                        i,
                        f" {raw_val}",
                        va="center", ha="left" if fval >= 0 else "right",
                        fontsize=8, color="#6b7280",
                    )

            plt.tight_layout()
            path = output_dir / f"shap_waterfall_{idx}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info("SHAP waterfall for passenger %d saved to %s", idx, path)

            if HAS_MLFLOW:
                mlflow.log_artifact(str(path))

    # ── LIME ──────────────────────────────────────────────────────

    def explain_local_lime(
        self,
        X_train_scaled: np.ndarray,
        X_test_scaled:  np.ndarray,
        X_test_df:      pd.DataFrame,
        output_dir:     Path,
        indices:        list[int] = None,
        num_features:   int = 8,
    ) -> None:
        """
        Local LIME explanation — builds a local linear surrogate model
        around each prediction and shows which features drive it.

        LIME complements SHAP:
          ─ SHAP tells you the exact mathematical contribution of each feature
          ─ LIME tells you "change this feature value → prediction flips"
        """
        if not HAS_LIME:
            logger.warning("LIME not installed. Run: pip install lime")
            return

        if indices is None:
            indices = list(range(min(3, len(X_test_scaled))))

        logger.info("Building LIME explainer…")
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_scaled,
            feature_names=self.feature_names,
            class_names=["Did not survive", "Survived"],
            mode="classification",
            discretize_continuous=True,
            random_state=42,
        )

        for idx in indices:
            pred  = self.model.predict(X_test_scaled[idx:idx+1])[0]
            prob  = self.model.predict_proba(X_test_scaled[idx:idx+1])[0][1]
            label = "Survived" if pred == 1 else "Did not survive"

            logger.info("Computing LIME explanation for passenger %d…", idx)
            explanation = lime_explainer.explain_instance(
                data_row=X_test_scaled[idx],
                predict_fn=self.model.predict_proba,
                num_features=num_features,
            )

            # ── Save HTML ─────────────────────────────────────────
            html_path = output_dir / f"lime_explanation_{idx}.html"
            explanation.save_to_file(str(html_path))
            logger.info("LIME HTML saved to %s", html_path)

            # ── Also save as matplotlib figure ────────────────────
            fig = explanation.as_pyplot_figure()
            fig.suptitle(
                f"LIME — Passenger #{idx} — {label} ({prob:.1%})",
                fontsize=12, fontweight="bold", y=1.02,
            )
            plt.tight_layout()
            png_path = output_dir / f"lime_explanation_{idx}.png"
            plt.savefig(png_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info("LIME PNG saved to %s", png_path)

            if HAS_MLFLOW:
                mlflow.log_artifact(str(html_path))
                mlflow.log_artifact(str(png_path))

    # ── Combined run ──────────────────────────────────────────────

    def run_all(
        self,
        X_train_scaled: np.ndarray,
        X_test_scaled:  np.ndarray,
        X_test_df:      pd.DataFrame,
        output_dir:     Path,
        local_indices:  list[int] = None,
    ) -> dict:
        """
        Run all explanations (global SHAP + local SHAP + LIME) in one call.
        Returns the ranked SHAP feature importance dict.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("═══ Running Model Explainability ════════════════════")
        ranked = {}

        if HAS_SHAP:
            ranked = self.explain_global_shap(X_train_scaled, X_test_scaled, output_dir)
            self.explain_local_shap(X_train_scaled, X_test_scaled, X_test_df, output_dir, local_indices)
        else:
            logger.warning("SHAP not available — skipping SHAP explanations")

        if HAS_LIME:
            self.explain_local_lime(X_train_scaled, X_test_scaled, X_test_df, output_dir, local_indices)
        else:
            logger.warning("LIME not available — skipping LIME explanations")

        logger.info("Explainability reports saved to %s", output_dir)
        return ranked


# ════════════════════════════════════════════════════════════════════
# Standalone CLI
# ════════════════════════════════════════════════════════════════════

def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_processing import DataProcessor

    parser = argparse.ArgumentParser(description="Generate SHAP + LIME model explanations")
    parser.add_argument("--model",        required=True, help="Path to model.pkl")
    parser.add_argument("--preprocessor", required=True, help="Path to preprocessor.pkl")
    parser.add_argument("--data",         required=True, help="Path to CSV (with Survived column)")
    parser.add_argument("--output",       default="reports", help="Output directory")
    parser.add_argument("--n-local",      type=int, default=5,
                        help="Number of passengers to explain locally (SHAP waterfall + LIME)")
    args = parser.parse_args()

    # ── Load model + preprocessor ─────────────────────────────────
    model     = joblib.load(args.model)
    processor = DataProcessor()
    processor.load_preprocessor(args.preprocessor)

    # ── Process data ──────────────────────────────────────────────
    logger.info("Loading and processing data from %s", args.data)
    df_raw = processor.load_data(args.data)
    df     = processor.handle_missing_values(df_raw)
    df     = processor.feature_engineering(df, is_training=False)
    df     = processor.encode_categorical_features(df, is_training=False)
    X      = df[processor.feature_names]
    X_raw  = df_raw   # keep original for annotation

    # ── Train/test split to get separate background for SHAP ──────
    from sklearn.model_selection import train_test_split
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    X_train_sc = processor.scale_features(X_train, is_training=False)
    X_test_sc  = processor.scale_features(X_test,  is_training=False)
    X_test_df  = X_raw.iloc[X_test.index]

    # ── Run explainability ────────────────────────────────────────
    explainer = ModelExplainer(model, processor, processor.feature_names)
    local_idx = list(range(min(args.n_local, len(X_test_sc))))

    explainer.run_all(
        X_train_scaled=X_train_sc,
        X_test_scaled=X_test_sc,
        X_test_df=X_test_df,
        output_dir=Path(args.output),
        local_indices=local_idx,
    )

    logger.info("✅ Explainability reports written to %s", args.output)


if __name__ == "__main__":
    main()
