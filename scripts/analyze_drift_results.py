"""
analyze_drift_results.py
─────────────────────────
Aggregates the two separate drift reports (data drift + concept drift)
into a single consolidated analysis JSON that is:
  1. Published as a build artifact for human review
  2. Used by check_drift_threshold.py to decide whether to alert

Called by: cm-monitoring-drift.yml → Analyze Drift Results step
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def summarize_data_drift(report: dict) -> dict:
    features = report.get("features", {})
    drifted = [f for f, v in features.items() if v.get("ks_test", {}).get("drift_detected")]
    return {
        "overall_drift_detected": report.get("overall_drift_detected", False),
        "drifted_features":       drifted,
        "drifted_feature_count":  len(drifted),
        "total_features_checked": len(features),
        "drift_percentage":       round(len(drifted) / max(len(features), 1) * 100, 1),
    }


def summarize_concept_drift(report: dict) -> dict:
    return {
        "concept_drift_detected":  report.get("concept_drift_detected", False),
        "reason":                  report.get("reason", ""),
        "performance_drop":        report.get("performance_drop"),
        "current_accuracy":        report.get("current_accuracy"),
        "baseline_accuracy":       report.get("baseline_accuracy"),
        "samples_evaluated":       report.get("samples_evaluated", 0),
        "metrics":                 report.get("metrics", {}),
    }


def determine_severity(data_summary: dict, concept_summary: dict) -> str:
    """Map drift signals to a severity level for alerting."""
    concept_drifted = concept_summary["concept_drift_detected"]
    data_drifted = data_summary["overall_drift_detected"]
    data_pct = data_summary["drift_percentage"]

    if concept_drifted and data_drifted:
        return "CRITICAL"
    if concept_drifted or data_pct > 50:
        return "HIGH"
    if data_pct > 20:
        return "MEDIUM"
    if data_drifted:
        return "LOW"
    return "NONE"


def main():
    parser = argparse.ArgumentParser(description="Consolidate drift reports")
    parser.add_argument("--drift-report",         required=False, help="Data drift report JSON")
    parser.add_argument("--concept-drift-report", required=False, help="Concept drift report JSON")
    parser.add_argument("--output",               default="analysis_results.json")
    args = parser.parse_args()

    logger.info("Loading drift reports…")
    
    # Load Data Drift Report if provided
    data_summary = summarize_data_drift({})
    if args.drift_report:
        try:
            data_report = load_json(args.drift_report)
            data_summary = summarize_data_drift(data_report)
        except Exception as e:
            logger.warning(f"Could not load data drift report {args.drift_report}: {e}")

    # Load Concept Drift Report if provided
    concept_summary = summarize_concept_drift({})
    if args.concept_drift_report:
        try:
            concept_report = load_json(args.concept_drift_report)
            concept_summary = summarize_concept_drift(concept_report)
        except Exception as e:
            logger.warning(f"Could not load concept drift report {args.concept_drift_report}: {e}")
    severity        = determine_severity(data_summary, concept_summary)

    any_drift = (
        data_summary["overall_drift_detected"]
        or concept_summary["concept_drift_detected"]
    )

    analysis = {
        "analysis_timestamp":    datetime.now(timezone.utc).isoformat(),
        "severity":              severity,
        "any_drift_detected":    any_drift,
        "action_required":       severity in ("CRITICAL", "HIGH"),
        "recommendation":        _recommendation(severity),
        "data_drift_summary":    data_summary,
        "concept_drift_summary": concept_summary,
    }

    with open(args.output, "w") as f:
        json.dump(analysis, f, indent=2)
    logger.info("Analysis written to %s  —  Severity: %s", args.output, severity)

    # Echo for pipeline log visibility
    print(f"\n{'='*50}")
    print(f"  DRIFT ANALYSIS RESULT: {severity}")
    print(f"  Any drift detected:    {any_drift}")
    print(f"  Recommendation:        {analysis['recommendation']}")
    print(f"{'='*50}\n")


def _recommendation(severity: str) -> str:
    return {
        "CRITICAL": "Immediate retraining required. Both data and concept drift detected.",
        "HIGH":     "Retraining strongly recommended. Significant drift detected.",
        "MEDIUM":   "Monitor closely. Consider retraining in next sprint.",
        "LOW":      "Minor data drift. No immediate action required.",
        "NONE":     "No drift detected. Model is healthy.",
    }[severity]


if __name__ == "__main__":
    main()
