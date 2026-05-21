"""
check_drift_threshold.py
─────────────────────────
Reads the consolidated analysis_results.json and exits with code 1
if severity warrants an alert — causing the CM pipeline's
Notify_and_Alert stage to trigger.

Called by: cm-monitoring-drift.yml → Check Drift Threshold step
           (continueOnError: true  so the pipeline continues to alerting)
"""

import argparse
import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# Severities that trigger the alert/retrain notification
ALERT_SEVERITIES = {"CRITICAL", "HIGH", "MEDIUM"}


def main():
    parser = argparse.ArgumentParser(description="Check drift threshold and set pipeline outcome")
    parser.add_argument("--drift-report",        default="drift_report.json")
    parser.add_argument("--concept-drift-report", default="concept_drift_report.json")
    parser.add_argument("--analysis",            default="analysis_results.json",
                        help="Consolidated analysis file (preferred if exists)")
    args = parser.parse_args()

    # Try consolidated analysis first (written by analyze_drift_results.py)
    try:
        with open(args.analysis) as f:
            analysis = json.load(f)
        severity       = analysis.get("severity", "NONE")
        any_drift      = analysis.get("any_drift_detected", False)
        recommendation = analysis.get("recommendation", "")
        logger.info("Loaded consolidated analysis: severity=%s", severity)
    except FileNotFoundError:
        # Fallback: read raw reports
        logger.warning("Consolidated analysis not found, reading raw reports")
        any_drift = False
        severity = "NONE"

        try:
            with open(args.drift_report) as f:
                data_report = json.load(f)
            if data_report.get("overall_drift_detected"):
                any_drift = True
                severity = "LOW"
        except FileNotFoundError:
            logger.warning("Data drift report not found")

        try:
            with open(args.concept_drift_report) as f:
                concept_report = json.load(f)
            if concept_report.get("concept_drift_detected"):
                any_drift = True
                severity = "HIGH" if severity == "LOW" else "MEDIUM"
        except FileNotFoundError:
            logger.warning("Concept drift report not found")

        recommendation = f"Severity: {severity}"

    # ── Report ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  Drift check result:  severity=%s", severity)
    logger.info("  Any drift detected:  %s", any_drift)
    logger.info("  Recommendation:      %s", recommendation)
    logger.info("=" * 60)

    if severity in ALERT_SEVERITIES:
        logger.warning(
            "Drift severity '%s' meets alert threshold. "
            "Exiting with code 1 to trigger Notify_and_Alert stage.",
            severity,
        )
        # Write a flag file for downstream steps to read
        with open("drift_detected.flag", "w") as f:
            json.dump({"severity": severity, "any_drift": any_drift}, f)
        sys.exit(1)

    logger.info("No actionable drift. Pipeline continues normally.")


if __name__ == "__main__":
    main()
