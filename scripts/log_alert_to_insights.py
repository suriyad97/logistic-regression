"""
log_alert_to_insights.py
─────────────────────────
Logs a custom alert event to Azure Application Insights when
drift is detected in production.

Why App Insights?
  - Creates a permanent, searchable audit trail of every drift event
  - Enables building dashboards and metric alerts in Azure Monitor
  - The ML team can query "how often did drift occur last quarter?"

Called by: cm-monitoring-drift.yml → Log Alert to Application Insights step
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def log_to_application_insights(
    instrumentation_key: str,
    severity: str,
    message: str,
    properties: dict = None,
) -> None:
    """
    Send a custom event to Application Insights using the
    opencensus-ext-azure package (lightweight, no SDK dependency).
    Falls back to a local log if the package is not installed.
    """
    try:
        from opencensus.ext.azure.log_exporter import AzureLogHandler

        ai_logger = logging.getLogger("drift_alert")
        ai_logger.addHandler(
            AzureLogHandler(connection_string=f"InstrumentationKey={instrumentation_key}")
        )
        ai_logger.setLevel(logging.WARNING)

        extra = {
            "custom_dimensions": {
                "severity":  severity,
                "message":   message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **(properties or {}),
            }
        }

        if severity.upper() in ("CRITICAL", "HIGH"):
            ai_logger.error("DriftAlert: %s — %s", severity, message, extra=extra)
        else:
            ai_logger.warning("DriftAlert: %s — %s", severity, message, extra=extra)

        logger.info("Alert sent to Application Insights (severity=%s)", severity)

    except ImportError:
        logger.warning(
            "opencensus-ext-azure not installed. Logging alert locally only.\n"
            "Install with: pip install opencensus-ext-azure"
        )
        # Still write to stdout for Azure DevOps log capture
        alert_payload = {
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "severity":   severity,
            "message":    message,
            "properties": properties or {},
        }
        print(f"\n[APP INSIGHTS ALERT] {json.dumps(alert_payload, indent=2)}\n")


def load_drift_context() -> dict:
    """Try to load drift analysis for extra alert context."""
    for fname in ("analysis_results.json", "drift_report.json"):
        try:
            with open(fname) as f:
                return json.load(f)
        except FileNotFoundError:
            continue
    return {}


def main():
    parser = argparse.ArgumentParser(description="Log drift alert to Application Insights")
    parser.add_argument("--severity",            default="HIGH")
    parser.add_argument("--message",             default="Data drift detected in production")
    parser.add_argument("--instrumentation-key", default=None,
                        help="App Insights instrumentation key (overrides env var)")
    parser.add_argument("--build-uri",           default=os.environ.get("BUILD_BUILDURI", ""))
    args = parser.parse_args()

    ikey = (
        args.instrumentation_key
        or os.environ.get("APPINSIGHTS_KEY")
        or os.environ.get("APPINSIGHTS_INSTRUMENTATIONKEY")
    )
    if not ikey:
        logger.error(
            "Application Insights instrumentation key not provided. "
            "Set APPINSIGHTS_KEY env var or pass --instrumentation-key."
        )
        # Don't fail the pipeline — alerting failure shouldn't block monitoring
        sys.exit(0)

    context = load_drift_context()
    properties = {
        "build_uri":    args.build_uri,
        "pipeline":     "cm-monitoring-drift",
        "model":        "titanic-logistic-regression",
        "environment":  "prod",
        "severity_level": args.severity,
    }
    if context:
        properties["drift_details"] = json.dumps(
            context.get("data_drift_summary", context), ensure_ascii=False
        )[:1024]  # App Insights property size limit

    log_to_application_insights(ikey, args.severity, args.message, properties)
    logger.info("Alert logging complete.")


if __name__ == "__main__":
    main()
