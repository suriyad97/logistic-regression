"""
generate_inference_report.py
─────────────────────────────
Generates a self-contained HTML report from a batch predictions CSV.
Published as an Azure DevOps Build Artifact so the team can review
predictions at a glance from the pipeline run page.

Called by: cd-inference-pipeline.yml → Generate Inference Report step
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Titanic Inference Report — {run_date}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0f172a; color: #e2e8f0; padding: 2rem; }}
    h1 {{ color: #38bdf8; margin-bottom: 0.25rem; }}
    h2 {{ color: #7dd3fc; margin: 1.5rem 0 0.75rem; }}
    .subtitle {{ color: #94a3b8; font-size: 0.9rem; margin-bottom: 2rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
              gap: 1rem; margin-bottom: 2rem; }}
    .card {{ background: #1e293b; border-radius: 12px; padding: 1.25rem;
              border: 1px solid #334155; }}
    .card-value {{ font-size: 2rem; font-weight: 700; color: #38bdf8; }}
    .card-label {{ font-size: 0.8rem; color: #94a3b8; margin-top: 0.25rem; }}
    .survived {{ color: #4ade80; }} .not-survived {{ color: #f87171; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
    th {{ background: #1e293b; color: #7dd3fc; text-align: left;
          padding: 0.6rem 0.8rem; border-bottom: 2px solid #334155; }}
    td {{ padding: 0.5rem 0.8rem; border-bottom: 1px solid #1e293b; }}
    tr:nth-child(even) {{ background: #0f1929; }}
    .badge {{ display: inline-block; border-radius: 4px; padding: 0.15rem 0.5rem;
               font-size: 0.75rem; font-weight: 600; }}
    .badge-survived {{ background: #14532d; color: #4ade80; }}
    .badge-not {{ background: #450a0a; color: #f87171; }}
    .bar-bg {{ background: #334155; border-radius: 4px; height: 6px; }}
    .bar-fill {{ background: #38bdf8; border-radius: 4px; height: 6px; }}
  </style>
</head>
<body>
  <h1>🚢 Titanic Inference Report</h1>
  <div class="subtitle">Generated: {run_date} UTC</div>

  <h2>Summary</h2>
  <div class="grid">
    <div class="card">
      <div class="card-value">{total}</div>
      <div class="card-label">Total Predictions</div>
    </div>
    <div class="card">
      <div class="card-value survived">{survived}</div>
      <div class="card-label">Predicted Survived</div>
    </div>
    <div class="card">
      <div class="card-value not-survived">{not_survived}</div>
      <div class="card-label">Predicted Not Survived</div>
    </div>
    <div class="card">
      <div class="card-value">{survival_rate}%</div>
      <div class="card-label">Survival Rate</div>
    </div>
    <div class="card">
      <div class="card-value">{mean_prob}</div>
      <div class="card-label">Mean Survival Probability</div>
    </div>
    <div class="card">
      <div class="card-value">{high_confidence}%</div>
      <div class="card-label">High Confidence (&gt;80%)</div>
    </div>
  </div>

  <h2>Probability Distribution (Deciles)</h2>
  <table>
    <tr><th>Decile</th><th>Min Prob</th><th>Max Prob</th><th>Count</th><th>Distribution</th></tr>
    {decile_rows}
  </table>

  <h2>Sample Predictions (top 50)</h2>
  <table>
    <tr>
      <th>#</th><th>Pclass</th><th>Sex</th><th>Age</th>
      <th>Fare</th><th>Survival Prob</th><th>Prediction</th>
    </tr>
    {sample_rows}
  </table>
</body>
</html>"""


def build_decile_rows(df: pd.DataFrame) -> str:
    probs = df["survival_probability"].values
    bins = np.linspace(0, 1, 11)
    rows = []
    for i in range(10):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi if i < 9 else probs <= hi)
        count = int(mask.sum())
        pct = count / len(probs) * 100
        bar_width = int(pct)
        rows.append(
            f"<tr><td>{lo:.1f}–{hi:.1f}</td>"
            f"<td>{lo:.2f}</td><td>{hi:.2f}</td><td>{count}</td>"
            f"<td><div class='bar-bg'><div class='bar-fill' style='width:{bar_width}%'></div></div>"
            f" {pct:.1f}%</td></tr>"
        )
    return "\n    ".join(rows)


def build_sample_rows(df: pd.DataFrame, n: int = 50) -> str:
    sample = df.head(n)
    rows = []
    for i, row in sample.iterrows():
        badge_cls = "badge-survived" if row.get("prediction", 0) == 1 else "badge-not"
        badge_txt = "Survived" if row.get("prediction", 0) == 1 else "Not Survived"
        prob = float(row.get("survival_probability", 0))
        rows.append(
            f"<tr><td>{i+1}</td>"
            f"<td>{int(row.get('Pclass', 0))}</td>"
            f"<td>{row.get('Sex', '—')}</td>"
            f"<td>{row.get('Age', '—')}</td>"
            f"<td>{row.get('Fare', '—')}</td>"
            f"<td>{prob:.1%}</td>"
            f"<td><span class='badge {badge_cls}'>{badge_txt}</span></td></tr>"
        )
    return "\n    ".join(rows)


def generate_report(predictions_path: str, output: str) -> None:
    pred_file = Path(predictions_path) / "predictions.csv"
    if not pred_file.exists():
        pred_file = Path(predictions_path)  # maybe direct file path

    df = pd.read_csv(pred_file)
    logger.info("Loaded %d predictions from %s", len(df), pred_file)

    survived = int((df["prediction"] == 1).sum())
    not_survived = len(df) - survived
    survival_rate = round(survived / len(df) * 100, 1)
    mean_prob = round(df["survival_probability"].mean(), 3)
    high_conf = round(
        ((df["survival_probability"] > 0.8) | (df["survival_probability"] < 0.2)).mean() * 100, 1
    )

    html = REPORT_TEMPLATE.format(
        run_date=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        total=len(df),
        survived=survived,
        not_survived=not_survived,
        survival_rate=survival_rate,
        mean_prob=mean_prob,
        high_confidence=high_conf,
        decile_rows=build_decile_rows(df),
        sample_rows=build_sample_rows(df),
    )

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Inference report written to %s", output)


def main():
    parser = argparse.ArgumentParser(description="Generate HTML inference report")
    parser.add_argument("--predictions-path", required=True, help="Folder with predictions.csv")
    parser.add_argument("--output",           default="report.html")
    args = parser.parse_args()
    generate_report(args.predictions_path, args.output)


if __name__ == "__main__":
    main()
