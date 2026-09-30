from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
summary = json.loads((ROOT / "reports/backtest_2026_summary.json").read_text(encoding="utf-8"))
rows = pd.read_csv(ROOT / "results/backtest_2026_rows.csv")

months = list(summary["by_month"])
mae = [summary["by_month"][month]["mae"] for month in months]
rho = [summary["by_month"][month]["spearman"] for month in months]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
ax = axes[0]
ax.plot(months, mae, marker="o", linewidth=2, color="#1f77b4", label="MAE")
ax.axhline(summary["overall"]["mae"], color="#1f77b4", linestyle="--", alpha=.55)
ax.set_title("2026 error by month")
ax.set_ylabel("Fantasy points (lower is better)")
ax.tick_params(axis="x", rotation=45)
ax.grid(alpha=.2)
ax2 = ax.twinx()
ax2.plot(months, rho, marker="s", linewidth=2, color="#d97706", label="Spearman")
ax2.set_ylabel("Rank correlation (higher is better)")

ax = axes[1]
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
actual_q = rows["actual_points"].quantile(quantiles)
predicted_q = rows["predicted_points"].quantile(quantiles)
labels = ["P10", "P25", "P50", "P75", "P90", "P95", "P99"]
ax.plot(labels, actual_q, marker="o", linewidth=2, label="Actual", color="#111827")
ax.plot(labels, predicted_q, marker="o", linewidth=2, label="Predicted", color="#059669")
ax.set_title("Prediction compression at the performance tail")
ax.set_ylabel("Fantasy points")
ax.grid(alpha=.2)
ax.legend(frameon=False)

fig.suptitle("Frozen 2025 model on 2026 T20 data", fontsize=14, fontweight="bold")
fig.tight_layout()
output = ROOT / "reports/backtest_2026_diagnostics.png"
fig.savefig(output, dpi=180, bbox_inches="tight")
print(output)
