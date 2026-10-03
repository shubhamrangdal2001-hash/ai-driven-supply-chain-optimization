"""
Forecast Risk Analysis
======================
Calculates Overstock Risk % and Stockout Risk % from real LSTM and SARIMA
prediction files in /reports and saves detailed results to:
    reports/forecast_risk_report.json

Formula
-------
  overstock_pct  = max(0, (forecast - actual) / actual) * 100
  stockout_pct   = max(0, (actual  - forecast) / actual) * 100

  overstock  → forecast OVER-estimated demand  → you ordered too much
  stockout   → forecast UNDER-estimated demand → you ran out of stock
"""

import os
import json
import pandas as pd

ROOT    = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, "reports")

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
lstm_path   = os.path.join(REPORTS, "lstm_predictions.csv")
sarima_path = os.path.join(REPORTS, "sarima_predictions.csv")

print("=" * 60)
print("  FORECAST RISK ANALYSIS")
print("=" * 60)

lstm_df = pd.read_csv(lstm_path, parse_dates=["Date"])
lstm_df.columns = lstm_df.columns.str.strip()
lstm_df = lstm_df.rename(columns={"lstm_forecast": "forecast", "Actual": "actual"})
lstm_df["model"] = "LSTM"

sarima_df = pd.read_csv(sarima_path, parse_dates=["Date"])
sarima_df.columns = sarima_df.columns.str.strip()
sarima_df = sarima_df.rename(columns={"sarima_forecast": "forecast", "Actual": "actual"})
sarima_df["model"] = "SARIMA"

print(f"\n[OK] Loaded {len(lstm_df):>3} LSTM   records  ({lstm_path})")
print(f"[OK] Loaded {len(sarima_df):>3} SARIMA  records  ({sarima_path})")


# ─────────────────────────────────────────────────────────────────────────────
# 2. CORE RISK CALCULATION
#    overstock_pct  = max(0, forecast - actual) / actual * 100
#    stockout_pct   = max(0, actual - forecast) / actual * 100
# ─────────────────────────────────────────────────────────────────────────────
def compute_risk(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["error"]         = d["forecast"] - d["actual"]
    d["error_pct"]     = (d["error"]                  / d["actual"]) * 100
    d["overstock_pct"] = (d["error"].clip(lower=0)    / d["actual"]) * 100
    d["stockout_pct"]  = ((-d["error"]).clip(lower=0) / d["actual"]) * 100
    return d

lstm_risk   = compute_risk(lstm_df)
sarima_risk = compute_risk(sarima_df)


# ─────────────────────────────────────────────────────────────────────────────
# 3. SUMMARY STATISTICS PER MODEL
# ─────────────────────────────────────────────────────────────────────────────
def summarise(df: pd.DataFrame, model_name: str) -> dict:
    n = len(df)
    overstock_periods = int((df["overstock_pct"] > 0).sum())
    stockout_periods  = int((df["stockout_pct"]  > 0).sum())

    def risk_level(avg):
        if avg < 5:   return "LOW RISK"
        if avg < 15:  return "MEDIUM RISK"
        return "HIGH RISK"

    return {
        "model": model_name,
        "periods_evaluated": n,
        "date_range": {
            "start": str(df["Date"].min().date()),
            "end":   str(df["Date"].max().date()),
        },
        "overstock": {
            "avg_overstock_pct":       round(df["overstock_pct"].mean(), 2),
            "max_overstock_pct":       round(df["overstock_pct"].max(),  2),
            "periods_overstocked":     overstock_periods,
            "overstock_frequency_pct": round(overstock_periods / n * 100, 1),
            "risk_level":              risk_level(df["overstock_pct"].mean()),
            "meaning": (
                f"On average the forecast was {df['overstock_pct'].mean():.2f}% HIGHER than "
                f"actual demand in {overstock_periods} of {n} periods -> excess inventory risk."
            ),
        },
        "stockout": {
            "avg_stockout_pct":        round(df["stockout_pct"].mean(), 2),
            "max_stockout_pct":        round(df["stockout_pct"].max(),  2),
            "periods_stockout":        stockout_periods,
            "stockout_frequency_pct":  round(stockout_periods / n * 100, 1),
            "risk_level":              risk_level(df["stockout_pct"].mean()),
            "meaning": (
                f"On average the forecast was {df['stockout_pct'].mean():.2f}% LOWER than "
                f"actual demand in {stockout_periods} of {n} periods -> stockout / lost-sales risk."
            ),
        },
        "forecast_accuracy": {
            "MAPE_pct":   round(df["error_pct"].abs().mean(), 2),
            "mean_error": round(df["error"].mean(), 2),
            "bias": (
                "over-forecast (excess inventory bias)"
                if df["error"].mean() > 0
                else "under-forecast (stockout bias)"
            ),
        },
        "period_detail": [
            {
                "date":          str(row["Date"].date()),
                "actual":        round(float(row["actual"]),       2),
                "forecast":      round(float(row["forecast"]),      2),
                "overstock_pct": round(float(row["overstock_pct"]), 2),
                "stockout_pct":  round(float(row["stockout_pct"]),  2),
                "error_pct":     round(float(row["error_pct"]),     2),
            }
            for _, row in df.iterrows()
        ],
    }

lstm_summary   = summarise(lstm_risk,   "LSTM")
sarima_summary = summarise(sarima_risk, "SARIMA")


# ─────────────────────────────────────────────────────────────────────────────
# 4. ENSEMBLE (average of both models)
# ─────────────────────────────────────────────────────────────────────────────
merged = pd.merge(
    lstm_risk[["Date", "actual", "overstock_pct", "stockout_pct"]].rename(
        columns={"overstock_pct": "lstm_over", "stockout_pct": "lstm_out"}),
    sarima_risk[["Date", "overstock_pct", "stockout_pct"]].rename(
        columns={"overstock_pct": "sarima_over", "stockout_pct": "sarima_out"}),
    on="Date", how="inner",
)
merged["ensemble_overstock_pct"] = (merged["lstm_over"] + merged["sarima_over"]) / 2
merged["ensemble_stockout_pct"]  = (merged["lstm_out"]  + merged["sarima_out"])  / 2

ensemble_summary = {
    "model":               "ENSEMBLE (LSTM + SARIMA average)",
    "avg_overstock_pct":   round(float(merged["ensemble_overstock_pct"].mean()), 2),
    "avg_stockout_pct":    round(float(merged["ensemble_stockout_pct"].mean()),  2),
    "periods_evaluated":   int(len(merged)),
    "note": (
        "Ensemble averages both model forecasts to smooth individual model bias. "
        "Use this as the safest risk estimate."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# 5. SAVE JSON
# ─────────────────────────────────────────────────────────────────────────────
report = {
    "report_title":  "Forecast Risk Analysis – Overstock & Stockout",
    "generated_at":  pd.Timestamp.now().isoformat(timespec="seconds"),
    "formula": {
        "overstock_pct": "max(0, (forecast - actual) / actual) * 100",
        "stockout_pct":  "max(0, (actual - forecast) / actual) * 100",
        "explanation": (
            "overstock = forecast too HIGH vs actual → surplus units on shelf. "
            "stockout  = forecast too LOW  vs actual → not enough units ordered."
        ),
    },
    "models":   [lstm_summary, sarima_summary],
    "ensemble": ensemble_summary,
}

out_path = os.path.join(REPORTS, "forecast_risk_report.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print(f"\n[SAVED] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. CONSOLE PRINT
# ─────────────────────────────────────────────────────────────────────────────
for s in [lstm_summary, sarima_summary]:
    print(f"\n{'-'*58}")
    print(f"  Model : {s['model']}")
    print(f"{'-'*58}")
    print(f"  Periods evaluated          : {s['periods_evaluated']}")
    print(f"  Date range                 : {s['date_range']['start']}  to  {s['date_range']['end']}")
    print()
    print(f"  [!] Overstock Risk")
    print(f"      Average overstock       : {s['overstock']['avg_overstock_pct']:>8.2f} %")
    print(f"      Max overstock (1 period): {s['overstock']['max_overstock_pct']:>8.2f} %")
    print(f"      Periods overstocked     : {s['overstock']['periods_overstocked']:>4}  "
          f"({s['overstock']['overstock_frequency_pct']} % of all periods)")
    print(f"      Risk level              : {s['overstock']['risk_level']}")
    print(f"      >> {s['overstock']['meaning']}")
    print()
    print(f"  [!!] Stockout Risk")
    print(f"      Average stockout        : {s['stockout']['avg_stockout_pct']:>8.2f} %")
    print(f"      Max stockout (1 period) : {s['stockout']['max_stockout_pct']:>8.2f} %")
    print(f"      Periods with stockout   : {s['stockout']['periods_stockout']:>4}  "
          f"({s['stockout']['stockout_frequency_pct']} % of all periods)")
    print(f"      Risk level              : {s['stockout']['risk_level']}")
    print(f"      >> {s['stockout']['meaning']}")
    print()
    print(f"  [~] Forecast Accuracy")
    print(f"      MAPE                    : {s['forecast_accuracy']['MAPE_pct']:>8.2f} %")
    print(f"      Mean error              : {s['forecast_accuracy']['mean_error']:>12,.0f} units")
    print(f"      Bias direction          : {s['forecast_accuracy']['bias']}")

print(f"\n{'-'*58}")
print(f"  ENSEMBLE (LSTM + SARIMA average)")
print(f"{'-'*58}")
print(f"  Avg overstock risk : {ensemble_summary['avg_overstock_pct']:>8.2f} %")
print(f"  Avg stockout  risk : {ensemble_summary['avg_stockout_pct']:>8.2f} %")
print(f"\n{'='*58}")
print(f"  Full JSON -> {out_path}")
print(f"{'='*58}\n")
