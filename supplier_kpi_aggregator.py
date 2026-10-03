"""
Supplier KPI Aggregator
========================
Reads dataco_clean.parquet and computes per-category / per-shipping-mode KPIs:
  - On-time delivery rate
  - Average delay days
  - Profit margin by category
  - Order volume & revenue trend
  - Supplier performance matrix

Outputs:
  reports/supplier_kpis.csv
  reports/category_kpis.csv
"""

import os, warnings
import pandas as pd

warnings.filterwarnings("ignore")

ROOT    = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, "reports")
os.makedirs(REPORTS, exist_ok=True)

print("=" * 60)
print("SUPPLIER KPI AGGREGATOR")
print("=" * 60)

parquet_path = os.path.join(REPORTS, "dataco_clean.parquet")
dataco_raw = os.path.join(ROOT, "DataCo Smart Supply Chain", "DataCoSupplyChainDataset.csv")

if os.path.exists(parquet_path):
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded DataCo parquet: {df.shape}")
elif os.path.exists(dataco_raw):
    df = pd.read_csv(dataco_raw, encoding="latin-1")
    df["order date (DateOrders)"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
    null_pct = df.isnull().mean()
    df.drop(columns=null_pct[null_pct > 0.4].index, inplace=True)
    num_cols = df.select_dtypes("number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df["delivery_delay"] = df["Days for shipment (scheduled)"] - df["Days for shipping (real)"]
    df["is_late"] = (df["delivery_delay"] < 0).astype(int)
    df["profit_margin"] = df["Benefit per order"] / (df["Sales"] + 1e-9)
    print(f"  Loaded DataCo raw CSV: {df.shape}")
else:
    raise FileNotFoundError(
        "No real DataCo data found. Place DataCoSupplyChainDataset.csv in "
        "'DataCo Smart Supply Chain/' or run phase1_data_loading.py.\n"
        "Synthetic KPI generation is disabled."
    )

# ── 1. Overall KPIs ───────────────────────────────────────────────────
on_time_rate  = 1 - df["is_late"].mean()
avg_delay     = df["delivery_delay"].mean()
avg_margin    = df["profit_margin"].mean()
total_revenue = df["Sales"].sum()

print(f"\n  Overall On-Time Rate   : {on_time_rate*100:.1f}%")
print(f"  Avg Delivery Delay     : {avg_delay:.2f} days")
print(f"  Avg Profit Margin      : {avg_margin*100:.1f}%")
print(f"  Total Revenue          : ${total_revenue:,.0f}")

# ── 2. Shipping Mode KPIs ─────────────────────────────────────────────
ship_kpis = (
    df.groupby("Shipping Mode")
    .agg(
        orders         =("Sales", "count"),
        revenue        =("Sales", "sum"),
        on_time_rate   =("is_late", lambda x: 1 - x.mean()),
        avg_delay_days =("delivery_delay", "mean"),
        avg_margin_pct =("profit_margin", lambda x: x.mean() * 100),
    )
    .round(3)
    .reset_index()
)
print(f"\n  Shipping Mode KPIs:\n{ship_kpis.to_string(index=False)}")

# ── 3. Category KPIs ──────────────────────────────────────────────────
cat_kpis = (
    df.groupby("Category Name")
    .agg(
        orders          =("Sales", "count"),
        revenue         =("Sales", "sum"),
        avg_order_value =("Sales", "mean"),
        on_time_rate    =("is_late", lambda x: 1 - x.mean()),
        profit_margin   =("profit_margin", lambda x: x.mean() * 100),
        total_units     =("Order Item Quantity", "sum"),
    )
    .round(2)
    .reset_index()
    .sort_values("revenue", ascending=False)
)
print(f"\n  Category KPIs:\n{cat_kpis.to_string(index=False)}")

# ── 4. Monthly Trend ──────────────────────────────────────────────────
monthly_trend = (
    df.groupby(["order_year", "order_month"])
    .agg(
        revenue   =("Sales", "sum"),
        orders    =("Sales", "count"),
        on_time_pct=("is_late", lambda x: (1 - x.mean()) * 100),
    )
    .round(2)
    .reset_index()
)

# ── 5. Supplier Performance Matrix (shipping-mode × category) ─────────
if "Category Name" in df.columns and "Shipping Mode" in df.columns:
    supplier_matrix = (
        df.groupby(["Shipping Mode", "Category Name"])["is_late"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "late_rate", "count": "orders"})
        .reset_index()
    )
    supplier_matrix["on_time_pct"] = ((1 - supplier_matrix["late_rate"]) * 100).round(1)
else:
    supplier_matrix = pd.DataFrame()

# ── Save outputs ──────────────────────────────────────────────────────
ship_kpis.to_csv(os.path.join(REPORTS, "supplier_kpis.csv"), index=False)
cat_kpis.to_csv(os.path.join(REPORTS, "category_kpis.csv"),  index=False)
monthly_trend.to_csv(os.path.join(REPORTS, "monthly_trend.csv"), index=False)
if not supplier_matrix.empty:
    supplier_matrix.to_csv(os.path.join(REPORTS, "supplier_matrix.csv"), index=False)

# Save overall summary for dashboard
summary = {
    "on_time_rate_pct"  : round(on_time_rate * 100, 1),
    "avg_delay_days"    : round(avg_delay, 2),
    "avg_margin_pct"    : round(avg_margin * 100, 1),
    "total_revenue"     : round(total_revenue, 0),
    "total_orders"      : int(len(df)),
}
import json
with open(os.path.join(REPORTS, "kpi_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  Saved → supplier_kpis.csv, category_kpis.csv, monthly_trend.csv, kpi_summary.json")
print("\nSupplier KPI aggregation complete.")
