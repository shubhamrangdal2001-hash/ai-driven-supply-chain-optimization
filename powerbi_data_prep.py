"""
Power BI Data Preparation — REAL DATA ONLY
============================================
Builds Power BI-ready CSVs from Kaggle source datasets and pipeline outputs.
No synthetic or mock data is generated. Missing source files raise FileNotFoundError.

Sources (all real):
  - DataCo Smart Supply Chain/DataCoSupplyChainDataset.csv
  - Product Demand Forecasting/Historical Product Demand.csv
  - Olist Brazilian E-Commerce/*.csv
  - reports/inventory_plan.csv, route_summary.csv, simulation_*.csv/json

Outputs: reports/powerbi/
"""

import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(ROOT, "reports")
PBI = os.path.join(REPORT, "powerbi")

DATACO_RAW = os.path.join(
    ROOT, "DataCo Smart Supply Chain", "DataCoSupplyChainDataset.csv"
)
DEMAND_RAW = os.path.join(
    ROOT, "Product Demand Forecasting", "Historical Product Demand.csv"
)
OLIST_DIR = os.path.join(ROOT, "Olist Brazilian E-Commerce")

DATACO_PARQUET = os.path.join(REPORT, "dataco_clean.parquet")
INV_PATH = os.path.join(REPORT, "inventory_plan.csv")
ROUTE_PATH = os.path.join(REPORT, "route_summary.csv")
SIM_PATH = os.path.join(REPORT, "simulation_results.csv")
SIM_SUM_PATH = os.path.join(REPORT, "simulation_summary.json")

# Expected sizes from real Kaggle datasets (tolerance allows minor cleaning drops)
EXPECTED_DATACO_ROWS = (175_000, 185_000)
EXPECTED_DEMAND_ROWS = (1_000_000, 1_100_000)
MIN_INVENTORY_SKUS = 2_000
SYNTHETIC_SKU_SIGNATURE = 200  # legacy fallback generated exactly 200 fake SKUs

DF_COLS = [
    "order date (DateOrders)",
    "order_year",
    "order_month",
    "order_quarter",
    "order_dayofweek",
    "Category Name",
    "Department Name",
    "Product Name",
    "Customer Segment",
    "Market",
    "Order Region",
    "Order Country",
    "Shipping Mode",
    "Delivery Status",
    "Days for shipment (scheduled)",
    "Days for shipping (real)",
    "delivery_delay",
    "is_late",
    "Late_delivery_risk",
    "Order Item Quantity",
    "Sales",
    "Benefit per order",
    "Order Item Discount Rate",
    "Order Item Profit Ratio",
    "profit_margin",
    "Order Status",
]


def _require(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required real data source missing: {label}\n  Expected: {path}"
        )


def _row_count_in_range(count: int, bounds: tuple[int, int], label: str) -> None:
    lo, hi = bounds
    if not lo <= count <= hi:
        raise ValueError(
            f"{label} row count {count:,} outside expected real-data range "
            f"[{lo:,}, {hi:,}]. Refusing to export — possible synthetic or corrupt input."
        )


def validate_real_sources() -> None:
    """Fail fast unless all Kaggle source files are present on disk."""
    _require(DATACO_RAW, "DataCo Supply Chain (Kaggle)")
    _require(DEMAND_RAW, "Historical Product Demand (Kaggle)")
    _require(os.path.join(OLIST_DIR, "olist_orders_dataset.csv"), "Olist orders (Kaggle)")
    _require(os.path.join(OLIST_DIR, "olist_customers_dataset.csv"), "Olist customers (Kaggle)")
    _require(INV_PATH, "inventory_plan.csv (run phase4_inventory_optimization.py)")
    _require(ROUTE_PATH, "route_summary.csv (run phase3_route_optimization.py)")
    _require(SIM_SUM_PATH, "simulation_summary.json (run safety_stock_simulation.py)")

    raw_dataco_rows = sum(1 for _ in open(DATACO_RAW, encoding="latin-1", errors="replace")) - 1
    _row_count_in_range(raw_dataco_rows, EXPECTED_DATACO_ROWS, "DataCo raw CSV")

    demand_rows = sum(1 for _ in open(DEMAND_RAW, encoding="utf-8", errors="replace")) - 1
    _row_count_in_range(demand_rows, EXPECTED_DEMAND_ROWS, "Historical Product Demand raw CSV")

    inv = pd.read_csv(INV_PATH, usecols=["Product_Code"])
    if len(inv) < MIN_INVENTORY_SKUS:
        raise ValueError(
            f"inventory_plan.csv has {len(inv)} SKUs (need >={MIN_INVENTORY_SKUS}). "
            "Likely synthetic fallback — re-run phase4 on real demand data."
        )
    if len(inv) == SYNTHETIC_SKU_SIGNATURE:
        raise ValueError(
            "inventory_plan.csv has exactly 200 SKUs — matches legacy synthetic signature. "
            "Re-run phase4_inventory_optimization.py on real data."
        )

    with open(SIM_SUM_PATH, encoding="utf-8") as f:
        sim_sum = json.load(f)
    if sim_sum.get("data_provenance") != "real_inventory_plan":
        raise ValueError(
            "simulation_summary.json missing data_provenance='real_inventory_plan'. "
            "Re-run safety_stock_simulation.py after phase4 (no synthetic SKUs)."
        )
    if sim_sum.get("source_sku_count", 0) < MIN_INVENTORY_SKUS:
        raise ValueError(
            f"Simulation source_sku_count={sim_sum.get('source_sku_count')} — not from real inventory."
        )

    print("  [OK] All real data sources validated")


def load_dataco() -> pd.DataFrame:
    """Load cleaned DataCo from raw Kaggle CSV (authoritative source for Power BI)."""
    _require(DATACO_RAW, "DataCo Supply Chain dataset")
    print("  Loading DataCo from raw Kaggle CSV (latin-1) …")
    df = pd.read_csv(DATACO_RAW, encoding="latin-1")

    df["order date (DateOrders)"] = pd.to_datetime(
        df["order date (DateOrders)"], errors="coerce"
    )
    null_pct = df.isnull().mean()
    df.drop(columns=null_pct[null_pct > 0.4].index, inplace=True)
    num_cols = df.select_dtypes("number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    df["order_year"] = df["order date (DateOrders)"].dt.year
    df["order_month"] = df["order date (DateOrders)"].dt.month
    df["order_quarter"] = df["order date (DateOrders)"].dt.quarter
    df["order_dayofweek"] = df["order date (DateOrders)"].dt.dayofweek
    df["delivery_delay"] = (
        df["Days for shipment (scheduled)"] - df["Days for shipping (real)"]
    )
    df["is_late"] = (df["delivery_delay"] < 0).astype(int)
    df["profit_margin"] = df["Benefit per order"] / (df["Sales"] + 1e-9)

    df = df[DF_COLS]
    _row_count_in_range(len(df), EXPECTED_DATACO_ROWS, "DataCo cleaned")
    print(f"  Loaded DataCo CSV: {df.shape[0]:,} rows (verified real)")
    return df


def build_fact_orders(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["order date (DateOrders)"] = pd.to_datetime(
        df["order date (DateOrders)"], errors="coerce"
    )
    df["Order_Date"] = df["order date (DateOrders)"].dt.strftime("%Y-%m-%d")

    fact = df[
        [
            "Order_Date",
            "order_year",
            "order_month",
            "order_quarter",
            "order_dayofweek",
            "Category Name",
            "Department Name",
            "Product Name",
            "Customer Segment",
            "Market",
            "Order Region",
            "Order Country",
            "Shipping Mode",
            "Delivery Status",
            "Days for shipment (scheduled)",
            "Days for shipping (real)",
            "delivery_delay",
            "is_late",
            "Late_delivery_risk",
            "Order Item Quantity",
            "Sales",
            "Benefit per order",
            "Order Item Discount Rate",
            "Order Item Profit Ratio",
            "profit_margin",
            "Order Status",
        ]
    ].copy()

    fact.rename(
        columns={
            "order_year": "Order_Year",
            "order_month": "Order_Month",
            "order_quarter": "Order_Quarter",
            "order_dayofweek": "Day_of_Week",
            "Category Name": "Category",
            "Department Name": "Department",
            "Product Name": "Product",
            "Customer Segment": "Customer_Segment",
            "Market": "Market",
            "Order Region": "Region",
            "Order Country": "Country",
            "Shipping Mode": "Shipping_Mode",
            "Delivery Status": "Delivery_Status",
            "Days for shipment (scheduled)": "Scheduled_Days",
            "Days for shipping (real)": "Actual_Days",
            "delivery_delay": "Delay_Days",
            "is_late": "Is_Late",
            "Late_delivery_risk": "Late_Risk",
            "Order Item Quantity": "Quantity",
            "Sales": "Revenue",
            "Benefit per order": "Profit",
            "Order Item Discount Rate": "Discount_Rate",
            "Order Item Profit Ratio": "Profit_Ratio",
            "profit_margin": "Profit_Margin",
            "Order Status": "Order_Status",
        },
        inplace=True,
    )
    fact["On_Time"] = 1 - fact["Is_Late"].fillna(0)
    return fact


def build_dim_date(fact: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(fact["Order_Date"], errors="coerce").dropna()
    start, end = dates.min(), dates.max()
    dim = pd.DataFrame({"Date": pd.date_range(start, end, freq="D")})
    dim["DateKey"] = dim["Date"].dt.strftime("%Y%m%d").astype(int)
    dim["Year"] = dim["Date"].dt.year
    dim["Month"] = dim["Date"].dt.month
    dim["Month_Name"] = dim["Date"].dt.strftime("%B")
    dim["Quarter"] = dim["Date"].dt.quarter
    dim["YearMonth"] = dim["Date"].dt.strftime("%Y-%m")
    dim["Day_of_Week"] = dim["Date"].dt.dayofweek
    dim["Day_Name"] = dim["Date"].dt.strftime("%A")
    dim["Is_Weekend"] = dim["Day_of_Week"].isin([5, 6]).astype(int)
    return dim


def build_demand_exports() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _require(DEMAND_RAW, "Historical Product Demand dataset")
    print("  Loading demand from raw CSV …")
    demand = pd.read_csv(DEMAND_RAW)
    demand["Date"] = pd.to_datetime(demand["Date"], errors="coerce")
    demand["Order_Demand"] = (
        demand["Order_Demand"]
        .astype(str)
        .str.replace(r"[()]", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
    )
    demand.dropna(subset=["Date", "Order_Demand"], inplace=True)

    monthly = (
        demand.assign(YearMonth=demand["Date"].dt.to_period("M").astype(str))
        .groupby(["YearMonth", "Warehouse", "Product_Code", "Product_Category"], as_index=False)
        .agg(Total_Demand=("Order_Demand", "sum"), Days_Active=("Date", "nunique"))
    )

    warehouse = (
        demand.groupby("Warehouse", as_index=False)
        .agg(
            Total_Demand=("Order_Demand", "sum"),
            Avg_Daily_Demand=("Order_Demand", "mean"),
            SKU_Count=("Product_Code", "nunique"),
            First_Date=("Date", "min"),
            Last_Date=("Date", "max"),
        )
        .round(2)
    )

    product_summary = (
        demand.groupby(["Product_Code", "Product_Category", "Warehouse"], as_index=False)
        .agg(
            Total_Demand=("Order_Demand", "sum"),
            Avg_Daily_Demand=("Order_Demand", "mean"),
            Std_Daily_Demand=("Order_Demand", "std"),
            Days_Active=("Date", "nunique"),
        )
        .round(2)
        .sort_values("Total_Demand", ascending=False)
    )
    print(f"  Demand rows: {len(demand):,} | monthly aggregates: {len(monthly):,}")
    return monthly, warehouse, product_summary


def build_olist_delivery_kpis() -> pd.DataFrame:
    orders_path = os.path.join(OLIST_DIR, "olist_orders_dataset.csv")
    customers_path = os.path.join(OLIST_DIR, "olist_customers_dataset.csv")
    _require(orders_path, "Olist orders dataset")
    _require(customers_path, "Olist customers dataset")

    orders = pd.read_csv(orders_path)
    customers = pd.read_csv(customers_path)

    for col in [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
    ]:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

    delivered = orders.dropna(subset=["order_delivered_customer_date"]).copy()
    delivered["Delivery_Days"] = (
        delivered["order_delivered_customer_date"]
        - delivered["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    merged = delivered.merge(customers, on="customer_id", how="left")
    kpi = (
        merged.groupby("customer_state", as_index=False)
        .agg(
            Orders=("order_id", "count"),
            Avg_Delivery_Days=("Delivery_Days", "mean"),
            Median_Delivery_Days=("Delivery_Days", "median"),
            P90_Delivery_Days=("Delivery_Days", lambda x: np.percentile(x, 90)),
        )
        .round(2)
        .sort_values("Orders", ascending=False)
    )
    kpi.rename(columns={"customer_state": "State"}, inplace=True)
    print(f"  Olist delivery KPIs: {len(kpi)} Brazilian states")
    return kpi


def validate_inventory_against_demand(inv: pd.DataFrame) -> None:
    """Confirm inventory SKUs exist in the real demand dataset."""
    demand_codes = pd.read_csv(DEMAND_RAW, usecols=["Product_Code"])["Product_Code"].unique()
    demand_set = set(demand_codes)
    inv_codes = set(inv["Product_Code"].astype(str))
    overlap = len(inv_codes & demand_set)
    if overlap < MIN_INVENTORY_SKUS:
        raise ValueError(
            f"Only {overlap} inventory SKUs match Historical Product Demand — "
            "inventory_plan may not be from real demand data."
        )
    print(f"  [OK] {overlap:,} inventory SKUs verified against demand dataset")


def main() -> None:
    os.makedirs(PBI, exist_ok=True)
    print("=" * 60)
    print("POWER BI DATA PREPARATION  (REAL DATA ONLY)")
    print("=" * 60)

    validate_real_sources()

    df = load_dataco()
    fact = build_fact_orders(df)

    # ── Core order analytics ─────────────────────────────────────
    print("\n[1/14] fact_orders …")
    fact.to_csv(os.path.join(PBI, "fact_orders.csv"), index=False)
    print(f"      {len(fact):,} rows")

    print("[2/14] dim_date …")
    dim_date = build_dim_date(fact)
    dim_date.to_csv(os.path.join(PBI, "dim_date.csv"), index=False)
    print(f"      {len(dim_date)} days")

    print("[3/14] kpi_summary …")
    kpi_rows = [
        ("Total Revenue", fact["Revenue"].sum(), "$", "Currency"),
        ("Total Orders", len(fact), "", "Integer"),
        ("Total Profit", fact["Profit"].sum(), "$", "Currency"),
        (
            "Avg Profit Margin %",
            round(fact["Profit_Margin"].mean() * 100, 1),
            "%",
            "Percent",
        ),
        ("On-Time Rate %", round(fact["On_Time"].mean() * 100, 1), "%", "Percent"),
        (
            "Late Delivery Risk %",
            round(fact["Late_Risk"].mean() * 100, 1),
            "%",
            "Percent",
        ),
        ("Avg Order Value", round(fact["Revenue"].mean(), 2), "$", "Currency"),
        ("Avg Delay Days", round(fact["Delay_Days"].mean(), 2), "days", "Decimal"),
        ("Total Units Sold", int(fact["Quantity"].sum()), "units", "Integer"),
        (
            "Avg Discount Rate %",
            round(fact["Discount_Rate"].mean() * 100, 1),
            "%",
            "Percent",
        ),
    ]
    kpi_df = pd.DataFrame(kpi_rows, columns=["KPI", "Value", "Unit", "Format"])
    kpi_df.to_csv(os.path.join(PBI, "kpi_summary.csv"), index=False)

    dashboard_kpis = pd.DataFrame(
        [
            {
                "Total_Revenue": round(fact["Revenue"].sum(), 2),
                "Total_Orders": len(fact),
                "Total_Profit": round(fact["Profit"].sum(), 2),
                "On_Time_Rate_Pct": round(fact["On_Time"].mean() * 100, 1),
            }
        ]
    )
    dashboard_kpis.to_csv(os.path.join(PBI, "dashboard_kpis.csv"), index=False)

    print("[4/14] monthly_trend …")
    monthly = (
        fact.groupby(["Order_Year", "Order_Month"])
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("Revenue", "count"),
            Profit=("Profit", "sum"),
            On_Time_Pct=("On_Time", lambda x: round(x.mean() * 100, 1)),
            Avg_Delay_Days=("Delay_Days", "mean"),
            Units_Sold=("Quantity", "sum"),
            Avg_Discount=("Discount_Rate", lambda x: round(x.mean() * 100, 2)),
        )
        .round(2)
        .reset_index()
    )
    monthly["YearMonth"] = (
        monthly["Order_Year"].astype(str)
        + "-"
        + monthly["Order_Month"].astype(str).str.zfill(2)
    )
    monthly.sort_values(["Order_Year", "Order_Month"], inplace=True)
    monthly.to_csv(os.path.join(PBI, "monthly_trend.csv"), index=False)

    print("[5/14] category_kpis …")
    cat_kpis = (
        fact.groupby("Category")
        .agg(
            Orders=("Revenue", "count"),
            Revenue=("Revenue", "sum"),
            Profit=("Profit", "sum"),
            Avg_Order_Value=("Revenue", "mean"),
            On_Time_Pct=("On_Time", lambda x: round(x.mean() * 100, 1)),
            Profit_Margin=("Profit_Margin", lambda x: round(x.mean() * 100, 2)),
            Units_Sold=("Quantity", "sum"),
            Avg_Discount=("Discount_Rate", lambda x: round(x.mean() * 100, 2)),
        )
        .round(2)
        .reset_index()
        .sort_values("Revenue", ascending=False)
    )
    cat_kpis.to_csv(os.path.join(PBI, "category_kpis.csv"), index=False)

    print("[6/14] region_kpis …")
    region_kpis = (
        fact.groupby(["Market", "Region"])
        .agg(
            Orders=("Revenue", "count"),
            Revenue=("Revenue", "sum"),
            Profit=("Profit", "sum"),
            On_Time_Pct=("On_Time", lambda x: round(x.mean() * 100, 1)),
            Avg_Delay=("Delay_Days", "mean"),
        )
        .round(2)
        .reset_index()
        .sort_values("Revenue", ascending=False)
    )
    region_kpis.to_csv(os.path.join(PBI, "region_kpis.csv"), index=False)

    print("[7/14] country_kpis …")
    country_kpis = (
        fact.groupby(["Market", "Region", "Country"])
        .agg(
            Orders=("Revenue", "count"),
            Revenue=("Revenue", "sum"),
            Profit=("Profit", "sum"),
            On_Time_Pct=("On_Time", lambda x: round(x.mean() * 100, 1)),
        )
        .round(2)
        .reset_index()
        .sort_values("Revenue", ascending=False)
    )
    country_kpis.to_csv(os.path.join(PBI, "country_kpis.csv"), index=False)

    print("[8/14] shipping_kpis …")
    ship_kpis = (
        fact.groupby("Shipping_Mode")
        .agg(
            Orders=("Revenue", "count"),
            Revenue=("Revenue", "sum"),
            Profit=("Profit", "sum"),
            On_Time_Pct=("On_Time", lambda x: round(x.mean() * 100, 1)),
            Avg_Delay_Days=("Delay_Days", "mean"),
            Profit_Margin=("Profit_Margin", lambda x: round(x.mean() * 100, 2)),
        )
        .round(2)
        .reset_index()
    )
    ship_kpis.to_csv(os.path.join(PBI, "shipping_kpis.csv"), index=False)

    print("[9/14] supplier_performance …")
    supplier = (
        fact.groupby(["Shipping_Mode", "Category"])
        .agg(
            Orders=("Revenue", "count"),
            Revenue=("Revenue", "sum"),
            Profit=("Profit", "sum"),
            On_Time_Pct=("On_Time", lambda x: round(x.mean() * 100, 1)),
            Avg_Delay=("Delay_Days", "mean"),
        )
        .round(2)
        .reset_index()
    )
    supplier.to_csv(os.path.join(PBI, "supplier_performance.csv"), index=False)

    # ── Demand (real Kaggle time series) ─────────────────────────
    print("\n[10/14] demand exports …")
    demand_monthly, demand_warehouse, demand_product = build_demand_exports()
    demand_monthly.to_csv(os.path.join(PBI, "fact_demand_monthly.csv"), index=False)
    demand_warehouse.to_csv(os.path.join(PBI, "dim_warehouse.csv"), index=False)
    demand_product.to_csv(os.path.join(PBI, "dim_product_demand.csv"), index=False)

    # ── Olist logistics ──────────────────────────────────────────
    print("[11/14] olist_delivery_kpis …")
    olist_kpi = build_olist_delivery_kpis()
    olist_kpi.to_csv(os.path.join(PBI, "olist_delivery_kpis.csv"), index=False)

    # ── Pipeline outputs (must exist from prior phases) ───────────
    print("[12/14] inventory_plan …")
    inv = pd.read_csv(INV_PATH)
    validate_inventory_against_demand(inv)
    inv.to_csv(os.path.join(PBI, "inventory_plan.csv"), index=False)
    print(f"      {len(inv):,} SKUs (verified real)")

    print("[13/14] route_summary …")
    _require(ROUTE_PATH, "route_summary.csv (run phase3_route_optimization.py)")
    routes = pd.read_csv(ROUTE_PATH)
    routes.to_csv(os.path.join(PBI, "fact_routes.csv"), index=False)
    print(f"      {len(routes)} vehicles")

    print("[14/14] simulation exports …")
    _require(SIM_SUM_PATH, "simulation_summary.json (run safety_stock_simulation.py)")
    with open(SIM_SUM_PATH, encoding="utf-8") as f:
        sim_sum = json.load(f)

    sim_kpis = pd.DataFrame(
        [
            {"Metric": "SKUs Simulated", "Value": sim_sum.get("n_skus_simulated", 0)},
            {
                "Metric": "Monte Carlo Iterations",
                "Value": sim_sum.get("n_monte_carlo_iterations", 0),
            },
            {
                "Metric": "Service Level Target",
                "Value": sim_sum.get("service_level_target", 0),
            },
            {"Metric": "Lead Time (days)", "Value": sim_sum.get("lead_time_days_mean", 0)},
            {
                "Metric": "Baseline Stockout Prob",
                "Value": sim_sum.get("baseline_avg_stockout_prob", 0),
            },
            {
                "Metric": "Optimised Stockout Prob",
                "Value": sim_sum.get("optimised_avg_stockout_prob", 0),
            },
            {
                "Metric": "Stockout Reduction %",
                "Value": sim_sum.get("stockout_reduction_pct", 0),
            },
            {
                "Metric": "Baseline Overstock Units",
                "Value": sim_sum.get("baseline_avg_overstock_units", 0),
            },
            {
                "Metric": "Optimised Overstock Units",
                "Value": sim_sum.get("optimised_avg_overstock_units", 0),
            },
        ]
    )
    sim_kpis.to_csv(os.path.join(PBI, "simulation_kpis.csv"), index=False)

    abc_rows = [
        {
            "ABC_Class": cls,
            "Overstock_Reduction_Pct": vals.get("overstock_reduction_pct", 0),
            "Stockout_Reduction_Pct": vals.get("stockout_reduction_pct", 0),
        }
        for cls, vals in sim_sum.get("abc_breakdown", {}).items()
    ]
    pd.DataFrame(abc_rows).to_csv(os.path.join(PBI, "abc_breakdown.csv"), index=False)

    if os.path.exists(SIM_PATH):
        sim = pd.read_csv(SIM_PATH)
        sim.to_csv(os.path.join(PBI, "simulation_results.csv"), index=False)
        print(f"      simulation_results: {len(sim):,} rows")

    # ── Data source manifest (audit trail) ─────────────────────────
    manifest = pd.DataFrame(
        [
            {
                "Table": "fact_orders",
                "Source": "DataCoSupplyChainDataset.csv",
                "Rows": len(fact),
                "Type": "Transactional (real)",
            },
            {
                "Table": "fact_demand_monthly",
                "Source": "Historical Product Demand.csv",
                "Rows": len(demand_monthly),
                "Type": "Aggregated (real)",
            },
            {
                "Table": "inventory_plan",
                "Source": "phase4 output",
                "Rows": len(inv),
                "Type": "Model output on real demand",
            },
            {
                "Table": "fact_routes",
                "Source": "phase3 VRP on Olist geo",
                "Rows": len(routes),
                "Type": "Optimization output (real stops)",
            },
            {
                "Table": "olist_delivery_kpis",
                "Source": "Olist orders + customers",
                "Rows": len(olist_kpi),
                "Type": "Aggregated (real)",
            },
            {
                "Table": "simulation_*",
                "Source": "Monte Carlo on inventory_plan",
                "Rows": sim_sum.get("n_skus_simulated", 0),
                "Type": "Simulation (based on real SKU stats)",
            },
        ]
    )
    manifest.to_csv(os.path.join(PBI, "data_source_manifest.csv"), index=False)

    certificate = {
        "certified_real_data_only": True,
        "synthetic_data_allowed": False,
        "validated_at": pd.Timestamp.now().isoformat(),
        "sources": {
            "dataco_raw_csv": DATACO_RAW,
            "dataco_rows": len(fact),
            "demand_raw_csv": DEMAND_RAW,
            "demand_monthly_rows": len(demand_monthly),
            "inventory_skus": len(inv),
            "inventory_verified_against_demand": True,
            "simulation_provenance": sim_sum.get("data_provenance"),
            "olist_states": len(olist_kpi),
        },
        "note": (
            "simulation_* tables are Monte Carlo projections on real SKU demand "
            "statistics — not synthetic order transactions."
        ),
    }
    with open(os.path.join(PBI, "REAL_DATA_CERTIFICATE.json"), "w", encoding="utf-8") as f:
        json.dump(certificate, f, indent=2)

    print("\n" + "=" * 60)
    print("[SUCCESS] Power BI data files generated (real data only)")
    print(f"  Location: {PBI}")
    print("=" * 60)
    total_size = 0
    for fname in sorted(os.listdir(PBI)):
        size = os.path.getsize(os.path.join(PBI, fname))
        total_size += size
        print(f"  {fname:42s}  {size / 1024:7.1f} KB")
    print(f"\n  Total: {total_size / 1024 / 1024:.1f} MB")
    print("\nNext: python powerbi/build_powerbi_desktop.py")
    print("Then: double-click powerbi/OPEN_IN_POWER_BI.bat")


if __name__ == "__main__":
    main()
