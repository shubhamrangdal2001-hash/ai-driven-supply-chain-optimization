# Microsoft Power BI — AI-Driven Supply Chain

Interactive analytics dashboard built on **real Kaggle datasets only** — no synthetic or mock data.

## Data Sources (100% Real)

| Table | Origin | Rows |
|-------|--------|------|
| `fact_orders` | [DataCo Supply Chain](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) | 180,519 |
| `fact_demand_monthly` | [Historical Product Demand](https://www.kaggle.com/datasets/aungpyaeap/product-demand-forecasting) | 136,026 |
| `dim_product_demand` | Same demand dataset (SKU summary) | ~2,160 |
| `inventory_plan` | Phase 4 EOQ/ABC on real demand | 2,160 |
| `fact_routes` | Phase 3 VRP on Olist geolocation | 5 |
| `olist_delivery_kpis` | [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) | 27 states |
| `simulation_*` | Monte Carlo on real SKU stats | 500 SKUs |

See `reports/powerbi/data_source_manifest.csv` for a full audit trail.

After each run, `powerbi_data_prep.py` writes **`REAL_DATA_CERTIFICATE.json`** — it only succeeds when:

- Raw Kaggle CSVs are on disk (DataCo ~180k rows, Demand ~1M rows)
- `inventory_plan.csv` has 2,000+ SKUs cross-checked against the demand file
- `simulation_summary.json` has `data_provenance: real_inventory_plan`
- **No synthetic fallbacks** — the script exits with an error if validation fails

## Quick Start

### 1. Generate data files

```powershell
cd "D:\AI-Driven Supply Chain"
python powerbi_data_prep.py
```

**Prerequisites** (real pipeline outputs — not synthetic):

- `DataCo Smart Supply Chain/DataCoSupplyChainDataset.csv`
- `Product Demand Forecasting/Historical Product Demand.csv`
- `Olist Brazilian E-Commerce/olist_orders_dataset.csv`
- `reports/inventory_plan.csv` → run `phase4_inventory_optimization.py`
- `reports/route_summary.csv` → run `phase3_route_optimization.py`
- `reports/simulation_summary.json` → run `safety_stock_simulation.py`

### 2. Build semantic model (optional — already generated)

```powershell
cd powerbi
python build_semantic_model.py
```

### 3. Build the Desktop project (required once)

```powershell
cd powerbi
python build_powerbi_desktop.py
```

This creates `model.bim` (compatible with all Power BI Desktop versions — no preview features needed).

### 4. Open in Power BI Desktop

**You must have [Power BI Desktop](https://aka.ms/pbidesktop) installed** (free from Microsoft).

| Method | Action |
|--------|--------|
| **Easiest** | Double-click **`OPEN_POWERBI.bat`** in the project root |
| From powerbi/ | Double-click `powerbi/OPEN_IN_POWER_BI.bat` |
| Manual | Double-click `powerbi/SupplyChainAnalytics.pbip` (requires Power BI Desktop installed) |

After opening:

1. **Home → Refresh** (CSVs load from `reports\powerbi` automatically)
2. Click **Transform data** → **Close & Apply**, or **Home → Refresh**
3. Add visuals using the layout guide below

**If the file still won't open:** Power BI Desktop is likely not installed. The `.pbip` file is not a standalone app — it requires Power BI Desktop (like `.docx` needs Word).

## Dashboard Pages (pre-built)

The PBIP project includes **4 complete pages** with **18 pre-built visuals**. After refresh, charts populate automatically:

| Page | Visuals |
|------|---------|
| **Executive Summary** | 4 KPI cards, revenue trend line, category donut, region bar |
| **Logistics & Shipping** | Shipping revenue & on-time bars, country bar, Brazil delivery table |
| **Inventory & Simulation** | ABC stockout bar, EOQ scatter, simulation & inventory tables |
| **Demand Forecasting** | Units trend line, warehouse bar, category treemap |

Rebuild visuals after data changes: `python build_powerbi_desktop.py`

### Manual visual reference (optional)

### Page 1 — Executive Summary

| Visual | Fields |
|--------|--------|
| Card × 4 | `[Total Revenue]`, `[Total Orders]`, `[Total Profit]`, `[On-Time Delivery %]` |
| Line chart | `Dim_Date[YearMonth]` × `[Total Revenue]` |
| Donut | `Fact_Orders[Category]` × `[Total Revenue]` |
| Map | `Fact_Orders[Country]` × `[Total Revenue]` |

### Page 2 — Logistics & Shipping

| Visual | Fields |
|--------|--------|
| Bar chart | `Shipping_KPIs[Shipping_Mode]` × `Revenue`, `On_Time_Pct` |
| Matrix | `Supplier_Performance[Shipping_Mode]` × `Category` × `On_Time_Pct` |
| Column chart | `Region_KPIs[Region]` × `Revenue` |
| Table | `Olist_Delivery_KPIs` — State, Orders, Avg_Delivery_Days |

### Page 3 — Inventory & Simulation

| Visual | Fields |
|--------|--------|
| Card | `Simulation_KPIs[Value]` filtered by Metric |
| Bar chart | `ABC_Breakdown[ABC_Class]` × `Stockout_Reduction_Pct` |
| Scatter | `Inventory_Plan[annual_demand]` × `safety_stock`, legend `ABC` |
| Table | Top 20 SKUs by `annual_demand` |

### Page 4 — Demand Forecasting

| Visual | Fields |
|--------|--------|
| Line chart | `Fact_Demand_Monthly[YearMonth]` × `Total_Demand` |
| Bar chart | `Dim_Warehouse[Warehouse]` × `Total_Demand` |
| Treemap | `Fact_Demand_Monthly[Product_Category]` × `Total_Demand` |

## DAX Measures (included in model)

Pre-built on `Fact_Orders`:

- `Total Revenue`, `Total Profit`, `Total Orders`
- `On-Time Delivery %`, `Avg Order Value`, `Profit Margin %`

Additional measures in `Measures.dax` for copy-paste.

## Project Structure

```
powerbi/
├── SupplyChainAnalytics.pbip          ← Open this in Power BI Desktop
├── SupplyChainAnalytics.Report/       ← Report pages (PBIR)
├── SupplyChainAnalytics.SemanticModel/← TMDL data model
├── build_semantic_model.py
├── Measures.dax
└── README.md

reports/powerbi/                       ← CSV data files (refresh source)
```

## Manual CSV Import (alternative)

If PBIP does not open, import CSVs manually:

1. **Home → Get Data → Text/CSV**
2. Select all files in `reports\powerbi\`
3. Set relationships: `Fact_Orders[Order_Date]` → `Dim_Date[Date]`
4. Create measures from `Measures.dax`

## Regenerating After Pipeline Changes

```powershell
python phase3_route_optimization.py      # if routes missing
python phase4_inventory_optimization.py    # if inventory missing
python safety_stock_simulation.py        # if simulation missing
python powerbi_data_prep.py              # refresh all BI CSVs
```

Then refresh the semantic model in Power BI Desktop.

## Notes

- **Simulation tables** are Monte Carlo modeled on real SKU demand statistics — labeled in `data_source_manifest.csv`
- **No synthetic fallbacks** — `powerbi_data_prep.py` raises `FileNotFoundError` if sources are missing
- Do **not** use `powerbi_dashboard.html` — it contains hardcoded sample metrics, not live data
