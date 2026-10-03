# 🔗 AI-Driven Supply Chain Optimization

> A full end-to-end ML system for **demand forecasting** and **logistics route optimization**, built with Python, PyTorch, XGBoost, OR-Tools, FastAPI, and Streamlit.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Directory Structure](#-directory-structure)
- [Datasets](#-datasets)
- [Installation](#-installation)
- [How to Run — Phase by Phase](#-how-to-run--phase-by-phase)
- [Model Results](#-model-results)
- [API Reference](#-api-reference)
- [Dashboard](#-dashboard)
- [Tech Stack](#-tech-stack)
- [Key Concepts](#-key-concepts)

---

## 🧭 Project Overview

This project builds a production-ready AI supply chain system with three core capabilities:

| Capability | Method | Dataset |
|---|---|---|
| Demand Forecasting | SARIMA → XGBoost → LSTM | Product Demand Forecasting |
| Route Optimization | Nearest-Neighbour VRP | Olist Brazilian E-Commerce |
| Inventory Planning | EOQ + Safety Stock + ABC + RF Classifier | DataCo Supply Chain |

The pipeline runs across **5 phases**, from raw data ingestion through to a live REST API and an interactive Streamlit dashboard.

---

## 🏗 Architecture

```
Raw Datasets
     │
     ▼
Phase 1: Data Loading & EDA
  └── dataco_clean.parquet
  └── demand_grouped.parquet
     │
     ├──► Phase 2a: SARIMA Baseline
     ├──► Phase 2b: XGBoost (Optuna HPO)  ──► xgb_demand.pkl
     └──► Phase 2c: LSTM (PyTorch)         ──► lstm_demand.pt + lstm_scaler.pkl
     │
     ├──► Phase 3: Route Optimization      ──► route_summary.csv + route_optimization.png
     │
     ├──► Phase 4: Inventory Optimization  ──► inventory_plan.csv
     │
     └──► Phase 5a: FastAPI Server         ──► REST API on :8000
          Phase 5b: Streamlit Dashboard    ──► UI on :8501
```

---

## 📁 Directory Structure

```
supply-chain-ai/
│
├── phase1_data_loading.py          # EDA, cleaning, feature engineering
├── phase2a_arima_baseline.py       # SARIMA model
├── phase2b_xgboost_forecast.py     # XGBoost + Optuna HPO
├── phase2c_lstm_model.py           # PyTorch LSTM
├── phase3_route_optimization.py    # Nearest-Neighbour VRP
├── phase4_inventory_optimization.py# EOQ, safety stock, ABC, RF classifier
├── phase5a_fastapi_server.py       # REST API (FastAPI + uvicorn)
├── phase5b_streamlit_dashboard.py  # Interactive dashboard
│
├── DataCo Smart Supply Chain/
│   └── DataCoSupplyChainDataset.csv
├── Product Demand Forecasting/
│   └── Historical Product Demand.csv
├── Olist Brazilian E-Commerce/
│   ├── olist_orders_dataset.csv
│   ├── olist_order_items_dataset.csv
│   ├── olist_sellers_dataset.csv
│   ├── olist_customers_dataset.csv
│   └── olist_geolocation_dataset.csv
│
├── src/
│   └── models/
│       ├── xgb_demand.pkl          # Saved XGBoost model
│       ├── lstm_demand.pt          # Saved LSTM weights
│       └── lstm_scaler.pkl         # MinMaxScaler for LSTM
│
└── reports/
    ├── dataco_clean.parquet        # Cleaned DataCo (Phase 1 output)
    ├── demand_grouped.parquet      # Aggregated demand (Phase 1 output)
    ├── eda_overview.png            # EDA plots
    ├── route_optimization.png      # VRP route map
    ├── route_summary.csv           # Per-vehicle route metrics
    └── inventory_plan.csv          # EOQ / safety stock per SKU
```

---

## 📦 Datasets

Download all four datasets from Kaggle before running:

| Dataset | Kaggle Link | Used In |
|---|---|---|
| **DataCo Smart Supply Chain** | [Download ↗](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) | Phases 1, 4, 5 |
| **Product Demand Forecasting** | [Download ↗](https://www.kaggle.com/datasets/felixzhao/productdemandforecasting) | Phases 1, 2a, 2b, 2c |
| **Olist Brazilian E-Commerce** | [Download ↗](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) | Phases 1, 3 |
| **Supply Chain Shipment Pricing** | [Download ↗](https://www.kaggle.com/datasets/divyeshardeshana/supply-chain-shipment-pricing-data) | Optional / Phase 4 |

### Quick Download via Kaggle CLI

```bash
# Install Kaggle CLI and place your API token at ~/.kaggle/kaggle.json
pip install kaggle

kaggle datasets download -d shashwatwork/dataco-smart-supply-chain-for-big-data-analysis \
    --unzip -p "./DataCo Smart Supply Chain"

kaggle datasets download -d felixzhao/productdemandforecasting \
    --unzip -p "./Product Demand Forecasting"

kaggle datasets download -d olistbr/brazilian-ecommerce \
    --unzip -p "./Olist Brazilian E-Commerce"
```

---

## ⚙️ Installation

### Requirements

- Python 3.11+
- pip

### Install all dependencies

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn \
            xgboost lightgbm statsmodels torch torchvision \
            optuna joblib scipy fastapi uvicorn pydantic \
            streamlit requests pyarrow
```

Or with a `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Docker (optional)

```bash
docker build -t supply-chain-ai .
docker run -p 8000:8000 -p 8501:8501 supply-chain-ai
```

---

## 🚀 How to Run — Phase by Phase

Run phases **in order**. Each phase reads outputs written by the previous one.

### Phase 1 — Data Loading & EDA

Loads all datasets, cleans DataCo, engineers lag/rolling features, and exports parquet files consumed by all downstream phases.

```bash
python phase1_data_loading.py
```

**Outputs:** `reports/dataco_clean.parquet`, `reports/demand_grouped.parquet`, `reports/eda_overview.png`

---

### Phase 2a — SARIMA Baseline

Fits a SARIMA(1,1,1)(1,1,1,52) model on the top-selling product's weekly demand. Provides a statistical baseline to beat.

```bash
python phase2a_arima_baseline.py
```

> ⚠️ SARIMA fitting can take 1–2 minutes due to weekly seasonal order.

---

### Phase 2b — XGBoost Forecast

Trains an XGBoost regressor with lag and calendar features. Uses **Optuna** for 30-trial hyperparameter search. Saves the best model to `src/models/xgb_demand.pkl`.

```bash
python phase2b_xgboost_forecast.py
```

---

### Phase 2c — LSTM Forecast

Trains a 2-layer LSTM in PyTorch on 30-day sequences with a 7-day forecast horizon. Uses `HuberLoss` and a learning-rate scheduler. Saves weights + scaler.

```bash
python phase2c_lstm_model.py
```

> GPU is used automatically if available (`torch.cuda.is_available()`).

---

### Phase 3 — Route Optimization

Builds a Haversine distance matrix for the top 200 delivery stops from Olist, then solves a 5-vehicle routing problem using a **greedy nearest-neighbour heuristic**. Saves a route map PNG and a CSV summary.

```bash
python phase3_route_optimization.py
```

---

### Phase 4 — Inventory Optimization

Computes per-SKU **EOQ**, **safety stock**, and **reorder point** at a 95% service level. Classifies products as A / B / C. Trains a Random Forest classifier to predict stockout risk from DataCo shipment features.

```bash
python phase4_inventory_optimization.py
```

---

### Phase 5a — FastAPI Server

Loads the LSTM model, XGBoost model, and inventory plan, then serves them as REST endpoints on port 8000.

```bash
python phase5a_fastapi_server.py
```

Interactive API docs available at: **http://localhost:8000/docs**

---

### Phase 5b — Streamlit Dashboard

Launches a 4-page interactive dashboard. Requires Phase 5a running for live demand forecasts (dashboard shows a warning badge if the API is offline).

```bash
streamlit run phase5b_streamlit_dashboard.py
```

Dashboard available at: **http://localhost:8501**

---

## 📊 Model Results

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| SARIMA baseline | — | — | ~18% |
| XGBoost + Optuna | — | — | ~12% |
| **LSTM (best)** | — | — | **~9%** |

> Exact metrics depend on the specific product and train/test split. Run the phases to get results for your data.

**Route Optimization:**

| Metric | Value |
|---|---|
| Vehicles | 5 |
| Stops optimized | 200 |
| Algorithm | Greedy Nearest-Neighbour |

**Inventory Planning:**

| Parameter | Value |
|---|---|
| Service Level | 95% |
| Lead Time | 7 days |
| Ordering Cost | $50 / order |
| Holding Cost | 20% of unit cost / year |

---

## 🔌 API Reference

Base URL: `http://localhost:8000`

### `GET /health`
Returns model loading status.

```json
{
  "status": "ok",
  "xgb_loaded": true,
  "lstm_loaded": true,
  "plan_loaded": true
}
```

### `POST /predict/demand`
Returns a 7-day demand forecast using the LSTM model.

**Request body:**
```json
{
  "product_code": "Product_1359",
  "warehouse": "Whse_A",
  "horizon_days": 7,
  "recent_demand": [120, 95, 200, 180, 145, 160, 130, "...30 values total"]
}
```

**Response:**
```json
{
  "product_code": "Product_1359",
  "warehouse": "Whse_A",
  "forecast_7days": [142.3, 138.7, 155.1, 149.0, 161.4, 144.8, 150.2],
  "total_forecast": 1041.5
}
```

### `GET /inventory/{product_code}`
Returns EOQ, safety stock, reorder point, and ABC class for a single SKU.

**Response:**
```json
{
  "Product_Code": "Product_1359",
  "EOQ": 843.0,
  "safety_stock": 215.0,
  "reorder_point": 387.0,
  "ABC": "A"
}
```

### `GET /inventory?abc_class=A&limit=50`
Lists SKUs filtered by ABC class (optional). Returns up to `limit` records.

---

## 📺 Dashboard

The Streamlit dashboard has four pages:

| Page | Description |
|---|---|
| **Overview** | KPI metrics, monthly revenue trend, shipping mode breakdown |
| **Demand Forecast** | Enter any product code + warehouse → get a live 7-day LSTM forecast from the API |
| **Route Optimizer** | Per-vehicle distance table + route map from Phase 3 |
| **Inventory Plan** | Filter SKUs by ABC class; scatter plot of EOQ vs safety stock; reorder point histogram |

### Microsoft Power BI

A full **Power BI Project (PBIP)** uses real Kaggle data only — no synthetic fallbacks.

**One-click open:**

```text
Double-click  OPEN_POWERBI.bat   (project root)
```

Or from terminal:

```powershell
.\OPEN_POWERBI.bat
```

Manual prep (optional):

```bash
python powerbi_data_prep.py
python powerbi/build_powerbi_desktop.py
```

| Page | Pre-built visuals |
|---|---|
| **Executive Summary** | 4 KPI cards, revenue trend, category donut, region bar |
| **Logistics & Shipping** | Shipping bars, country revenue, Brazil delivery table |
| **Inventory & Simulation** | ABC bar, inventory scatter, simulation tables |
| **Demand Forecasting** | Demand trend, warehouse bar, category treemap |

See [`powerbi/README.md`](powerbi/README.md) for setup, DAX measures, and visual layout guide.

---

## 🛠 Tech Stack

| Area | Libraries |
|---|---|
| Data | `pandas`, `numpy`, `pyarrow` |
| Visualisation | `matplotlib`, `seaborn`, `plotly` |
| ML / Forecasting | `scikit-learn`, `xgboost`, `statsmodels` |
| Deep Learning | `torch` (PyTorch), `torch.nn` |
| HPO | `optuna` |
| Optimization | Greedy NN heuristic (extensible to `ortools`) |
| Inventory | `scipy.stats`, `sklearn.ensemble` |
| API | `fastapi`, `uvicorn`, `pydantic` |
| Dashboard | `streamlit`, `plotly.express` |
| BI / Analytics | Microsoft Power BI Desktop (PBIP + TMDL) |
| Serialization | `joblib`, `parquet` |

---

## 📖 Key Concepts

**EOQ (Economic Order Quantity)** — minimises total ordering + holding cost:
```
EOQ = sqrt( (2 × Annual_Demand × Ordering_Cost) / Holding_Cost )
```

**Safety Stock** — buffer stock at a 95% service level:
```
SS = Z × sqrt( L × σ_d² + (d̄ × σ_L)² )
```
where `Z=1.645`, `L` = lead time days, `σ_d` = daily demand std, `σ_L` = lead time std.

**Reorder Point:**
```
ROP = (Average_Daily_Demand × Lead_Time) + Safety_Stock
```

**ABC Classification:**
- **A items** — top 80% of revenue; tightest inventory control
- **B items** — next 15% (80–95% cumulative)
- **C items** — remaining 5%; relaxed control

**VRP (Vehicle Routing Problem)** — assigns delivery stops to vehicles minimising total distance, subject to vehicle capacity constraints.

---

## 🗓 Project Timeline

| Phase | Description | Duration |
|---|---|---|
| 1 | Data loading, cleaning, EDA | Weeks 1–2 |
| 2 | Demand forecasting (SARIMA, XGBoost, LSTM) | Weeks 3–5 |
| 3 | Route optimization | Weeks 6–8 |
| 4 | Inventory optimization | Weeks 9–10 |
| 5 | API + dashboard + deployment | Weeks 11–12 |

---

## 📄 License

This project is for educational and research purposes.
