"""
Phase 1 – Data Loading & EDA
Loads all datasets, cleans DataCo, engineers features, produces EDA plots.
Exports `demand_grouped` and `df` to Parquet for downstream phases.
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ── Resolve root directory (always relative to this file) ─────────────
ROOT    = os.path.dirname(os.path.abspath(__file__))
DATA_DC = os.path.join(ROOT, 'DataCo Smart Supply Chain', 'DataCoSupplyChainDataset.csv')
DATA_DM = os.path.join(ROOT, 'Product Demand Forecasting', 'Historical Product Demand.csv')
DATA_OR = os.path.join(ROOT, 'Olist Brazilian E-Commerce')
REPORTS = os.path.join(ROOT, 'reports')
os.makedirs(REPORTS, exist_ok=True)

# ── 1. Load DataCo (primary dataset) ─────────────────────────────────
print("=" * 60)
print("PHASE 1 – DATA LOADING & EDA")
print("=" * 60)
print("\n[1/5] Loading DataCo Supply Chain dataset …")
df = pd.read_csv(DATA_DC, encoding='latin-1')
print(f"  DataCo shape: {df.shape}")

# ── 2. Load demand forecasting data ──────────────────────────────────
print("[2/5] Loading Product Demand dataset …")
demand = pd.read_csv(DATA_DM)
demand['Date']         = pd.to_datetime(demand['Date'], errors='coerce')
demand['Order_Demand'] = (
    demand['Order_Demand']
    .astype(str).str.replace(r'[()]', '', regex=True)
    .pipe(pd.to_numeric, errors='coerce')
)
demand.dropna(subset=['Date', 'Order_Demand'], inplace=True)
print(f"  Demand shape: {demand.shape}")

# ── 3. Load Olist for route / geo analysis ────────────────────────────
print("[3/5] Loading Olist datasets …")
orders    = pd.read_csv(os.path.join(DATA_OR, 'olist_orders_dataset.csv'))
items     = pd.read_csv(os.path.join(DATA_OR, 'olist_order_items_dataset.csv'))
sellers   = pd.read_csv(os.path.join(DATA_OR, 'olist_sellers_dataset.csv'))
customers = pd.read_csv(os.path.join(DATA_OR, 'olist_customers_dataset.csv'))
print(f"  Orders: {orders.shape}  Items: {items.shape}")

# ── 4. Clean DataCo ──────────────────────────────────────────────────
print("[4/5] Cleaning & feature engineering …")
df['order date (DateOrders)']    = pd.to_datetime(df['order date (DateOrders)'],    errors='coerce')
df['shipping date (DateOrders)'] = pd.to_datetime(df['shipping date (DateOrders)'], errors='coerce')

null_pct = df.isnull().mean()
df.drop(columns=null_pct[null_pct > 0.4].index, inplace=True)

num_cols = df.select_dtypes('number').columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
print(f"  Nulls after cleaning: {df.isnull().sum().sum()}")

# ── 5. Feature Engineering ───────────────────────────────────────────
df['order_year']      = df['order date (DateOrders)'].dt.year
df['order_month']     = df['order date (DateOrders)'].dt.month
df['order_quarter']   = df['order date (DateOrders)'].dt.quarter
df['order_dayofweek'] = df['order date (DateOrders)'].dt.dayofweek
df['delivery_delay']  = (
    df['Days for shipment (scheduled)'] - df['Days for shipping (real)']
)
df['is_late']         = (df['delivery_delay'] < 0).astype(int)
df['profit_margin']   = df['Benefit per order'] / (df['Sales'] + 1e-9)

# Demand lag/rolling features
demand_grouped = (
    demand.groupby(['Warehouse', 'Product_Code', 'Date'])['Order_Demand']
    .sum().reset_index()
)
demand_grouped.sort_values('Date', inplace=True)
for lag in [7, 14, 28]:
    demand_grouped[f'lag_{lag}'] = (
        demand_grouped.groupby(['Warehouse', 'Product_Code'])['Order_Demand'].shift(lag)
    )
demand_grouped['rolling_mean_7'] = (
    demand_grouped.groupby(['Warehouse', 'Product_Code'])['Order_Demand']
    .transform(lambda x: x.rolling(7).mean())
)
demand_grouped['rolling_std_7'] = (
    demand_grouped.groupby(['Warehouse', 'Product_Code'])['Order_Demand']
    .transform(lambda x: x.rolling(7).std())
)
print("  Feature engineering complete.")

# ── 6. EDA Visualisations ─────────────────────────────────────────────
print("[5/5] Generating EDA plots …")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

monthly = df.groupby('order_month')['Sales'].sum()
axes[0, 0].bar(monthly.index, monthly.values, color='steelblue')
axes[0, 0].set_title('Monthly Sales Volume')
axes[0, 0].set_xlabel('Month'); axes[0, 0].set_ylabel('Total Sales ($)')

ship_mode = df['Shipping Mode'].value_counts()
axes[0, 1].pie(ship_mode, labels=ship_mode.index, autopct='%1.1f%%', startangle=90)
axes[0, 1].set_title('Shipping Mode Distribution')

axes[1, 0].hist(df['delivery_delay'].dropna(), bins=30, color='salmon', edgecolor='white')
axes[1, 0].set_title('Delivery Delay Distribution')
axes[1, 0].axvline(0, color='red', linestyle='--')

sample = demand_grouped[demand_grouped['Product_Code'] == demand_grouped['Product_Code'].iloc[0]]
axes[1, 1].plot(sample['Date'], sample['Order_Demand'])
axes[1, 1].set_title('Sample Product Demand Over Time')

plt.tight_layout()
out_path = os.path.join(REPORTS, 'eda_overview.png')
plt.savefig(out_path, dpi=150)
plt.close()
print(f"  EDA plot saved → {out_path}")

# ── 7. Export for downstream phases ──────────────────────────────────
df.to_parquet(os.path.join(REPORTS, 'dataco_clean.parquet'), index=False)
demand_grouped.to_parquet(os.path.join(REPORTS, 'demand_grouped.parquet'), index=False)
print("\nPhase 1 complete. Artefacts saved to ./reports/")