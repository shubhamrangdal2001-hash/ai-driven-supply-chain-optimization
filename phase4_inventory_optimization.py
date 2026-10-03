"""
Phase 4 – Inventory Optimisation
Reads demand_grouped + dataco_clean from Phase 1 parquet output.
Computes EOQ, Safety Stock, Reorder Point, ABC classification,
and trains a Random Forest stockout-risk classifier.
Saves inventory_plan.csv to reports/.
"""

import os, warnings
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

warnings.filterwarnings('ignore')

ROOT    = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, 'reports')

print("=" * 60)
print("PHASE 4 - INVENTORY OPTIMISATION")
print("=" * 60)

# ── Load data ─────────────────────────────────────────────────────────
demand_grouped = pd.read_parquet(os.path.join(REPORTS, 'demand_grouped.parquet'))
df             = pd.read_parquet(os.path.join(REPORTS, 'dataco_clean.parquet'))
print(f"  demand_grouped: {demand_grouped.shape}  |  dataco: {df.shape}")

SERVICE_LEVEL = 0.95
Z_SCORE       = norm.ppf(SERVICE_LEVEL)

# ── 1. Demand statistics per SKU ─────────────────────────────────────
sku_stats = (
    demand_grouped.groupby('Product_Code')['Order_Demand']
    .agg(
        avg_daily_demand='mean',
        std_daily_demand='std',
        total_demand='sum',
        days_active='count'
    ).reset_index()
)
sku_stats['std_daily_demand'] = sku_stats['std_daily_demand'].fillna(0)
sku_stats['cv'] = sku_stats['std_daily_demand'] / sku_stats['avg_daily_demand'].clip(1)
print(f"  SKUs: {len(sku_stats):,}")

# ── 2. EOQ ────────────────────────────────────────────────────────────
ORDERING_COST    = 50
HOLDING_COST_PCT = 0.2
AVG_UNIT_COST    = 10
HOLDING_COST     = HOLDING_COST_PCT * AVG_UNIT_COST

sku_stats['annual_demand'] = sku_stats['avg_daily_demand'] * 365
sku_stats['EOQ'] = np.sqrt(
    (2 * sku_stats['annual_demand'] * ORDERING_COST) / HOLDING_COST
).round(0)

# ── 3. Safety Stock & Reorder Point ──────────────────────────────────
LEAD_TIME_DAYS     = 7
LEAD_TIME_STD_DAYS = 2

sku_stats['demand_during_lead'] = sku_stats['avg_daily_demand'] * LEAD_TIME_DAYS
sku_stats['safety_stock'] = (
    Z_SCORE * np.sqrt(
        LEAD_TIME_DAYS * sku_stats['std_daily_demand']**2
        + (sku_stats['avg_daily_demand'] * LEAD_TIME_STD_DAYS)**2
    )
).round(0)
sku_stats['reorder_point'] = (
    sku_stats['demand_during_lead'] + sku_stats['safety_stock']
).round(0)

# ── 4. ABC Classification ─────────────────────────────────────────────
sku_stats['revenue_pct']   = sku_stats['total_demand'] / sku_stats['total_demand'].sum()
sku_stats = sku_stats.sort_values('revenue_pct', ascending=False)
sku_stats['cumulative_pct'] = sku_stats['revenue_pct'].cumsum()

def abc_class(cum):
    if   cum <= 0.80: return 'A'
    elif cum <= 0.95: return 'B'
    else:             return 'C'

sku_stats['ABC'] = sku_stats['cumulative_pct'].apply(abc_class)
print(f"\n  ABC Distribution:\n{sku_stats['ABC'].value_counts().to_string()}")

# ── 5. Stockout Risk Classifier ───────────────────────────────────────
needed_cols = [
    'Order Item Quantity', 'Days for shipment (scheduled)',
    'Days for shipping (real)', 'order_month',
    'order_quarter', 'Shipping Mode', 'Category Name'
]
available = [c for c in needed_cols if c in df.columns]
if len(available) < 5:
    print("\n  [WARN] Not enough columns for stockout classifier – skipping.")
else:
    df_sc = df[available].dropna()
    df_sc = df_sc.copy()
    df_sc['stockout_risk'] = (
        df_sc['Days for shipping (real)'] > df_sc['Days for shipment (scheduled)'] * 1.5
    ).astype(int)

    cat_cols = [c for c in ['Shipping Mode', 'Category Name'] if c in df_sc.columns]
    df_enc   = pd.get_dummies(df_sc, columns=cat_cols, drop_first=True)
    feat_cols = [c for c in df_enc.columns if c != 'stockout_risk']

    X     = df_enc[feat_cols].values
    y     = df_enc['stockout_risk'].values
    split = int(len(X) * 0.8)

    clf = RandomForestClassifier(n_estimators=100, max_depth=8,
                                 class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(X[:split], y[:split])
    report = classification_report(y[split:], clf.predict(X[split:]))
    print(f"\n  Stockout Risk Classifier:\n{report}")

# ── 6. Save ───────────────────────────────────────────────────────────
out_path = os.path.join(REPORTS, 'inventory_plan.csv')
sku_stats.to_csv(out_path, index=False)
print(f"  Inventory plan saved -> {out_path}")
print("\nPhase 4 complete.")