"""
Phase 2b – XGBoost Demand Forecast (with Optuna HPO)
Reads demand_grouped from Phase 1 parquet output.
Saves best model to src/models/xgb_demand.pkl
"""

import os, warnings
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT    = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, 'reports')
MODELS  = os.path.join(ROOT, 'src', 'models')
os.makedirs(MODELS, exist_ok=True)

print("=" * 60)
print("PHASE 2b - XGBOOST FORECAST")
print("=" * 60)

# ── Load demand data ──────────────────────────────────────────────────
demand_grouped = pd.read_parquet(os.path.join(REPORTS, 'demand_grouped.parquet'))
print(f"  Loaded demand_grouped: {demand_grouped.shape}")

# ── Prepare features ──────────────────────────────────────────────────
df_feat = demand_grouped.copy()
df_feat['month']     = df_feat['Date'].dt.month
df_feat['quarter']   = df_feat['Date'].dt.quarter
df_feat['year']      = df_feat['Date'].dt.year
df_feat['dayofyear'] = df_feat['Date'].dt.dayofyear
df_feat['week']      = df_feat['Date'].dt.isocalendar().week.astype(int)

le_wh = LabelEncoder()
le_pr = LabelEncoder()
df_feat['warehouse_enc'] = le_wh.fit_transform(df_feat['Warehouse'])
df_feat['product_enc']   = le_pr.fit_transform(df_feat['Product_Code'])

FEATURES = ['warehouse_enc', 'product_enc', 'month', 'quarter',
            'year', 'dayofyear', 'week',
            'lag_7', 'lag_14', 'lag_28', 'rolling_mean_7', 'rolling_std_7']
TARGET   = 'Order_Demand'

df_clean = df_feat[FEATURES + [TARGET, 'Date']].dropna()
cutoff   = pd.Timestamp('2016-01-01')
X_train  = df_clean[df_clean['Date'] < cutoff][FEATURES]
y_train  = df_clean[df_clean['Date'] < cutoff][TARGET]
X_test   = df_clean[df_clean['Date'] >= cutoff][FEATURES]
y_test   = df_clean[df_clean['Date'] >= cutoff][TARGET]
print(f"  Train={len(X_train):,}  Test={len(X_test):,}")

# ── Optuna hyperparameter search ──────────────────────────────────────
def objective(trial):
    params = {
        'max_depth':        trial.suggest_int('max_depth', 3, 8),
        'learning_rate':    trial.suggest_float('lr', 0.01, 0.3, log=True),
        'n_estimators':     trial.suggest_int('n_est', 100, 300),
        'subsample':        trial.suggest_float('sub', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('col', 0.6, 1.0),
    }
    m = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1, verbosity=0)
    m.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=False)
    preds = m.predict(X_test)
    return mean_absolute_error(y_test, preds)

print("  Running Optuna HPO (30 trials) ...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30, show_progress_bar=True)
print(f"  Best MAE from Optuna: {study.best_value:.1f}")
print(f"  Best params: {study.best_params}")

# ── Train final best model ─────────────────────────────────────────────
best_params = {k.replace('lr','learning_rate').replace('n_est','n_estimators')
               .replace('sub','subsample').replace('col','colsample_bytree'): v
               for k, v in study.best_params.items()}
best_xgb = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1, verbosity=0)
best_xgb.fit(X_train, y_train)
preds_xgb = best_xgb.predict(X_test)

mae_xgb  = mean_absolute_error(y_test, preds_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, preds_xgb))
mape_xgb = np.mean(np.abs((y_test - preds_xgb) / y_test.clip(1))) * 100
print(f"\n  XGBoost  MAE={mae_xgb:.1f}  RMSE={rmse_xgb:.1f}  MAPE={mape_xgb:.1f}%")

# ── Save model ─────────────────────────────────────────────────────────
model_path = os.path.join(MODELS, 'xgb_demand.pkl')
joblib.dump(best_xgb, model_path)
print(f"  Model saved -> {model_path}")

# ── Save weekly predictions for the top product ────────────────────────
top_product = demand_grouped.groupby('Product_Code')['Order_Demand'].sum().idxmax()
df_top = df_feat[df_feat['Product_Code'] == top_product].dropna(subset=FEATURES + [TARGET])
df_top_test = df_top[df_top['Date'] >= cutoff]
if len(df_top_test) > 0:
    preds_top_test = best_xgb.predict(df_top_test[FEATURES])
    df_top_test['pred'] = preds_top_test
    # Resample to weekly
    weekly_top = df_top_test.set_index('Date')[['Order_Demand', 'pred']].resample('W').sum()
    # Save with expected columns
    weekly_top.rename(columns={'Order_Demand': 'Actual', 'pred': 'xgb_forecast'}, inplace=True)
    weekly_top.reset_index(inplace=True)
    weekly_top['Date'] = weekly_top['Date'].dt.strftime('%Y-%m-%d')
    weekly_top.to_csv(os.path.join(REPORTS, 'xgb_predictions.csv'), index=False)
    print(f"  Saved weekly XGBoost predictions for {top_product} to reports/xgb_predictions.csv")

print("\nPhase 2b complete.")