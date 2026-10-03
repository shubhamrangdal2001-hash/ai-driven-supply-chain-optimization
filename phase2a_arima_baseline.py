"""
Phase 2a – ARIMA / SARIMA Baseline Forecast
Reads demand_grouped from Phase 1 parquet output.
Fits SARIMA on the top-selling product's weekly time-series.
"""

import os, warnings
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings('ignore')

ROOT    = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, 'reports')

print("=" * 60)
print("PHASE 2a - SARIMA BASELINE")
print("=" * 60)

# ── Load demand data ──────────────────────────────────────────────────
demand_grouped = pd.read_parquet(os.path.join(REPORTS, 'demand_grouped.parquet'))
print(f"  Loaded demand_grouped: {demand_grouped.shape}")

# ── Use single top-selling product for ARIMA ──────────────────────────
top_product = (
    demand_grouped.groupby('Product_Code')['Order_Demand'].sum().idxmax()
)
print(f"  Top product: {top_product}")

ts = (
    demand_grouped[demand_grouped['Product_Code'] == top_product]
    .set_index('Date')['Order_Demand']
    .resample('W').sum()
    .dropna()
)
print(f"  Time-series length: {len(ts)} weeks")

# ── Test stationarity ─────────────────────────────────────────────────
adf_result = adfuller(ts)
print(f"  ADF p-value: {adf_result[1]:.4f}")
if adf_result[1] > 0.05:
    ts_diff = ts.diff().dropna()
    print("  -> Applied first-order differencing")
else:
    ts_diff = ts
    print("  -> Series is already stationary")

# ── Train / Test split (80 / 20) ──────────────────────────────────────
split     = int(len(ts) * 0.8)
train, test = ts[:split], ts[split:]
print(f"  Train={len(train)} weeks  Test={len(test)} weeks")

# ── Fit SARIMA(1,1,1)(1,1,1,52) ──────────────────────────────────────
print("  Fitting SARIMA ... (may take ~1-2 min)")
model     = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52),
                    enforce_stationarity=False)
sarima_fit = model.fit(disp=False)

# ── Forecast ──────────────────────────────────────────────────────────
forecast   = sarima_fit.forecast(steps=len(test))
mae_arima  = np.mean(np.abs(test.values - forecast.values))
mape_arima = np.mean(np.abs((test.values - forecast.values) / test.values.clip(1))) * 100
rmse_arima = np.sqrt(np.mean((test.values - forecast.values) ** 2))

print(f"\n  SARIMA  MAE={mae_arima:.1f}  RMSE={rmse_arima:.1f}  MAPE={mape_arima:.1f}%")

# Save predictions for the test period
forecast_df = pd.DataFrame({
    'Date': test.index.strftime('%Y-%m-%d'),
    'Actual': test.values,
    'sarima_forecast': forecast.values
})
forecast_df.to_csv(os.path.join(REPORTS, 'sarima_predictions.csv'), index=False)
print(f"  Saved SARIMA predictions to reports/sarima_predictions.csv")

print("\nPhase 2a complete.")