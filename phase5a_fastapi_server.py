"""
Phase 5a - FastAPI Inference Server
Loads XGBoost + LSTM models and exposes REST endpoints.
Run:  python phase5a_fastapi_server.py
"""

import os
import asyncio
import joblib
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

ROOT    = os.path.dirname(os.path.abspath(__file__))
MODELS  = os.path.join(ROOT, 'src', 'models')
REPORTS = os.path.join(ROOT, 'reports')

# ── LSTM definition (must match Phase 2c) ─────────────────────────────
class DemandLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden=128, layers=2, dropout=0.2, pred_len=7):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, pred_len)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = self.norm(out[:, -1, :])
        return self.head(out)

# ── Global model references ───────────────────────────────────────────
xgb_model  = None
lstm_model = None
scaler     = None
sku_plan   = None

# ── Lifespan (replaces deprecated @app.on_event("startup")) ──────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global xgb_model, lstm_model, scaler, sku_plan
    xgb_path    = os.path.join(MODELS, 'xgb_demand.pkl')
    lstm_path   = os.path.join(MODELS, 'lstm_demand.pt')
    scaler_path = os.path.join(MODELS, 'lstm_scaler.pkl')
    plan_path   = os.path.join(REPORTS, 'inventory_plan.csv')

    print("\n[Startup] Loading models ...")

    if os.path.exists(xgb_path):
        xgb_model = joblib.load(xgb_path)
        print("  [OK] XGBoost model loaded.")
    else:
        print(f"  [WARN] XGBoost model not found at {xgb_path}")

    if os.path.exists(lstm_path) and os.path.exists(scaler_path):
        scaler     = joblib.load(scaler_path)
        lstm_model = DemandLSTM(pred_len=7)
        # weights_only=True avoids deprecation warning in PyTorch >= 2.0
        lstm_model.load_state_dict(
            torch.load(lstm_path, map_location='cpu', weights_only=True)
        )
        lstm_model.eval()
        print("  [OK] LSTM model loaded.")
    else:
        print("  [WARN] LSTM model or scaler not found. Run Phase 2c first.")

    if os.path.exists(plan_path):
        sku_plan = pd.read_csv(plan_path)
        print(f"  [OK] Inventory plan loaded ({len(sku_plan):,} SKUs).")
    else:
        print(f"  [WARN] inventory_plan.csv not found. Run Phase 4 first.")

    print("[Startup] Ready.\n")
    yield  # server runs here
    print("[Shutdown] Cleaning up.")

app = FastAPI(title="Supply Chain AI API", version="1.0", lifespan=lifespan)

# ── Schemas ───────────────────────────────────────────────────────────
class DemandRequest(BaseModel):
    product_code:  str
    warehouse:     str
    horizon_days:  int = 7
    recent_demand: List[float]   # last 30 daily values

# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Supply Chain AI API running", "docs": "/docs"}

@app.get("/health")
def health():
    return {
        "status":      "ok",
        "xgb_loaded":  xgb_model  is not None,
        "lstm_loaded": lstm_model is not None,
        "plan_loaded": sku_plan   is not None,
    }

@app.post("/predict/demand")
def predict_demand(req: DemandRequest):
    if lstm_model is None or scaler is None:
        raise HTTPException(status_code=503,
                            detail="LSTM model not loaded. Run Phase 2c first.")
    seq        = np.array(req.recent_demand[-30:], dtype=np.float32)
    seq_scaled = scaler.transform(seq.reshape(-1, 1)).flatten()
    tensor     = torch.FloatTensor(seq_scaled).unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        pred = lstm_model(tensor).numpy().flatten()
    forecast = scaler.inverse_transform(pred.reshape(-1, 1)).flatten().tolist()
    return {
        "product_code":   req.product_code,
        "warehouse":      req.warehouse,
        "forecast_7days": forecast,
        "total_forecast": round(sum(forecast), 1)
    }

@app.get("/inventory/{product_code}")
def get_inventory_plan(product_code: str):
    if sku_plan is None:
        raise HTTPException(status_code=503,
                            detail="Inventory plan not loaded. Run Phase 4 first.")
    row = sku_plan[sku_plan['Product_Code'] == product_code]
    if row.empty:
        raise HTTPException(status_code=404,
                            detail=f"Product '{product_code}' not found.")
    return row.iloc[0][['Product_Code', 'EOQ', 'safety_stock',
                         'reorder_point', 'ABC']].to_dict()

@app.get("/inventory")
def list_inventory(abc_class: str = None, limit: int = 50):
    """List all SKUs, optionally filtered by ABC class."""
    if sku_plan is None:
        raise HTTPException(status_code=503,
                            detail="Inventory plan not loaded. Run Phase 4 first.")
    df = sku_plan.copy()
    if abc_class:
        df = df[df['ABC'] == abc_class.upper()]
    cols = ['Product_Code', 'EOQ', 'safety_stock', 'reorder_point', 'ABC']
    return df[cols].head(limit).to_dict(orient='records')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002, reload=False)