"""
Phase 2c – LSTM Demand Forecast
Reads demand_grouped from Phase 1 parquet output.
Trains a 2-layer LSTM on the top-selling product's time-series.
Saves model weights to src/models/lstm_demand.pt
Saves the MinMaxScaler to src/models/lstm_scaler.pkl
"""

import os, warnings
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

ROOT    = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, 'reports')
MODELS  = os.path.join(ROOT, 'src', 'models')
os.makedirs(MODELS, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────
SEQ_LEN  = 30
PRED_LEN = 7
HIDDEN   = 128
LAYERS   = 2
DROPOUT  = 0.2
EPOCHS   = 50
BATCH    = 64
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 60)
print("PHASE 2c - LSTM FORECAST")
print("=" * 60)
print(f"  Device: {DEVICE}")

# ── Load data ─────────────────────────────────────────────────────────
demand_grouped = pd.read_parquet(os.path.join(REPORTS, 'demand_grouped.parquet'))
top_product    = (
    demand_grouped.groupby('Product_Code')['Order_Demand'].sum().idxmax()
)
ts = (
    demand_grouped[demand_grouped['Product_Code'] == top_product]
    .set_index('Date')['Order_Demand']
    .resample('W').sum()
    .dropna()
)
print(f"  Top product: {top_product}  |  Series length: {len(ts)} weeks")

# ── Scale ─────────────────────────────────────────────────────────────
scaler = MinMaxScaler()
values = scaler.fit_transform(ts.values.reshape(-1, 1)).flatten()

def make_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i: i + seq_len])
        y.append(data[i + seq_len: i + seq_len + pred_len])
    return np.array(X), np.array(y)

X_all, y_all = make_sequences(values, SEQ_LEN, PRED_LEN)
split  = int(len(X_all) * 0.8)

X_tr = torch.FloatTensor(X_all[:split]).unsqueeze(-1).to(DEVICE)
y_tr = torch.FloatTensor(y_all[:split]).to(DEVICE)
X_te = torch.FloatTensor(X_all[split:]).unsqueeze(-1).to(DEVICE)
y_te = torch.FloatTensor(y_all[split:]).to(DEVICE)
print(f"  Train sequences={X_tr.shape[0]}  Test sequences={X_te.shape[0]}")

loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH, shuffle=True)

# ── LSTM Model ────────────────────────────────────────────────────────
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

model_lstm = DemandLSTM(pred_len=PRED_LEN).to(DEVICE)
optimizer  = torch.optim.Adam(model_lstm.parameters(), lr=1e-3)
scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
criterion  = nn.HuberLoss()

# ── Training loop ─────────────────────────────────────────────────────
print(f"  Training LSTM for {EPOCHS} epochs ...")
for epoch in range(1, EPOCHS + 1):
    model_lstm.train()
    epoch_loss = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        pred = model_lstm(xb)
        loss = criterion(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model_lstm.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step(epoch_loss)
    if epoch % 10 == 0:
        print(f"    Epoch {epoch:3d}/{EPOCHS}  Loss={epoch_loss/len(loader):.4f}")

# ── Evaluate ──────────────────────────────────────────────────────────
model_lstm.eval()
with torch.no_grad():
    preds_lstm_scaled = model_lstm(X_te).cpu().numpy()

preds_lstm  = scaler.inverse_transform(preds_lstm_scaled.reshape(-1, 1)).flatten()
actual_lstm = scaler.inverse_transform(y_te.cpu().numpy().reshape(-1, 1)).flatten()

mae_lstm  = mean_absolute_error(actual_lstm, preds_lstm)
rmse_lstm = np.sqrt(mean_squared_error(actual_lstm, preds_lstm))
mape_lstm = np.mean(np.abs((actual_lstm - preds_lstm) / actual_lstm.clip(1))) * 100
print(f"\n  LSTM  MAE={mae_lstm:.1f}  RMSE={rmse_lstm:.1f}  MAPE={mape_lstm:.1f}%")

# ── Save ──────────────────────────────────────────────────────────────
pt_path     = os.path.join(MODELS, 'lstm_demand.pt')
scaler_path = os.path.join(MODELS, 'lstm_scaler.pkl')
torch.save(model_lstm.state_dict(), pt_path)
joblib.dump(scaler, scaler_path)
print(f"  Model  saved -> {pt_path}")
print(f"  Scaler saved -> {scaler_path}")

# ── Save weekly predictions for the top product ────────────────────────
test_dates = ts.index[split + SEQ_LEN : split + SEQ_LEN + len(X_te)]
test_actuals = ts.values[split + SEQ_LEN : split + SEQ_LEN + len(X_te)]
preds_lstm_weekly = scaler.inverse_transform(preds_lstm_scaled[:, 0].reshape(-1, 1)).flatten()

lstm_forecast_df = pd.DataFrame({
    'Date': test_dates.strftime('%Y-%m-%d'),
    'Actual': test_actuals,
    'lstm_forecast': preds_lstm_weekly
})
lstm_forecast_df.to_csv(os.path.join(REPORTS, 'lstm_predictions.csv'), index=False)
print(f"  Saved weekly LSTM predictions for {top_product} to reports/lstm_predictions.csv")

print("\nPhase 2c complete.")