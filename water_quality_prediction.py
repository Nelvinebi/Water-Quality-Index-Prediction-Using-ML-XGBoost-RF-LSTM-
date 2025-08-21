# water_quality_prediction.py 
"""
Water Quality Index Prediction using Random Forest, XGBoost, and LSTM
Synthetic dataset with >100 points
"""

import numpy as np
import pandas as pd
import os, math, warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Try optional libraries
try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except:
    HAS_TF = False


# =============== Synthetic Dataset ===================
def seasonal(n, period=12, amplitude=1.0, phase=0.0):
    t = np.arange(n)
    return amplitude * np.sin(2*np.pi*(t/period) + phase)

def trend(n, slope=0.0, start=0.0):
    return start + slope*np.arange(n)

def generate_data(n_months=900):
    np.random.seed(42)
    dates = pd.date_range("2010-01-01", periods=n_months, freq="MS")

    temp = 20 + seasonal(n_months, 12, 5, 0.5) + np.random.normal(0, 1.2, n_months)
    turbidity = np.clip(5 + 3*seasonal(n_months, 6) + np.random.normal(0, 1.4, n_months) + 0.02*trend(n_months, 0.05), 0, None)
    ph = np.clip(7.2 + 0.2*seasonal(n_months, 12, 1.0, 1.0) + np.random.normal(0, 0.08, n_months), 6.2, 8.8)
    do = np.clip(8.5 - 0.25*(temp-20) + np.random.normal(0, 0.4, n_months), 2.5, 14.0)
    bod = np.clip(2.5 + 0.5*seasonal(n_months, 12, 1.0, 2.0) + np.random.normal(0, 0.4, n_months) + 0.02*turbidity, 0.5, None)
    cod = np.clip(10 + 2.2*seasonal(n_months, 12, 1.0, 2.5) + np.random.normal(0, 1.6, n_months) + 1.5*bod, 2, None)
    nitrate = np.clip(2.0 + 0.8*seasonal(n_months, 12, 1.0, 0.2) + np.random.normal(0, 0.3, n_months) + 0.03*turbidity, 0, None)
    phosphate = np.clip(0.4 + 0.15*seasonal(n_months, 12, 1.0, 0.8) + np.random.normal(0, 0.05, n_months) + 0.01*turbidity, 0, None)
    tds = np.clip(150 + 40*seasonal(n_months, 12, 1.0, 1.4) + np.random.normal(0, 20, n_months) + 3.0*temp, 50, None)
    conductivity = np.clip(300 + 1.5*tds + np.random.normal(0, 30, n_months), 100, None)
    fecal_coliform = np.clip(30 + 8*seasonal(n_months, 12, 1.0, 2.1) + np.random.normal(0, 6, n_months) + 1.5*turbidity, 0, None)

    df = pd.DataFrame({
        "date": dates,
        "temperature_C": temp,
        "turbidity_NTU": turbidity,
        "pH": ph,
        "DO_mg_L": do,
        "BOD_mg_L": bod,
        "COD_mg_L": cod,
        "nitrate_mg_L": nitrate,
        "phosphate_mg_L": phosphate,
        "TDS_mg_L": tds,
        "conductivity_uS_cm": conductivity,
        "fecal_coliform_CFU_100mL": fecal_coliform
    })

    wqi = (
        12 +
        8*np.clip(1 - np.abs(df["pH"]-7.0)/1.5, 0, 1) +
        15*np.clip(df["DO_mg_L"]/12, 0, 1) -
        8*np.tanh(df["turbidity_NTU"]/15) -
        10*np.tanh(df["BOD_mg_L"]/6) -
        10*np.tanh(df["COD_mg_L"]/40) -
        6*np.tanh(df["nitrate_mg_L"]/6) -
        6*np.tanh(df["phosphate_mg_L"]/0.8) -
        8*np.tanh(df["fecal_coliform_CFU_100mL"]/150) -
        5*np.tanh(df["TDS_mg_L"]/600)
    )
    wqi = np.clip(wqi + np.random.normal(0, 2.0, n_months), 0, 100)
    df["WQI"] = wqi
    return df

# Generate and save
df = generate_data()
os.makedirs("artifacts", exist_ok=True)
df.to_csv("artifacts/water_quality_synthetic.csv", index=False)
df.to_excel("artifacts/water_quality_synthetic.xlsx", index=False)

# =============== Train/Test Split ===================
features = [c for c in df.columns if c not in ["date","WQI"]]
X = df[features].values
y = df["WQI"].values
split_idx = int(0.8*len(df))
X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]

# =============== Random Forest ===================
rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")

evaluate("Random Forest", y_test, rf_pred)

# =============== XGBoost ===================
if HAS_XGB:
    xgb_model = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    evaluate("XGBoost", y_test, xgb_pred)

# =============== LSTM ===================
if HAS_TF:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    def make_sequences(X, y, window=12):
        Xs, ys = [], []
        for i in range(len(X)-window+1):
            Xs.append(X[i:i+window])
            ys.append(y[i+window-1])
        return np.array(Xs), np.array(ys)

    X_seq, y_seq = make_sequences(X_scaled, y, window=12)
    seq_split = int(0.8*len(X_seq))
    X_seq_train, y_seq_train = X_seq[:seq_split], y_seq[:seq_split]
    X_seq_test, y_seq_test = X_seq[seq_split:], y_seq[seq_split:]

    model = keras.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=(12, X_seq.shape[2])),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dense(16, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_seq_train, y_seq_train, epochs=10, batch_size=32, verbose=0)
    lstm_pred = model.predict(X_seq_test).ravel()
    evaluate("LSTM", y_seq_test, lstm_pred)

print("\nDataset saved to 'artifacts/water_quality_synthetic.csv' and '.xlsx'")