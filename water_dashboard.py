"""
💧 Water Quality Index Prediction Dashboard
Models: Random Forest · XGBoost · LSTM
Author: Water Quality Intelligence System
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
import warnings
warnings.filterwarnings("ignore")

# ── Optional models ──────────────────────────────────────────────────────────
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except ImportError:
    HAS_TF = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Water Quality Intelligence",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --deep:    #03224C;
    --navy:    #064789;
    --ocean:   #1565C0;
    --sky:     #1E88E5;
    --azure:   #42A5F5;
    --mist:    #90CAF9;
    --ice:     #BBDEFB;
    --frost:   #E3F2FD;
    --foam:    #F0F8FF;
    --teal:    #00838F;
    --cyan:    #00ACC1;
    --gold:    #FFB300;
    --coral:   #EF5350;
    --white:   #FFFFFF;
    --text:    #0D2137;
    --sub:     #546E7A;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: var(--foam);
    color: var(--text);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--deep) 0%, var(--navy) 60%, #0A3D6B 100%);
    border-right: 3px solid var(--azure);
}
section[data-testid="stSidebar"] * { color: var(--ice) !important; }
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label {
    color: var(--mist) !important;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
section[data-testid="stSidebar"] hr { border-color: rgba(144,202,249,0.2) !important; }
section[data-testid="stSidebar"] .stRadio > div { gap: 0.3rem; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, var(--deep) 0%, var(--navy) 45%, var(--teal) 100%);
    border-radius: 20px;
    padding: 2.8rem 3.2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "💧";
    position: absolute; right: 2.5rem; top: 50%;
    transform: translateY(-50%);
    font-size: 8rem; opacity: 0.07;
}
.hero::after {
    content: "";
    position: absolute; bottom: -40px; left: -40px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(66,165,245,0.15), transparent 70%);
}
.hero-badge {
    display: inline-block;
    background: rgba(66,165,245,0.18);
    border: 1px solid rgba(66,165,245,0.4);
    color: var(--mist);
    font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase;
    padding: 0.28rem 0.85rem; border-radius: 30px;
    margin-right: 0.4rem; margin-bottom: 1rem;
    display: inline-block;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(1.8rem, 4vw, 2.8rem);
    font-weight: 800;
    color: var(--white);
    line-height: 1.1; margin-bottom: 0.7rem;
}
.hero h1 span { color: var(--azure); }
.hero p {
    color: var(--mist); font-size: 0.95rem; font-weight: 300;
    max-width: 620px; margin: 0;
}

/* ── KPI cards ── */
.kpi-card {
    background: var(--white);
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    box-shadow: 0 2px 16px rgba(3,34,76,0.08);
    border-top: 4px solid var(--sky);
    margin-bottom: 0.5rem;
}
.kpi-card.teal  { border-top-color: var(--teal); }
.kpi-card.navy  { border-top-color: var(--navy); }
.kpi-card.gold  { border-top-color: var(--gold); }
.kpi-card.coral { border-top-color: var(--coral); }
.kpi-icon  { font-size: 1.5rem; margin-bottom: 0.3rem; }
.kpi-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.1em; color: #90A4AE; font-weight: 600; }
.kpi-value { font-family: 'Syne', sans-serif; font-size: 1.9rem; color: var(--deep); line-height: 1.1; font-weight: 700; }
.kpi-sub   { font-size: 0.72rem; color: #B0BEC5; margin-top: 0.15rem; }

/* ── Section titles ── */
.sec-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.35rem; font-weight: 700;
    color: var(--navy);
    border-left: 4px solid var(--azure);
    padding-left: 0.75rem;
    margin: 2rem 0 1rem;
}

/* ── Prediction box ── */
.pred-box {
    background: linear-gradient(135deg, var(--ocean), var(--teal));
    border-radius: 18px; padding: 2rem;
    text-align: center;
    box-shadow: 0 8px 30px rgba(21,101,192,0.3);
}
.pred-value {
    font-family: 'Syne', sans-serif;
    font-size: 4rem; font-weight: 800;
    color: var(--ice); line-height: 1;
}
.pred-label { color: rgba(187,222,251,0.8); font-size: 0.9rem; margin-top: 0.3rem; }
.pred-grade {
    display: inline-block;
    margin-top: 0.8rem;
    padding: 0.35rem 1.2rem;
    border-radius: 30px;
    font-size: 0.8rem; font-weight: 700;
    background: rgba(255,255,255,0.15);
    color: var(--white);
    letter-spacing: 0.05em;
}

/* ── Model badge ── */
.model-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px; font-size: 0.72rem; font-weight: 700;
    background: var(--frost); color: var(--ocean);
    border: 1px solid var(--ice);
    margin-right: 0.3rem;
}

/* ── WQI grade legend ── */
.grade-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1rem; }
.grade-chip {
    padding: 0.3rem 0.8rem; border-radius: 20px;
    font-size: 0.72rem; font-weight: 700;
}

/* ── Chart panels ── */
.chart-panel {
    background: var(--white);
    border-radius: 16px; padding: 1.2rem;
    box-shadow: 0 2px 16px rgba(3,34,76,0.07);
    border: 1px solid rgba(144,202,249,0.2);
    margin-bottom: 1rem;
}

/* ── Footer ── */
.footer {
    text-align: center; padding: 2rem 1rem 1rem;
    font-size: 0.75rem; color: #90A4AE;
    border-top: 1px solid var(--ice); margin-top: 3rem;
}
.footer a { color: var(--sky); }
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
FEATURES = [
    "temperature_C", "turbidity_NTU", "pH", "DO_mg_L", "BOD_mg_L",
    "COD_mg_L", "nitrate_mg_L", "phosphate_mg_L", "TDS_mg_L",
    "conductivity_uS_cm", "fecal_coliform_CFU_100mL"
]
FEAT_LABELS = {
    "temperature_C": "Temperature (°C)",
    "turbidity_NTU": "Turbidity (NTU)",
    "pH": "pH",
    "DO_mg_L": "Dissolved Oxygen (mg/L)",
    "BOD_mg_L": "BOD (mg/L)",
    "COD_mg_L": "COD (mg/L)",
    "nitrate_mg_L": "Nitrate (mg/L)",
    "phosphate_mg_L": "Phosphate (mg/L)",
    "TDS_mg_L": "TDS (mg/L)",
    "conductivity_uS_cm": "Conductivity (µS/cm)",
    "fecal_coliform_CFU_100mL": "Fecal Coliform (CFU/100mL)",
}

SKY_SCALE  = ["#E3F2FD", "#90CAF9", "#42A5F5", "#1E88E5", "#1565C0", "#03224C"]
DIVE_SCALE = ["#00ACC1", "#1565C0", "#03224C"]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="#F0F8FF",
    font=dict(family="Inter", color="#0D2137"),
    margin=dict(t=40, b=30, l=20, r=20),
)

def wqi_grade(v):
    if v >= 90: return ("Excellent", "#1565C0")
    if v >= 70: return ("Good",      "#1E88E5")
    if v >= 50: return ("Fair",      "#00ACC1")
    if v >= 25: return ("Poor",      "#FFB300")
    return ("Very Poor", "#EF5350")

def eval_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return mae, rmse, r2

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("water_quality_synthetic.csv", parse_dates=["date"])
    return df

# ── Train models ───────────────────────────────────────────────────────────────
@st.cache_resource
def train_models(n_estimators, xgb_lr, use_lstm):
    df = load_data()
    X = df[FEATURES].values
    y = df["WQI"].values
    split_idx = int(0.8 * len(df))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test,  y_test  = X[split_idx:], y[split_idx:]

    # Random Forest
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_metrics = eval_metrics(y_test, rf_pred)
    feat_imp = pd.Series(rf.feature_importances_, index=FEATURES)

    results = {
        "RF":   {"pred": rf_pred, "metrics": rf_metrics, "label": "Random Forest"},
    }

    # XGBoost
    if HAS_XGB:
        xgb_model = xgb.XGBRegressor(
            n_estimators=400, learning_rate=xgb_lr,
            max_depth=5, random_state=42, verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        results["XGB"] = {"pred": xgb_pred, "metrics": eval_metrics(y_test, xgb_pred), "label": "XGBoost"}

    # LSTM
    lstm_y_test = y_test
    if HAS_TF and use_lstm:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        def make_sequences(Xs, ys, w=12):
            Xa, ya = [], []
            for i in range(len(Xs) - w + 1):
                Xa.append(Xs[i:i+w]); ya.append(ys[i+w-1])
            return np.array(Xa), np.array(ya)

        X_seq, y_seq = make_sequences(X_scaled, y, 12)
        seq_split = int(0.8 * len(X_seq))
        Xst, yst = X_seq[:seq_split], y_seq[:seq_split]
        Xsv, ysv = X_seq[seq_split:], y_seq[seq_split:]

        lstm_model = keras.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=(12, X_seq.shape[2])),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ])
        lstm_model.compile(optimizer="adam", loss="mse")
        lstm_model.fit(Xst, yst, epochs=10, batch_size=32, verbose=0)
        lstm_pred = lstm_model.predict(Xsv, verbose=0).ravel()
        results["LSTM"] = {"pred": lstm_pred, "metrics": eval_metrics(ysv, lstm_pred), "label": "LSTM"}
        lstm_y_test = ysv

    return results, y_test, lstm_y_test, feat_imp, X_train, y_train, FEATURES


df = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💧 Model Settings")
    st.markdown("---")
    n_est   = st.slider("RF Trees",        50, 500, 300, 50)
    xgb_lr  = st.slider("XGBoost LR",      0.01, 0.3, 0.05, 0.01)
    use_lstm = st.checkbox("Enable LSTM (slower)", value=False)

    st.markdown("---")
    st.markdown("### 🔬 WQI Predictor")
    st.caption("Adjust water parameters:")
    inp = {}
    inp["temperature_C"]              = st.slider("Temperature (°C)",       float(df.temperature_C.min()),             float(df.temperature_C.max()),             float(df.temperature_C.mean()),             0.1)
    inp["turbidity_NTU"]              = st.slider("Turbidity (NTU)",         float(df.turbidity_NTU.min()),             float(df.turbidity_NTU.max()),             float(df.turbidity_NTU.mean()),             0.1)
    inp["pH"]                         = st.slider("pH",                      float(df.pH.min()),                        float(df.pH.max()),                        float(df.pH.mean()),                        0.01)
    inp["DO_mg_L"]                    = st.slider("DO (mg/L)",               float(df.DO_mg_L.min()),                   float(df.DO_mg_L.max()),                   float(df.DO_mg_L.mean()),                   0.1)
    inp["BOD_mg_L"]                   = st.slider("BOD (mg/L)",              float(df.BOD_mg_L.min()),                  float(df.BOD_mg_L.max()),                  float(df.BOD_mg_L.mean()),                  0.1)
    inp["COD_mg_L"]                   = st.slider("COD (mg/L)",              float(df.COD_mg_L.min()),                  float(df.COD_mg_L.max()),                  float(df.COD_mg_L.mean()),                  0.1)
    inp["nitrate_mg_L"]               = st.slider("Nitrate (mg/L)",          float(df.nitrate_mg_L.min()),              float(df.nitrate_mg_L.max()),              float(df.nitrate_mg_L.mean()),              0.01)
    inp["phosphate_mg_L"]             = st.slider("Phosphate (mg/L)",        float(df.phosphate_mg_L.min()),            float(df.phosphate_mg_L.max()),            float(df.phosphate_mg_L.mean()),            0.001)
    inp["TDS_mg_L"]                   = st.slider("TDS (mg/L)",              float(df.TDS_mg_L.min()),                  float(df.TDS_mg_L.max()),                  float(df.TDS_mg_L.mean()),                  1.0)
    inp["conductivity_uS_cm"]         = st.slider("Conductivity (µS/cm)",    float(df.conductivity_uS_cm.min()),        float(df.conductivity_uS_cm.max()),        float(df.conductivity_uS_cm.mean()),        1.0)
    inp["fecal_coliform_CFU_100mL"]   = st.slider("Fecal Coliform (CFU)",    float(df.fecal_coliform_CFU_100mL.min()), float(df.fecal_coliform_CFU_100mL.max()), float(df.fecal_coliform_CFU_100mL.mean()), 0.1)
    st.markdown("---")
    st.caption("💧 Water Quality Intelligence · RF · XGBoost · LSTM")

# ── Train ──────────────────────────────────────────────────────────────────────
with st.spinner("🔄 Training models…"):
    results, y_test, lstm_y_test, feat_imp, X_train, y_train, feat_cols = train_models(n_est, xgb_lr, use_lstm)

rf_model = RandomForestRegressor(n_estimators=n_est, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
user_arr   = np.array([[inp[f] for f in FEATURES]])
user_pred  = float(rf_model.predict(user_arr)[0])
grade, grade_color = wqi_grade(user_pred)

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div>
    <span class="hero-badge">💧 Water Quality Intelligence</span>
    <span class="hero-badge">🤖 RF · XGBoost · LSTM</span>
    <span class="hero-badge">📊 900 Samples · 11 Features</span>
  </div>
  <h1>Water Quality Index<br><span>Prediction Dashboard</span></h1>
  <p>Multi-model ML pipeline integrating physicochemical & biological indicators to predict WQI — supporting real-time water safety monitoring and environmental compliance.</p>
</div>
""", unsafe_allow_html=True)

# ── KPI STRIP ─────────────────────────────────────────────────────────────────
rf_mae, rf_rmse, rf_r2 = results["RF"]["metrics"]
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""<div class="kpi-card navy"><div class="kpi-icon">🗃</div>
    <div class="kpi-label">Dataset</div><div class="kpi-value">{len(df)}</div>
    <div class="kpi-sub">monthly samples</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-icon">🎯</div>
    <div class="kpi-label">RF R² Score</div><div class="kpi-value">{rf_r2:.3f}</div>
    <div class="kpi-sub">variance explained</div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="kpi-card teal"><div class="kpi-icon">📉</div>
    <div class="kpi-label">RF RMSE</div><div class="kpi-value">{rf_rmse:.3f}</div>
    <div class="kpi-sub">WQI units</div></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="kpi-card gold"><div class="kpi-icon">💧</div>
    <div class="kpi-label">Mean WQI</div><div class="kpi-value">{df.WQI.mean():.2f}</div>
    <div class="kpi-sub">across all samples</div></div>""", unsafe_allow_html=True)
with c5:
    models_active = sum([1, HAS_XGB, HAS_TF and use_lstm])
    st.markdown(f"""<div class="kpi-card coral"><div class="kpi-icon">🤖</div>
    <div class="kpi-label">Models Active</div><div class="kpi-value">{models_active}</div>
    <div class="kpi-sub">of 3 available</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── LIVE WQI PREDICTOR ────────────────────────────────────────────────────────
st.markdown('<div class="sec-title">🔬 Live WQI Predictor</div>', unsafe_allow_html=True)
p1, p2 = st.columns([1, 1])

with p1:
    st.markdown(f"""
    <div class="pred-box">
      <div class="pred-label">Predicted Water Quality Index</div>
      <div class="pred-value">{user_pred:.2f}</div>
      <div class="pred-label">WQI Score (0 – 100)</div>
      <div class="pred-grade" style="background:{grade_color}88;">{grade} Water Quality</div>
    </div>""", unsafe_allow_html=True)

with p2:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=user_pred,
        delta={"reference": float(df.WQI.mean()), "valueformat": ".2f"},
        title={"text": "vs. Dataset Average", "font": {"size": 12, "family": "Inter"}},
        number={"font": {"size": 26, "family": "Syne", "color": "#03224C"}},
        gauge={
            "axis": {"range": [0, df.WQI.max() + 2], "tickfont": {"size": 9}},
            "bar":  {"color": "#1E88E5"},
            "steps": [
                {"range": [0, 5],               "color": "#FFCDD2"},
                {"range": [5, 12],              "color": "#BBDEFB"},
                {"range": [12, df.WQI.max()+2], "color": "#C8E6C9"},
            ],
            "threshold": {"line": {"color": "#FFB300", "width": 3},
                          "thickness": 0.8, "value": df.WQI.mean()},
        }
    ))
    fig_gauge.update_layout(height=260, margin=dict(t=40,b=10,l=20,r=20), **{k:v for k,v in PLOTLY_LAYOUT.items() if k != 'margin'})
    st.plotly_chart(fig_gauge, use_container_width=True)

# ── WQI GRADE LEGEND ─────────────────────────────────────────────────────────
st.markdown("""
<div class="grade-row">
  <span class="grade-chip" style="background:#FFCDD2;color:#B71C1C;">⚠ 0–25 Very Poor</span>
  <span class="grade-chip" style="background:#FFE0B2;color:#E65100;">🟠 25–50 Poor</span>
  <span class="grade-chip" style="background:#FFF9C4;color:#F57F17;">🟡 50–70 Fair</span>
  <span class="grade-chip" style="background:#BBDEFB;color:#0D47A1;">🔵 70–90 Good</span>
  <span class="grade-chip" style="background:#C8E6C9;color:#1B5E20;">🟢 90–100 Excellent</span>
</div>
""", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Model Performance",
    "🌊 WQI Trends",
    "🔬 Feature Analysis",
    "🗺 Data Explorer",
    "📋 Raw Data",
])

# ── TAB 1: MODEL PERFORMANCE ─────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="sec-title">Model Comparison</div>', unsafe_allow_html=True)

    # Metrics comparison table
    metric_rows = []
    for key, res in results.items():
        mae, rmse, r2 = res["metrics"]
        metric_rows.append({"Model": res["label"], "MAE": round(mae,3), "RMSE": round(rmse,3), "R²": round(r2,3)})
    met_df = pd.DataFrame(metric_rows)

    fig_met = go.Figure()
    colors = ["#1E88E5", "#00ACC1", "#0D47A1"]
    for i, col in enumerate(["MAE","RMSE","R²"]):
        fig_met.add_trace(go.Bar(
            name=col, x=met_df["Model"], y=met_df[col],
            marker_color=colors[i], text=met_df[col].round(3),
            textposition="outside", width=0.2,
        ))
    fig_met.update_layout(
        barmode="group", title="Model Metrics Comparison",
        height=340, **PLOTLY_LAYOUT,
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_met, use_container_width=True)

    # Actual vs Predicted per model
    avp_cols = st.columns(len(results))
    for idx, (key, res) in enumerate(results.items()):
        yt = lstm_y_test if key == "LSTM" else y_test
        pred = res["pred"]
        with avp_cols[idx]:
            residuals = yt - pred
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(yt), y=list(pred), mode="markers",
                marker=dict(color=list(residuals), colorscale="RdBu",
                            size=6, opacity=0.75, showscale=True,
                            colorbar=dict(title="Residual", thickness=10, len=0.6)),
                hovertemplate="Actual: %{x:.2f}<br>Pred: %{y:.2f}<extra></extra>",
            ))
            lim = [min(min(yt), min(pred)), max(max(yt), max(pred))]
            fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                line=dict(color="#FFB300", dash="dash", width=2), showlegend=False))
            fig.update_layout(
                title=f"{res['label']} — Actual vs Predicted",
                xaxis_title="Actual WQI", yaxis_title="Predicted WQI",
                height=320, **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Residual distributions
    st.markdown('<div class="sec-title">Residual Distributions</div>', unsafe_allow_html=True)
    res_cols = st.columns(len(results))
    pal = ["#1E88E5", "#00ACC1", "#0D47A1"]
    for idx, (key, res) in enumerate(results.items()):
        yt = lstm_y_test if key == "LSTM" else y_test
        resid = yt - res["pred"]
        with res_cols[idx]:
            fig_r = go.Figure(go.Histogram(x=resid, nbinsx=25,
                marker_color=pal[idx], opacity=0.85))
            fig_r.add_vline(x=0, line_dash="dash", line_color="#FFB300", line_width=2)
            fig_r.update_layout(title=f"{res['label']} Residuals",
                xaxis_title="Residual (WQI)", yaxis_title="Count",
                height=260, **PLOTLY_LAYOUT)
            st.plotly_chart(fig_r, use_container_width=True)


# ── TAB 2: WQI TRENDS ────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="sec-title">Water Quality Index Over Time</div>', unsafe_allow_html=True)

    # Full WQI time series
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=df["date"], y=df["WQI"],
        mode="lines", name="WQI",
        line=dict(color="#1E88E5", width=1.5),
        fill="tozeroy", fillcolor="rgba(30,136,229,0.1)",
    ))
    # Rolling 12-month average
    roll = df.set_index("date")["WQI"].rolling(12).mean()
    fig_ts.add_trace(go.Scatter(
        x=roll.index, y=roll.values,
        mode="lines", name="12-Month MA",
        line=dict(color="#FFB300", width=2.5),
    ))
    fig_ts.add_hline(y=df.WQI.mean(), line_dash="dot",
        line_color="#EF5350", annotation_text="Mean WQI",
        annotation_position="top right")
    fig_ts.update_layout(
        title="WQI Time Series with 12-Month Moving Average",
        xaxis_title="Date", yaxis_title="WQI",
        height=380, **PLOTLY_LAYOUT,
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    t1, t2 = st.columns(2)
    with t1:
        # Yearly average
        yearly = df.groupby(df["date"].dt.year)["WQI"].mean().reset_index()
        yearly.columns = ["Year", "Avg WQI"]
        fig_yr = px.bar(yearly, x="Year", y="Avg WQI",
            color="Avg WQI", color_continuous_scale=SKY_SCALE,
            title="Yearly Average WQI")
        fig_yr.update_layout(height=320, **PLOTLY_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig_yr, use_container_width=True)

    with t2:
        # Monthly seasonality
        monthly = df.groupby(df["date"].dt.month)["WQI"].mean().reset_index()
        monthly.columns = ["Month", "Avg WQI"]
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly["Month Name"] = monthly["Month"].apply(lambda x: month_names[x-1])
        fig_mo = px.line(monthly, x="Month Name", y="Avg WQI",
            markers=True, title="Monthly WQI Seasonality",
            color_discrete_sequence=["#1E88E5"])
        fig_mo.update_traces(line_width=2.5, marker_size=8)
        fig_mo.update_layout(height=320, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_mo, use_container_width=True)

    # WQI grade distribution over time
    df["Grade"] = df["WQI"].apply(lambda v: wqi_grade(v)[0])
    grade_counts = df["Grade"].value_counts().reset_index()
    grade_counts.columns = ["Grade", "Count"]
    grade_color_map = {"Excellent":"#1565C0","Good":"#42A5F5","Fair":"#00ACC1","Poor":"#FFB300","Very Poor":"#EF5350"}
    fig_pie = px.pie(grade_counts, names="Grade", values="Count",
        color="Grade", color_discrete_map=grade_color_map,
        title="WQI Grade Distribution — All 900 Samples",
        hole=0.45)
    fig_pie.update_layout(height=340, **PLOTLY_LAYOUT)
    st.plotly_chart(fig_pie, use_container_width=True)


# ── TAB 3: FEATURE ANALYSIS ──────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="sec-title">Feature Importance & Correlation</div>', unsafe_allow_html=True)
    fa1, fa2 = st.columns(2)

    with fa1:
        fi_df = feat_imp.reset_index()
        fi_df.columns = ["Feature", "Importance"]
        fi_df["Label"] = fi_df["Feature"].map(FEAT_LABELS)
        fig_fi = px.bar(fi_df.sort_values("Importance"),
            x="Importance", y="Label", orientation="h",
            color="Importance", color_continuous_scale=SKY_SCALE,
            text=fi_df.sort_values("Importance")["Importance"].apply(lambda v: f"{v:.3f}"))
        fig_fi.update_traces(textposition="outside")
        fig_fi.update_layout(title="RF Feature Importance",
            xaxis_title="Importance Score", yaxis_title="",
            coloraxis_showscale=False, height=400, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_fi, use_container_width=True)

    with fa2:
        corr = df[FEATURES + ["WQI"]].corr()
        rename_map = {**FEAT_LABELS, "WQI": "WQI"}
        corr_renamed = corr.rename(columns=rename_map, index=rename_map)
        fig_heat = px.imshow(corr_renamed, text_auto=".2f",
            color_continuous_scale=["#B71C1C","#FAFAFA","#0D47A1"],
            zmin=-1, zmax=1, aspect="auto",
            title="Feature Correlation Matrix")
        fig_heat.update_layout(height=400, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_heat, use_container_width=True)

    # Scatter: any feature vs WQI
    st.markdown('<div class="sec-title">Feature vs WQI Scatter</div>', unsafe_allow_html=True)
    sel = st.selectbox("Select feature", [FEAT_LABELS[f] for f in FEATURES], index=7)
    raw = {v: k for k, v in FEAT_LABELS.items()}[sel]
    fig_sc = px.scatter(df, x=raw, y="WQI",
        color="WQI", color_continuous_scale=SKY_SCALE,
        trendline="ols", opacity=0.6,
        labels={raw: sel, "WQI": "WQI"},
        title=f"{sel} vs WQI")
    fig_sc.update_layout(height=360, **PLOTLY_LAYOUT, coloraxis_showscale=False)
    st.plotly_chart(fig_sc, use_container_width=True)


# ── TAB 4: DATA EXPLORER ─────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="sec-title">Distribution Explorer</div>', unsafe_allow_html=True)

    # Feature distributions
    fig_dist = make_subplots(rows=3, cols=4,
        subplot_titles=[FEAT_LABELS[f] for f in FEATURES] + ["WQI"])
    all_feats = FEATURES + ["WQI"]
    blues = ["#BBDEFB","#90CAF9","#64B5F6","#42A5F5","#1E88E5",
             "#1565C0","#0D47A1","#00ACC1","#006064","#004D40","#00838F","#0097A7"]
    for i, feat in enumerate(all_feats):
        r, c = divmod(i, 4)
        fig_dist.add_trace(
            go.Histogram(x=df[feat], marker_color=blues[i % len(blues)],
                         opacity=0.82, nbinsx=20, showlegend=False),
            row=r+1, col=c+1
        )
    fig_dist.update_layout(height=520, title_text="Parameter Distributions",
        margin=dict(t=60,b=20,l=20,r=20),
        **{k:v for k,v in PLOTLY_LAYOUT.items() if k != 'margin'})
    st.plotly_chart(fig_dist, use_container_width=True)

    e1, e2 = st.columns(2)
    with e1:
        # Box plots normalised
        norm_df = df[FEATURES].copy()
        norm_df = (norm_df - norm_df.min()) / (norm_df.max() - norm_df.min())
        norm_df.columns = [FEAT_LABELS[f] for f in FEATURES]
        fig_box = go.Figure()
        for i, col in enumerate(norm_df.columns):
            fig_box.add_trace(go.Box(y=norm_df[col], name=col,
                marker_color=blues[i % len(blues)], showlegend=False,
                line_color=blues[i % len(blues)]))
        fig_box.update_layout(title="Normalised Parameter Box Plots",
            yaxis_title="Normalised Value", height=360,
            xaxis_tickangle=-35, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_box, use_container_width=True)

    with e2:
        # WQI vs time scatter coloured by grade
        fig_sc2 = px.scatter(df, x="date", y="WQI",
            color="WQI", color_continuous_scale=["#EF5350","#FFB300","#42A5F5","#1565C0"],
            title="WQI Over Time — Coloured by Score", opacity=0.6)
        fig_sc2.update_layout(height=360, **PLOTLY_LAYOUT, coloraxis_showscale=True)
        st.plotly_chart(fig_sc2, use_container_width=True)


# ── TAB 5: RAW DATA ──────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="sec-title">Dataset Preview</div>', unsafe_allow_html=True)
    display_df = df.rename(columns={**FEAT_LABELS, "WQI": "WQI"})
    st.dataframe(
        display_df.style.background_gradient(subset=["WQI"], cmap="Blues").format(precision=3),
        use_container_width=True, height=420
    )
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("**Summary Statistics**")
        st.dataframe(df[FEATURES + ["WQI"]].describe().round(3), use_container_width=True)
    with d2:
        st.markdown("**Missing Values**")
        miss = df.isnull().sum().reset_index()
        miss.columns = ["Column", "Missing"]
        st.dataframe(miss, use_container_width=True)
    with d3:
        st.markdown("**Download**")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download CSV", csv, "water_quality_data.csv", "text/csv")
        xl = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Excel-ready", xl, "water_quality_data.csv", "text/csv")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  💧 Water Quality Intelligence Dashboard &nbsp;·&nbsp;
  Random Forest · XGBoost · LSTM &nbsp;·&nbsp;
  Built with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)
