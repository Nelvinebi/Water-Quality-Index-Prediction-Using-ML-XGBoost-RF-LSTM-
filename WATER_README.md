# 💧 Water Quality Index Prediction Using ML · XGBoost · RF · LSTM

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-064789?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RF-1E88E5?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boost-00ACC1?style=for-the-badge&logo=xgboost&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-0D47A1?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-1565C0?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-42A5F5?style=for-the-badge)

**A multi-model ML & Deep Learning pipeline integrating physicochemical and biological water parameters to predict the Water Quality Index — supporting real-time environmental monitoring and water safety compliance.**

[🚀 Run Dashboard](#️-how-to-run) · [📊 Results](#-model-results--performance) · [📐 Methodology](#-methodology) · [👤 Author](#-author)

</div>

---

## 📌 Project Overview

This project demonstrates how **Machine Learning** (Random Forest & XGBoost) and **Deep Learning** (LSTM) can be used to predict the **Water Quality Index (WQI)** from environmental physicochemical and biological parameters.

A synthetic dataset of **900 monthly records** (2010 onward) was generated to simulate real-world water quality monitoring, incorporating seasonal patterns, environmental trends, and realistic noise. The framework is designed to scale to real IoT sensor data and field deployments.

### 🌊 Key Objectives

- Build an integrated water quality dataset with 11 environmental features
- Train and compare three model architectures: **RF, XGBoost, LSTM**
- Evaluate performance using industry-standard regression metrics
- Interpret environmental drivers of WQI through feature importance analysis
- Deploy an interactive prediction dashboard via Streamlit

### 🌍 Applications

| Domain | Use Case |
|---|---|
| 🏭 Environmental Compliance | Automated WQI monitoring against regulatory thresholds |
| 🚰 Water Safety | Early warning systems for drinking water treatment plants |
| 🌿 Ecological Monitoring | Aquatic ecosystem health assessment |
| 📡 IoT Integration | Real-time WQI prediction from field sensor streams |
| 🏛 Policy & Planning | Data-driven water resource management decisions |

---

## 📊 Key Results at a Glance

<div align="center">

| Metric | Random Forest | XGBoost | LSTM |
|:---:|:---:|:---:|:---:|
| 🎯 **R² Score** | **0.712** | — | — |
| 📉 **RMSE** | **2.331** | — | — |
| 📐 **MAE** | **1.884** | — | — |

> XGBoost and LSTM metrics vary per run. RF is the primary benchmarked model.

| Statistic | Value |
|:---:|:---:|
| 🗃 **Dataset Size** | 900 monthly samples |
| 💧 **Mean WQI** | 10.59 |
| 📈 **WQI Range** | 0.0 – 20.84 |
| 🔬 **Features** | 11 physicochemical & biological |

</div>

---

## 💧 WQI Grade Classification

```
Score Range   Grade         Water Safety Status
──────────────────────────────────────────────────
  90 – 100  │ 🟢 Excellent │ Safe — No treatment needed
  70 –  90  │ 🔵 Good      │ Acceptable — Minor treatment
  50 –  70  │ 🟡 Fair      │ Moderate concern — Treatment advised
  25 –  50  │ 🟠 Poor      │ High concern — Treatment required
   0 –  25  │ 🔴 Very Poor │ Unsafe — Not suitable for use
```

---

## 🗂 Project Structure

```
Water-Quality-Index-Prediction-Using-ML-XGBoost-RF-LSTM/
│
├── 📊 water_quality_synthetic.csv     # Synthetic dataset (900 samples)
├── 📊 water_quality_synthetic.xlsx    # Excel format dataset
├── 🐍 water_quality_prediction.py     # Core ML training & evaluation script
├── 🖥  dashboard.py                   # Streamlit interactive dashboard
├── 📋 requirements.txt                # Project dependencies
├── 📁 artifacts/                      # Generated model outputs
│   ├── water_quality_synthetic.csv
│   └── water_quality_synthetic.xlsx
└── 📖 README.md                       # Documentation
```

---

## 📋 Dataset & Features

**Type:** Synthetic · **Samples:** 900 · **Frequency:** Monthly · **Period:** 2010 onward

| Feature | Unit | Category | Correlation with WQI | Role |
|---|:---:|:---:|:---:|:---:|
| 🌡 **Temperature** | °C | Physical | −0.603 | Input |
| 🌊 **Turbidity** | NTU | Physical | −0.397 | Input |
| ⚗ **pH** | — | Chemical | −0.683 | Input |
| 💨 **Dissolved Oxygen** | mg/L | Chemical | +0.606 | Input |
| 🧪 **BOD** | mg/L | Chemical | −0.453 | Input |
| 🧫 **COD** | mg/L | Chemical | −0.259 | Input |
| 🌿 **Nitrate** | mg/L | Chemical | −0.517 | Input |
| 🔬 **Phosphate** | mg/L | Chemical | **−0.740** | Input |
| 💎 **TDS** | mg/L | Physical | −0.677 | Input |
| ⚡ **Conductivity** | µS/cm | Physical | −0.624 | Input |
| 🦠 **Fecal Coliform** | CFU/100mL | Biological | −0.452 | Input |
| **💧 WQI** | 0–100 | — | Target | **Output** |

> ⚠️ All correlations are negative (contamination increases → WQI decreases) except **Dissolved Oxygen** (+0.606) — higher DO indicates healthier water.

---

## 📐 Methodology

```
┌──────────────────────────────────────────────────────────────────────┐
│                     ML PIPELINE WORKFLOW                             │
├──────────┬──────────┬──────────────────────────┬────────────────────┤
│  STEP 1  │  STEP 2  │         STEP 3           │       STEP 4       │
│    📥    │    🔀    │           🤖             │         📊         │
│  Data    │  Split   │      Model Training      │    Evaluation      │
│  Synth   │  80/20   │                          │  & Interpretation  │
├──────────┼──────────┼──────────────────────────┼────────────────────┤
│ Generate │ Temporal │ • Random Forest (300 est)│ MAE, RMSE, R²      │
│ 900-row  │ split —  │ • XGBoost (400 est,      │                    │
│ synthetic│ no data  │   lr=0.05, depth=5)      │ Feature importance │
│ dataset  │ leakage  │ • LSTM (64→32, w=12mo)  │ Actual vs Pred     │
│          │          │   with Dropout(0.2)      │ plots              │
└──────────┴──────────┴──────────────────────────┴────────────────────┘
```

### Data Generation — Synthetic Design

The dataset simulates realistic seasonal and environmental dynamics:

```python
# Seasonal patterns with noise — example: Temperature
temp = 20 + seasonal(n, period=12, amplitude=5) + noise(σ=1.2)

# WQI formula — multi-parameter composite index
WQI = 12
    + 8  · clip(1 - |pH - 7.0| / 1.5)    # pH penalty
    + 15 · clip(DO / 12)                   # DO reward
    - 8  · tanh(turbidity / 15)            # turbidity penalty
    - 10 · tanh(BOD / 6)                   # BOD penalty
    - 6  · tanh(phosphate / 0.8)           # phosphate penalty
    + noise(σ=2.0)
```

---

## 📊 Model Results & Performance

### 1️⃣ Random Forest — Feature Importance

```
Phosphate (mg/L)          ████████████████████████████████████████  48.8%
Turbidity (NTU)           ████████████                               11.6%
pH                        █████████                                   7.9%
BOD (mg/L)                ████████                                    7.3%
TDS (mg/L)                ████████                                    7.2%
COD (mg/L)                █████                                       4.3%
Fecal Coliform            ████                                        3.1%
Dissolved Oxygen          ████                                        3.2%
Nitrate (mg/L)            ███                                         2.5%
Temperature (°C)          ██                                          2.1%
Conductivity (µS/cm)      ██                                          2.1%
```

> **Key insight:** Phosphate alone accounts for nearly **49% of model importance** — making it the single most critical predictor of WQI in this dataset. This aligns with its strong negative correlation (r = −0.740).

---

### 2️⃣ Feature Correlation with WQI

```
Direction    Feature               Correlation   Interpretation
──────────────────────────────────────────────────────────────────────
POSITIVE ↑   Dissolved Oxygen      +0.606        Higher DO → better quality
──────────────────────────────────────────────────────────────────────
NEGATIVE ↓   Phosphate             −0.740        Strongest contaminant driver
             pH                    −0.683        Acidity/alkalinity impact
             TDS                   −0.677        Dissolved solids load
             Conductivity          −0.624        Ionic contamination proxy
             Temperature           −0.603        Heat stress reduces WQI
             Nitrate               −0.517        Eutrophication indicator
             BOD                   −0.453        Organic waste indicator
             Fecal Coliform        −0.452        Biological contamination
             Turbidity             −0.397        Suspended solids
             COD                   −0.259        Chemical oxygen demand
```

---

### 3️⃣ Actual vs Predicted — Random Forest (Test Set, 40 samples)

```
WQI  20 |         ×
(pred)18 |              ×
     16 |       ×           ×    ×
     14 |  ×  ·   ×  ×   ×    ×   ×  ×
     12 |    ×  ·    ·  ×  ·    ×  ·  ×
     10 |  ·    ×   ·     ·   ×      ·
      8 |    ·    ×    ·    ×    ·
      6 |  ×   ·    ·          ·    ×
      4 |    ×    ×    ×  ×         ×
      2 |        ·               ·    ×
      0 |_________________________________________
          0    5   10   15   20   25   30   35   40
                       Sample Index
               · Actual   × Predicted  (RF R² = 0.712)
```

---

### 4️⃣ WQI Distribution — 900 Samples

```
Count
 100+|
  90 |          ████
  80 |     ████ ████ ████
  70 |     ████ ████ ████
  60 |████ ████ ████ ████ ████
  50 |████ ████ ████ ████ ████
  40 |████ ████ ████ ████ ████ ████
  30 |████ ████ ████ ████ ████ ████ ████
  10 |████ ████ ████ ████ ████ ████ ████ ████
   0 |────────────────────────────────────────────
      0    2    4    6    8   10   12   14   16   20
                        WQI Score
   Mean: 10.59 │ Std: 4.31 │ Min: 0.0 │ Max: 20.84
```

---

### 5️⃣ Seasonal WQI Pattern

```
WQI   12 |      ●
Avg   11 | ●          ●         ●    ●
      10 |    ●    ●    ●    ●    ●    ●
       9 |                          ●
       8 |
       7 |
       ──────────────────────────────────────────
         Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec
         
         Lowest WQI in summer months (Jul–Sep) — thermal & biological stress
         Highest WQI in winter months (Jan, Nov) — cooler, lower runoff
```

---

## 🤖 Models & Architecture

### 🌲 Random Forest Regressor
```
Config: n_estimators=300, random_state=42, n_jobs=-1
Split:  Temporal 80/20 (no shuffling — preserves time order)
MAE:    1.884  |  RMSE: 2.331  |  R²: 0.712
```
**Why RF?** Captures nonlinear interactions between water parameters, robust to outliers, provides native feature importance — ideal for environmental tabular data.

### ⚡ XGBoost Regressor
```
Config: n_estimators=400, learning_rate=0.05, max_depth=5
Split:  Temporal 80/20
Status: Optional (pip install xgboost)
```
**Why XGBoost?** Gradient boosting with L1/L2 regularization handles correlated features (e.g. TDS ↔ conductivity) more gracefully than standard trees.

### 🧠 LSTM (Long Short-Term Memory)
```
Architecture: LSTM(64, return_seq=True) → Dropout(0.2)
              → LSTM(32) → Dense(16, ReLU) → Dense(1)
Window:       12 months (1 year lookback)
Optimizer:    Adam  |  Loss: MSE  |  Epochs: 10
Status:       Optional (pip install tensorflow)
```
**Why LSTM?** Water quality exhibits strong seasonal autocorrelation — a 12-month sliding window captures annual cycles that tabular models miss.

---

## ▶️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/Nelvinebi/Water-Quality-Index-Prediction-Using-ML-XGBoost-RF-LSTM.git
cd Water-Quality-Index-Prediction-Using-ML-XGBoost-RF-LSTM
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3a. Run the ML Script
```bash
python water_quality_prediction.py
```

### 3b. Launch the Streamlit Dashboard
```bash
streamlit run dashboard.py
```

> ⚠️ Ensure `water_quality_synthetic.csv` is in the same directory as `dashboard.py`.

---

## 🛠 Tools & Technologies

| Tool | Purpose |
|---|---|
| ![Python](https://img.shields.io/badge/-Python-064789?logo=python&logoColor=white) | Core language |
| ![scikit-learn](https://img.shields.io/badge/-scikit--learn-1E88E5?logo=scikit-learn&logoColor=white) | Random Forest, preprocessing, metrics |
| ![XGBoost](https://img.shields.io/badge/-XGBoost-00ACC1?logoColor=white) | Gradient boosting regressor |
| ![TensorFlow](https://img.shields.io/badge/-TensorFlow-0D47A1?logo=tensorflow&logoColor=white) | LSTM deep learning model |
| ![Pandas](https://img.shields.io/badge/-Pandas-1565C0?logo=pandas&logoColor=white) | Data wrangling |
| ![NumPy](https://img.shields.io/badge/-NumPy-42A5F5?logo=numpy&logoColor=white) | Numerical computing |
| ![Plotly](https://img.shields.io/badge/-Plotly-064789?logo=plotly&logoColor=white) | Interactive visualizations |
| ![Streamlit](https://img.shields.io/badge/-Streamlit-1E88E5?logo=streamlit&logoColor=white) | Web dashboard |
| ![Matplotlib](https://img.shields.io/badge/-Matplotlib-00ACC1?logoColor=white) | Static plotting |

---

## 💡 Key Insights

1. **Phosphate is the dominant driver** — 48.8% feature importance and r = −0.740; reducing phosphate inputs has the highest leverage on WQI improvement
2. **Dissolved Oxygen is the only positive predictor** — r = +0.606; aeration strategies directly improve WQI
3. **pH and TDS are strongly negative** — chemical balance and dissolved solids management are critical treatment targets
4. **Seasonal degradation in summer** — WQI consistently dips July–September due to thermal stress and increased biological activity
5. **LSTM adds temporal context** — the 12-month lookback captures annual WQI cycles invisible to static tabular models
6. **Ensemble learning is robust** — RF's 0.712 R² on 900 synthetic samples demonstrates strong generalisation across the WQI range

---

## ⚠️ Limitations

- Dataset is **synthetic** — not derived from real sensor or field measurements
- LSTM requires **TensorFlow** installation (optional)
- No **spatial component** — geographic variation in water quality not modelled
- Single waterbody type — results may not generalise across river, lake, and groundwater systems
- Minimal **hyperparameter tuning** on LSTM architecture

---

## 🔮 Future Work

- [ ] Integrate **real water quality datasets** (WHO, USGS, EPA open data)
- [ ] Deploy as a **REST API** for real-time WQI prediction from IoT sensors
- [ ] Add **spatial mapping** with GeoPandas and Folium for multi-site monitoring
- [ ] Implement **Temporal Fusion Transformers** for improved time-series forecasting
- [ ] Build **alert system** — automated notifications when WQI drops below threshold
- [ ] **Hyperparameter optimisation** via Optuna or Ray Tune

---

## 👤 Author

<div align="center">

**AGBOZU EBINGIYE NELVIN**

*Environmental Data Scientist · GIS · Remote Sensing · Machine Learning*

End-to-end environmental intelligence solutions for flood risk, water quality, vegetation monitoring, and climate-resilient planning.

[![GitHub](https://img.shields.io/badge/GitHub-Nelvinebi-064789?style=for-the-badge&logo=github)](https://github.com/Nelvinebi)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-agbozu--ebi-1E88E5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/agbozu-ebi/)

</div>

---

<div align="center">

💧 *"Clean water is not a luxury — it is a right. Data science helps us protect it."*

**MIT License** · © 2025 Agbozu Ebingiye Nelvin

</div>
