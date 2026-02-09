Water-Quality-Index-Prediction-Using-ML-XGBoost-RF-LSTM
📌 Project Overview

This project demonstrates how Machine Learning (Random Forest & XGBoost) and Deep Learning (LSTM) can be used to predict Water Quality Index (WQI) from environmental parameters.

A synthetic dataset with over 100 data points was generated to mimic real-world water quality measurements such as pH, dissolved oxygen, turbidity, BOD, COD, nitrates, phosphates, TDS, conductivity, and fecal coliform levels.

The goal is to evaluate multiple models and compare their performance in predicting WQI.

📂 Dataset

A synthetic dataset (water_quality_synthetic.csv & .xlsx) is generated with 900 monthly records from 2010–2019.

Features include:

temperature_C

turbidity_NTU

pH

DO_mg_L

BOD_mg_L

COD_mg_L

nitrate_mg_L

phosphate_mg_L

TDS_mg_L

conductivity_uS_cm

fecal_coliform_CFU_100mL

Target:

WQI (Water Quality Index, scaled 0–100)

⚙️ Installation

Clone the repository and install dependencies:

git clone https://github.com/yourusername/Water-Quality-Index-Prediction-Using-ML-XGBoost-RF-LSTM.git
cd Water-Quality-Index-Prediction-Using-ML-XGBoost-RF-LSTM
pip install -r requirements.txt


Required libraries:

numpy

pandas

matplotlib

scikit-learn

xgboost (optional)

tensorflow (optional, for LSTM)

🚀 Usage

Run the Python script:

python water_quality_prediction.py


This will:

Generate synthetic dataset

Train models:

Random Forest

XGBoost (if installed)

LSTM (if TensorFlow is installed)

Evaluate models (MAE, RMSE, R²)

Save dataset to artifacts/

📊 Model Evaluation

Each model is evaluated using:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

R² Score

A leaderboard is printed to compare performance.

📈 Results Visualization

The project includes:

Actual vs Predicted plots

Feature importance for Random Forest & XGBoost

Training history plot for LSTM

📦 Output

artifacts/water_quality_synthetic.csv

artifacts/water_quality_synthetic.xlsx

Console logs showing model performance

🔮 Future Work

Test with real water quality datasets

Deploy as a web API for real-time WQI prediction

Integrate with IoT sensors for field applications

👨‍💻 Author

Name: AGBOZU EBINGIYE NELVIN

GitHub: https://github.com/Nelvinebi
LinkedIn: https://www.linkedin.com/in/agbozu-ebi/
