# QuadInvestorAI

AI-powered PSX portfolio advisor using:
- Kaggle PSX historical data
- Live PSX data from https://dps.psx.com.pk
- XGBoost
- LSTM
- Weighted ensemble
- CVXPY portfolio optimization
- SHAP explanations
- Streamlit dashboard
- Template-based GenAI-style investment reports

## Setup

```bash
cd QuadInvestorAI
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Add your Kaggle dataset

Put your 15-year PSX CSV file here:

```text
data/raw/psx_kaggle.csv
```

Expected columns can be any common variant of:

```text
Date, Symbol, Open, High, Low, Close, Volume
```

The code automatically normalizes names like `ticker`, `company`, `price`, `vol`, etc.

## Train models

```bash
python -m src.train_pipeline --csv data/raw/psx_kaggle.csv --horizon 5 --threshold 0.01
```

This creates:

```text
models/xgboost_model.pkl
models/lstm_model.keras
models/scaler.pkl
models/feature_columns.json
models/metrics.json
data/processed/features_dataset.csv
```

If TensorFlow is too heavy/slow, you can still run the dashboard with XGBoost only.

## Run dashboard

```bash
streamlit run app/streamlit_app.py
```

## Disclaimer

This is an educational AI system. It does not guarantee profit and is not professional financial advice.
