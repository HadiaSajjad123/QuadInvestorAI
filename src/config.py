from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LIVE_DIR = DATA_DIR / "live"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

DEFAULT_HISTORICAL_CSV = RAW_DIR / "psx_kaggle.csv"
PROCESSED_FEATURES_CSV = PROCESSED_DIR / "features_dataset.csv"
LIVE_MARKET_CSV = LIVE_DIR / "latest_market_watch.csv"

XGB_MODEL_PATH = MODELS_DIR / "xgboost_model.pkl"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_model.keras"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"
METRICS_PATH = MODELS_DIR / "metrics.json"

for p in [RAW_DIR, PROCESSED_DIR, LIVE_DIR, MODELS_DIR, REPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)
