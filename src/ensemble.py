import json
from pathlib import Path

import pandas as pd


def load_saved_threshold(default: float = 0.50):
    path = Path("models/xgb_best_threshold.json")
    if not path.exists():
        return default
    try:
        return float(json.loads(path.read_text()).get("best_threshold", default))
    except Exception:
        return default


def ensemble_probability(xgb_prob, lstm_prob=None, xgb_weight: float = 0.7, lstm_weight: float = 0.3):
    """
    Ensemble with higher weight for XGBoost (better accuracy)
    """
    if lstm_prob is None or pd.isna(lstm_prob):
        return float(xgb_prob)
    return float(xgb_weight * xgb_prob + lstm_weight * lstm_prob)


def label_recommendation(probability: float, model_threshold: float):
    """
    Labels for investment recommendations
    """
    if probability >= 0.65:
        return "Strong Buy"
    if probability >= 0.55:
        return "Buy"
    if probability >= model_threshold:
        return "Watch"
    return "Avoid"


def add_ensemble_predictions(
    df: pd.DataFrame,
    xgb_weight: float = 0.7,
    lstm_weight: float = 0.3,
    threshold: float | None = None,
):
    df = df.copy()
    if "lstm_prob" not in df.columns:
        df["lstm_prob"] = pd.NA

    if threshold is None:
        threshold = load_saved_threshold(default=0.50)

    df["uptrend_probability"] = df.apply(
        lambda r: ensemble_probability(r["xgb_prob"], r["lstm_prob"], xgb_weight, lstm_weight), axis=1
    )

    df["prediction"] = (df["uptrend_probability"] >= threshold).astype(int)
    df["prediction_threshold"] = threshold
    df["recommendation_label"] = df["uptrend_probability"].apply(
        lambda p: label_recommendation(float(p), threshold)
    )

    return df