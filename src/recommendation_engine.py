import json
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from .config import LIVE_DIR, LSTM_MODEL_PATH, XGB_MODEL_PATH, FEATURE_COLUMNS_PATH, SCALER_PATH
from .data_ingestion import fetch_market_watch
from .ensemble import add_ensemble_predictions, load_saved_threshold
from .explainability import get_shap_top_factors
from .feature_engineering import add_technical_indicators
from .optimizer import (
    calculate_expected_returns,
    covariance_from_history,
    create_investment_plan,
    optimize_portfolio,
)
from .report_generator import generate_investment_report, generate_portfolio_summary
from .symbol_utils import is_probably_equity


def load_feature_columns():
    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError("feature_columns.json not found. Train model first.")
    return json.loads(FEATURE_COLUMNS_PATH.read_text())


def load_xgb_model():
    if not XGB_MODEL_PATH.exists():
        raise FileNotFoundError("xgboost_model.pkl not found. Train model first.")
    return joblib.load(XGB_MODEL_PATH)


def load_lstm_model_if_available():
    if not LSTM_MODEL_PATH.exists():
        return None
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(LSTM_MODEL_PATH)
    except Exception:
        return None


def load_common_symbols(path="data/processed/common_usable_symbols.csv"):
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    col = "symbol" if "symbol" in df.columns else df.columns[0]
    return set(df[col].astype(str).str.upper().str.strip())


def prepare_latest_feature_rows(historical_df: pd.DataFrame):
    """Get latest feature rows for each symbol"""
    needed = {"rsi_14", "macd", "return_20d", "volume_ratio"}
    if not needed.issubset(set(historical_df.columns)):
        historical_df = add_technical_indicators(historical_df)
    
    historical_df = historical_df.copy()
    historical_df["symbol"] = historical_df["symbol"].astype(str).str.upper().str.strip()
    historical_df = historical_df[historical_df["symbol"].apply(is_probably_equity)]
    
    latest_rows = (
        historical_df
        .sort_values("date")
        .groupby("symbol", as_index=False)
        .tail(1)
    )
    return latest_rows.reset_index(drop=True), historical_df


def predict_candidates(historical_features_df: pd.DataFrame, use_lstm: bool = True) -> pd.DataFrame:
    """Predict uptrend probabilities for all stocks"""
    feature_cols = load_feature_columns()
    xgb_model = load_xgb_model()
    
    latest_rows, feature_df = prepare_latest_feature_rows(historical_features_df)
    
    # Filter to only features that exist
    available_features = [f for f in feature_cols if f in latest_rows.columns]
    latest_rows = latest_rows.dropna(subset=available_features)
    
    if len(latest_rows) == 0:
        raise ValueError("No valid rows after filtering features")
    
    # XGBoost predictions
    latest_rows["xgb_prob"] = xgb_model.predict_proba(latest_rows[available_features])[:, 1]
    
    # LSTM predictions if available
    if use_lstm:
        lstm_model = load_lstm_model_if_available()
        if lstm_model is not None:
            try:
                from .train_lstm import predict_lstm_enhanced
                lstm_probs = predict_lstm_enhanced(lstm_model, feature_df, feature_cols, sequence_length=30)
                latest_rows = latest_rows.merge(lstm_probs, on="symbol", how="left")
            except Exception as e:
                print(f"LSTM prediction failed: {e}")
                latest_rows["lstm_prob"] = pd.NA
        else:
            latest_rows["lstm_prob"] = pd.NA
    
    # Ensemble predictions
    predictions = add_ensemble_predictions(latest_rows)
    
    return predictions.sort_values("uptrend_probability", ascending=False).reset_index(drop=True)


def save_live_prediction_log(predictions: pd.DataFrame, plan: pd.DataFrame):
    """Save prediction log"""
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LIVE_DIR / "live_predictions_log.csv"
    
    selected_symbols = set(plan[plan["symbol"] != "CASH"]["symbol"].astype(str)) if len(plan) > 0 else set()
    
    log_df = predictions.copy()
    log_df["run_timestamp"] = pd.Timestamp.now()
    log_df["selected_in_portfolio"] = log_df["symbol"].isin(selected_symbols)
    
    keep_cols = ["run_timestamp", "date", "symbol", "close", "xgb_prob", "lstm_prob", 
                 "uptrend_probability", "prediction", "recommendation_label", "selected_in_portfolio"]
    keep_cols = [c for c in keep_cols if c in log_df.columns]
    log_df = log_df[keep_cols]
    
    if log_path.exists():
        old = pd.read_csv(log_path)
        log_df = pd.concat([old, log_df], ignore_index=True)
    
    log_df.to_csv(log_path, index=False)
    return log_path


def recommend_portfolio(
    historical_features_df: pd.DataFrame,
    investment_amount: float,
    risk_profile: str = "Moderate",
    top_n: int = 8,
    min_probability: float = 0.50,  # LOWERED from 0.55
    max_weight: float = 0.25,
    use_lstm: bool = True,
    live_df: pd.DataFrame = None,
    common_symbols_file: str = None,
    min_live_price: float = 5.0,
    min_live_volume: float = 10000,  # LOWERED from 50000
):
    """Generate investment recommendations"""
    
    print(f"Generating recommendations with: risk={risk_profile}, min_prob={min_probability}, top_n={top_n}")
    
    # Get predictions
    predictions = predict_candidates(historical_features_df, use_lstm=use_lstm)
    
    # Get live data
    if live_df is None:
        live_df = fetch_market_watch(save=True)
    
    live_df = live_df.copy()
    live_df["symbol"] = live_df["symbol"].astype(str).str.upper().str.strip()
    live_df = live_df[live_df["symbol"].apply(is_probably_equity)]
    
    # Softer filters - don't filter out too many stocks
    live_df = live_df[live_df["current"].fillna(0) >= min_live_price]
    # Volume filter is optional - comment out if too restrictive
    # live_df = live_df[live_df["volume"].fillna(0) >= min_live_volume]
    
    live_prices = live_df.drop_duplicates("symbol").set_index("symbol")["current"]
    
    # Filter predictions to live symbols
    predictions = predictions[predictions["symbol"].isin(live_prices.index)].copy()
    
    # Apply common symbols filter if provided
    if common_symbols_file:
        common_symbols = load_common_symbols(common_symbols_file)
        if common_symbols:
            predictions = predictions[predictions["symbol"].isin(common_symbols)].copy()
    
    if predictions.empty:
        raise ValueError("No usable common symbols found after filtering")
    
    # LOWER THE THRESHOLD to get more stocks
    effective_min_probability = max(min_probability, 0.45)  # Lower minimum
    
    # Get candidates above threshold
    candidates = predictions[predictions["uptrend_probability"] >= effective_min_probability].copy()
    
    print(f"Found {len(candidates)} candidates above {effective_min_probability:.2f} probability")
    
    if len(candidates) < 3:
        # If not enough candidates, lower the threshold further
        effective_min_probability = max(effective_min_probability - 0.10, 0.35)
        candidates = predictions[predictions["uptrend_probability"] >= effective_min_probability].copy()
        print(f"Lowered threshold to {effective_min_probability:.2f}, found {len(candidates)} candidates")
    
    if len(candidates) == 0:
        # Last resort: take top 8 stocks regardless of probability
        candidates = predictions.head(top_n).copy()
        print(f"Taking top {len(candidates)} stocks by probability")
    
    # Limit to top_n
    candidates = candidates.head(top_n).copy()
    
    # Calculate expected returns for optimization
    expected_returns = calculate_expected_returns(candidates)
    
    # Ensure we have enough symbols for diversification
    if len(expected_returns) < 2:
        # Add more stocks from predictions
        additional = predictions.head(top_n * 2)[~predictions["symbol"].isin(expected_returns.index)]
        for _, row in additional.iterrows():
            expected_returns[row["symbol"]] = row["uptrend_probability"] * 0.08
        expected_returns = expected_returns.head(top_n)
    
    symbols = expected_returns.index.tolist()
    print(f"Optimizing portfolio for {len(symbols)} symbols")
    
    # Calculate covariance matrix
    cov = covariance_from_history(historical_features_df, symbols)
    
    # Adjust max_weight based on number of stocks
    adjusted_max_weight = min(max_weight, 0.35)  # Cap at 35%
    if len(symbols) > 0:
        # Ensure max_weight doesn't prevent diversification
        adjusted_max_weight = max(adjusted_max_weight, 1.0 / len(symbols))
    
    # Optimize portfolio
    weights = optimize_portfolio(
        expected_returns,
        cov,
        risk_profile=risk_profile,
        max_weight=adjusted_max_weight,
        min_weight=0.03  # Minimum 3% allocation per stock
    )
    
    # Create investment plan
    plan = create_investment_plan(weights, live_prices, investment_amount)
    
    # Merge with prediction data
    plan = plan.merge(
        predictions[["symbol", "uptrend_probability", "recommendation_label", "xgb_prob", "lstm_prob"]],
        on="symbol",
        how="left"
    )
    
    # Calculate target returns
    target_return_percent = 12.0 if risk_profile == "Conservative" else 15.0 if risk_profile == "Moderate" else 18.0
    target_return_decimal = target_return_percent / 100
    
    risk_df = historical_features_df.copy()
    risk_df["symbol"] = risk_df["symbol"].astype(str).str.upper().str.strip()
    
    annual_risk = (
        risk_df[risk_df["symbol"].isin(plan["symbol"].astype(str))]
        .groupby("symbol")["return_1d"]
        .std()
        .mul(252 ** 0.5)
        .mul(100)
    )
    
    plan["target_return_percent"] = plan["symbol"].apply(lambda x: target_return_percent if x != "CASH" else 0.0)
    plan["target_price_pkr"] = plan.apply(
        lambda r: r["latest_price_pkr"] * (1 + target_return_decimal) if r["symbol"] != "CASH" else 1.0, axis=1
    )
    plan["target_profit_pkr"] = plan.apply(
        lambda r: r["actual_investment_pkr"] * target_return_decimal if r["symbol"] != "CASH" else 0.0, axis=1
    )
    plan["probability_weighted_target_profit_pkr"] = plan.apply(
        lambda r: r["target_profit_pkr"] * float(r.get("uptrend_probability", 0) or 0) if r["symbol"] != "CASH" else 0.0, axis=1
    )
    plan["estimated_annual_risk_percent"] = plan["symbol"].map(annual_risk).fillna(0)
    
    # Save log
    log_path = save_live_prediction_log(predictions, plan)
    
    # Generate reports and explanations
    xgb_model = load_xgb_model()
    feature_cols = load_feature_columns()
    available_features = [f for f in feature_cols if f in predictions.columns]
    
    reports = {}
    explanations = {}
    
    for _, row in plan[plan["symbol"] != "CASH"].iterrows():
        symbol = row["symbol"]
        sample = predictions[predictions["symbol"] == symbol]
        
        if not sample.empty and len(available_features) > 0:
            X_sample = sample[available_features].head(1)
            factors = get_shap_top_factors(xgb_model, X_sample, top_n=5)
            explanations[symbol] = factors
        else:
            explanations[symbol] = []
        
        reports[symbol] = generate_investment_report(
            symbol=symbol,
            probability=float(row.get("uptrend_probability", 0.5)),
            risk_profile=risk_profile,
            allocation_percent=float(row["allocation_percent"]),
            amount_pkr=float(row["actual_investment_pkr"]),
            latest_price=float(row["latest_price_pkr"]),
            shares=int(row["shares_to_buy"]),
            shap_factors=explanations[symbol],
            target_return_percent=float(row.get("target_return_percent", target_return_percent)),
            target_profit_pkr=float(row.get("target_profit_pkr", 0.0)),
            estimated_annual_risk_percent=float(row.get("estimated_annual_risk_percent", 0.0)),
        )
    
    summary = generate_portfolio_summary(plan, risk_profile, investment_amount)
    
    return {
        "plan": plan,
        "predictions": predictions,
        "reports": reports,
        "explanations": explanations,
        "summary": summary,
        "live_prediction_log": str(log_path),
    }