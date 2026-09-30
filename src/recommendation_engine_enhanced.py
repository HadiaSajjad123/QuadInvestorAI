"""
Enhanced Recommendation Engine with SHAP Explainability
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict

from src.config import XGB_MODEL_PATH, LSTM_MODEL_PATH, FEATURE_COLUMNS_PATH
from src.data_ingestion import fetch_market_watch
from src.optimizer import optimize_portfolio, covariance_from_history
from src.report_generator import LLMReportGenerator
from src.explainability import get_shap_top_factors
from src.index_mapper import get_index_mapping, filter_by_index


def safe_lstm_predict(lstm_model, stock_data, features, sequence_length=30):
    """Safe LSTM prediction with automatic feature alignment"""
    lstm_feature_path = Path('models/lstm_feature_columns.json')
    if lstm_feature_path.exists():
        with open(lstm_feature_path, 'r') as f:
            lstm_features = json.load(f)
    else:
        lstm_features = features
    
    if len(stock_data) < sequence_length:
        return 0.5
    
    available_features = [f for f in lstm_features if f in stock_data.columns]
    
    if not available_features:
        return 0.5
    
    seq_data = stock_data[available_features].tail(sequence_length).values
    
    if len(available_features) != len(lstm_features):
        full_seq = np.zeros((sequence_length, len(lstm_features)))
        for i, feat in enumerate(lstm_features):
            if feat in available_features:
                col_idx = available_features.index(feat)
                full_seq[:, i] = seq_data[:, col_idx]
        seq_data = full_seq
    
    seq_data = np.nan_to_num(seq_data, nan=0.0, posinf=0.0, neginf=0.0)
    seq_data = seq_data.reshape(1, sequence_length, -1)
    
    try:
        pred = float(lstm_model.predict(seq_data, verbose=0)[0][0])
        return pred
    except Exception as e:
        print(f"LSTM prediction error: {e}")
        return 0.5


def recommend_portfolio_advanced(
    historical_features_df: pd.DataFrame,
    investment_amount: float,
    risk_profile: str = "Moderate",
    top_n: int = 8,
    min_probability: float = 0.50,
    max_weight: float = 0.30,
    use_lstm: bool = True,
    ensemble_method: str = "weighted_average",
    live_df: pd.DataFrame = None,
    market_filter: str = None,  # NEW: Market filter (KSE100, KSE30, etc.)
) -> Dict:
    """
    Enhanced recommendation with SHAP explainability and market filtering
    """
    print(f"🚀 Using ensemble method: {ensemble_method}")
    if market_filter:
        print(f"🎯 Market filter: {market_filter}")
    
    # Load models
    import importlib
    load_model = None
    for keras_module in ("tensorflow.keras.models", "keras.models"):
        try:
            load_model = importlib.import_module(keras_module).load_model
            break
        except (ImportError, ModuleNotFoundError):
            continue
    
    # Load XGBoost
    if not XGB_MODEL_PATH.exists():
        raise FileNotFoundError("XGBoost model not found")
    xgb_model = joblib.load(XGB_MODEL_PATH)
    print("✅ XGBoost model loaded")
    
    # Load LSTM if available
    lstm_model = None
    if use_lstm and LSTM_MODEL_PATH.exists():
        try:
            lstm_model = load_model(LSTM_MODEL_PATH)
            print("✅ LSTM model loaded")
        except Exception as e:
            print(f"⚠️ Could not load LSTM: {e}")
    else:
        print("⚠️ LSTM not available, using XGBoost only")
    
    # Load features
    feature_cols = json.loads(FEATURE_COLUMNS_PATH.read_text())
    print(f"✅ Loaded {len(feature_cols)} features")
    
    # Load LSTM features
    lstm_feature_path = Path('models/lstm_feature_columns.json')
    if lstm_feature_path.exists():
        with open(lstm_feature_path, 'r') as f:
            lstm_features = json.load(f)
        print(f"✅ Loaded {len(lstm_features)} LSTM features")
    else:
        lstm_features = feature_cols
    
    # Get live data
    if live_df is None:
        live_df = fetch_market_watch(save=True)
    
    live_df = live_df.copy()
    live_df["symbol"] = live_df["symbol"].astype(str).str.upper().str.strip()
    live_prices = live_df.drop_duplicates("symbol").set_index("symbol")["current"]
    print(f"✅ Got live prices for {len(live_prices)} stocks")
    
    # Get latest features
    latest = historical_features_df.sort_values("date").groupby("symbol").last().reset_index()
    latest = latest[latest["symbol"].isin(live_prices.index)]
    print(f"✅ Prepared features for {len(latest)} stocks")
    
    # Add missing features
    for feat in lstm_features:
        if feat not in latest.columns:
            latest[feat] = 0
    
    available_features = [f for f in feature_cols if f in latest.columns]
    
    # Get XGBoost predictions
    xgb_probs = xgb_model.predict_proba(latest[available_features])[:, 1]
    latest["xgb_prob"] = xgb_probs
    
    # Get LSTM predictions
    if lstm_model is not None:
        print("🔄 Computing LSTM predictions...")
        lstm_probs = []
        for idx, symbol in enumerate(latest["symbol"]):
            stock_data = historical_features_df[historical_features_df["symbol"] == symbol].sort_values("date")
            pred = safe_lstm_predict(lstm_model, stock_data, feature_cols)
            lstm_probs.append(pred)
            if (idx + 1) % 50 == 0:
                print(f"   Processed {idx + 1}/{len(latest)} stocks")
        latest["lstm_prob"] = lstm_probs
        print(f"✅ LSTM predictions complete")
    else:
        latest["lstm_prob"] = 0.5
    
    # Apply ensemble
    if ensemble_method == "weighted_average":
        latest["ensemble_prob"] = 0.6 * latest["xgb_prob"] + 0.4 * latest["lstm_prob"]
        print("✅ Using Weighted Average (60% XGBoost, 40% LSTM)")
    elif ensemble_method == "rank_average":
        xgb_rank = latest["xgb_prob"].rank(pct=True)
        lstm_rank = latest["lstm_prob"].rank(pct=True)
        latest["ensemble_prob"] = (xgb_rank + lstm_rank) / 2
    elif ensemble_method == "max":
        latest["ensemble_prob"] = np.maximum(latest["xgb_prob"], latest["lstm_prob"])
    elif ensemble_method == "min":
        latest["ensemble_prob"] = np.minimum(latest["xgb_prob"], latest["lstm_prob"])
    else:
        latest["ensemble_prob"] = latest["xgb_prob"]
    
    # Filter candidates
    candidates = latest[latest["ensemble_prob"] >= min_probability].copy()
    
    # ============================================
    # APPLY MARKET FILTER (NEW)
    # ============================================
    if market_filter:
        print(f"🔄 Applying market filter: {market_filter}")
        
        # Get index mapping
        mapping_df = get_index_mapping(live_df, historical_features_df)
        
        # Filter candidates by index
        candidates = filter_by_index(candidates, market_filter, mapping_df)
        print(f"   After filter: {len(candidates)} stocks remaining")
    
    # Take top N
    candidates = candidates.nlargest(top_n, "ensemble_prob")
    
    if len(candidates) == 0:
        print(f"⚠️ No stocks met criteria, taking top {top_n} stocks")
        candidates = latest.nlargest(top_n, "xgb_prob")
    
    # Get current prices
    candidates["current_price"] = candidates["symbol"].map(live_prices)
    candidates = candidates.dropna(subset=["current_price"])
    
    # Portfolio allocation
    n_stocks = min(len(candidates), top_n)
    weight_per_stock = min(1.0 / n_stocks, max_weight)
    weights = {symbol: weight_per_stock for symbol in candidates["symbol"].iloc[:n_stocks]}
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    # Create investment plan
    plan = []
    total_invested = 0
    
    for symbol, weight in weights.items():
        price = live_prices.get(symbol, 0)
        amount = investment_amount * weight
        shares = int(amount // price) if price > 0 else 0
        actual_amount = shares * price
        
        row = candidates[candidates["symbol"] == symbol].iloc[0]
        
        plan.append({
            "symbol": symbol,
            "weight": weight,
            "allocation_percent": weight * 100,
            "recommended_amount_pkr": amount,
            "latest_price_pkr": price,
            "shares_to_buy": shares,
            "actual_investment_pkr": actual_amount,
            "uptrend_probability": float(row["ensemble_prob"]),
            "xgb_prob": float(row["xgb_prob"]),
            "lstm_prob": float(row["lstm_prob"]),
        })
        total_invested += actual_amount
    
    # Add cash
    cash = investment_amount - total_invested
    if cash > 0:
        plan.append({
            "symbol": "CASH",
            "weight": cash / investment_amount,
            "allocation_percent": (cash / investment_amount) * 100,
            "recommended_amount_pkr": cash,
            "latest_price_pkr": 1,
            "shares_to_buy": 0,
            "actual_investment_pkr": cash,
            "uptrend_probability": 1.0,
            "xgb_prob": 1.0,
            "lstm_prob": 1.0,
        })
    
    plan_df = pd.DataFrame(plan)
    
    # Generate SHAP explanations
    print("\n🔍 Generating SHAP explanations...")
    
    explanations = {}
    
    for _, row in plan_df.iterrows():
        if row["symbol"] != "CASH":
            symbol = row["symbol"]
            stock_data = candidates[candidates["symbol"] == symbol]
            
            if len(stock_data) > 0 and len(available_features) > 0:
                X_sample = stock_data[available_features]
                
                try:
                    shap_factors = get_shap_top_factors(xgb_model, X_sample, top_n=5)
                    explanations[symbol] = shap_factors
                    print(f"   ✅ SHAP generated for {symbol}")
                except Exception as e:
                    print(f"   ⚠️ SHAP failed for {symbol}: {e}")
                    explanations[symbol] = []
            else:
                explanations[symbol] = []
    
    print(f"✅ SHAP explanations generated for {len(explanations)} stocks")
    
    # Generate AI reports
    reports = {}
    llm = LLMReportGenerator()
    
    for _, row in plan_df.iterrows():
        if row["symbol"] != "CASH":
            symbol = row["symbol"]
            prob = row["uptrend_probability"]
            factors = explanations.get(symbol, [])
            
            reports[symbol] = llm.generate_simple_report(
                symbol=symbol,
                probability=prob,
                allocation_percent=row["allocation_percent"],
                amount_pkr=row["actual_investment_pkr"],
                latest_price=row["latest_price_pkr"],
                shares=int(row["shares_to_buy"]),
                target_return_percent=15.0,
                target_profit_pkr=row["actual_investment_pkr"] * 0.15,
                risk_level=risk_profile,
                factors=factors
            )
    
    # Generate summary
    market_info = f" | Market: {market_filter}" if market_filter else ""
    summary_lines = [
        "╔══════════════════════════════════════════════════════════════╗",
        "║                    YOUR INVESTMENT PLAN                      ║",
        "╚══════════════════════════════════════════════════════════════╝",
        "",
        f"💰 TOTAL INVESTMENT: PKR {investment_amount:,.0f}",
        f"📊 RISK PROFILE: {risk_profile.upper()}{market_info}",
        f"📈 STOCKS SELECTED: {len(plan_df[plan_df['symbol'] != 'CASH'])}",
        "",
        "📈 WHERE YOUR MONEY GOES:",
    ]
    
    for _, row in plan_df.iterrows():
        if row["symbol"] != "CASH":
            prob = row["uptrend_probability"]
            summary_lines.append(f"  • {row['symbol']}: PKR {row['actual_investment_pkr']:,.0f} ({row['allocation_percent']:.1f}%) - Confidence: {prob:.0%}")
    
    if cash > 0:
        summary_lines.append(f"\n💵 CASH RESERVE: PKR {cash:,.0f}")
    
    summary_lines.extend([
        "",
        "⚠️ REMEMBER: This is AI-generated guidance, not guaranteed returns.",
        "---",
        f"QuadInvestorAI - Powered by XGBoost + LSTM Ensemble{market_info}",
    ])
    
    summary = "\n".join(summary_lines)
    
    return {
        "plan": plan_df,
        "predictions": candidates,
        "reports": reports,
        "explanations": explanations,
        "summary": summary,
        "ensemble_method": ensemble_method,
        "market_filter": market_filter,
    }


__all__ = ['recommend_portfolio_advanced']