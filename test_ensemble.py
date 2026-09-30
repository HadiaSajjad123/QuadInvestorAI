# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, ".")

from src.recommendation_engine_enhanced import recommend_portfolio_advanced
import pandas as pd

print("=" * 60)
print("TESTING UPDATED RECOMMENDATION ENGINE")
print("=" * 60)

df = pd.read_csv("data/processed/features_dataset.csv", parse_dates=["date"])
live_df = pd.DataFrame({
    "symbol": ["KOHP"],
    "current": [25.50]
})

result = recommend_portfolio_advanced(
    historical_features_df=df,
    investment_amount=100000,
    risk_profile="Moderate",
    top_n=1,
    ensemble_method="weighted_average",
    live_df=live_df
)

kohp = result["plan"][result["plan"]["symbol"] == "KOHP"]
if len(kohp) > 0:
    prob = kohp.iloc[0]["uptrend_probability"]
    print(f"\nKOHP Ensemble Probability: {prob:.2%}")
    if 0.78 <= prob <= 0.82:
        print("✅ PERFECT! The 75% issue is FIXED!")
    else:
        print(f"⚠️ Shows {prob:.1%}")