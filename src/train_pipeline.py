import argparse
import json

from .config import METRICS_PATH, PROCESSED_FEATURES_CSV
from .data_ingestion import load_historical_csv
from .feature_engineering import add_technical_indicators, create_target, get_feature_columns
from .preprocessing import clean_historical_data
from .symbol_utils import is_probably_equity
from .train_lstm import train_lstm_model
from .train_xgboost import train_xgboost_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to Kaggle PSX historical CSV")
    parser.add_argument("--horizon", type=int, default=5, help="Future days for target")
    parser.add_argument("--threshold", type=float, default=0.01, help="Future return threshold")
    parser.add_argument("--skip-lstm", action="store_true", help="Skip LSTM training")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--min-records", type=int, default=300)
    parser.add_argument(
        "--include-non-equity",
        action="store_true",
        help="Include ETFs/TFCs/bonds/preferred/right shares",
    )
    args = parser.parse_args()

    print("Loading historical data...")
    raw_df = load_historical_csv(args.csv)

    print("Cleaning data...")
    clean_df = clean_historical_data(raw_df, min_records_per_stock=args.min_records)

    if not args.include_non_equity:
        before = clean_df["symbol"].nunique()
        clean_df = clean_df[clean_df["symbol"].apply(is_probably_equity)].copy()
        after = clean_df["symbol"].nunique()
        print(f"Filtered non-equity instruments: {before} -> {after} symbols")

    print("Adding technical indicators...")
    feature_df = add_technical_indicators(clean_df)

    print("Creating target labels...")
    final_df = create_target(feature_df, horizon=args.horizon, threshold=args.threshold)

    feature_cols = get_feature_columns(final_df)
    print(f"Rows: {len(final_df):,}, Stocks: {final_df['symbol'].nunique()}, Features: {len(feature_cols)}")

    PROCESSED_FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(PROCESSED_FEATURES_CSV, index=False)
    print(f"Saved processed features to {PROCESSED_FEATURES_CSV}")

    print("Training XGBoost...")
    _, xgb_metrics = train_xgboost_model(final_df, feature_cols)
    print(json.dumps(xgb_metrics, indent=2))

    if not args.skip_lstm:
        try:
            print("Training LSTM...")
            result = train_lstm_model(final_df, feature_cols, sequence_length=args.sequence_length)
            
            # Handle different return types (2 or 3 values)
            if isinstance(result, tuple):
                if len(result) == 3:
                    _, lstm_metrics, _ = result
                elif len(result) == 2:
                    _, lstm_metrics = result
                else:
                    lstm_metrics = result
            else:
                lstm_metrics = result
            
            print(json.dumps(lstm_metrics, indent=2))
        except Exception as e:
            print(f"LSTM training skipped/failed: {e}")

    print(f"Training complete. Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()