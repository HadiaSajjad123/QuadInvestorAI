import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from .config import FEATURE_COLUMNS_PATH, PROCESSED_FEATURES_CSV, XGB_MODEL_PATH
from .ensemble import load_saved_threshold
from .symbol_utils import is_probably_equity


def load_model_and_features():
    if not XGB_MODEL_PATH.exists():
        raise FileNotFoundError("models/xgboost_model.pkl not found. Train the model first.")
    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError("models/feature_columns.json not found. Train the model first.")
    model = joblib.load(XGB_MODEL_PATH)
    feature_cols = json.loads(FEATURE_COLUMNS_PATH.read_text())
    return model, feature_cols


def load_common_symbols(path="data/processed/common_usable_symbols.csv"):
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    col = "symbol" if "symbol" in df.columns else df.columns[0]
    return set(df[col].astype(str).str.upper().str.strip())


def load_features(features_csv=PROCESSED_FEATURES_CSV):
    df = pd.read_csv(features_csv, parse_dates=["date"])
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    return df


def predict_as_of_date(
    as_of_date: str,
    features_csv=PROCESSED_FEATURES_CSV,
    top_n: int = 10,
    min_probability: float = 0.60,
    common_symbols_file: str | None = "data/processed/common_usable_symbols.csv",
    equities_only: bool = True,
):
    model, feature_cols = load_model_and_features()
    model_threshold = load_saved_threshold(default=0.50)

    df = load_features(features_csv)
    as_of_date = pd.to_datetime(as_of_date)

    if equities_only:
        df = df[df["symbol"].apply(is_probably_equity)]

    common_symbols = load_common_symbols(common_symbols_file) if common_symbols_file else None
    if common_symbols:
        df = df[df["symbol"].isin(common_symbols)]

    hist = df[df["date"] <= as_of_date].copy()
    if hist.empty:
        raise ValueError(f"No data found on/before {as_of_date.date()}.")

    latest = hist.sort_values("date").groupby("symbol", as_index=False).tail(1).copy()
    latest = latest.dropna(subset=feature_cols + ["future_return", "target", "close"])
    if latest.empty:
        raise ValueError("No rows with complete features and future labels found.")

    latest["probability"] = model.predict_proba(latest[feature_cols])[:, 1]
    latest["predicted_label"] = (latest["probability"] >= model_threshold).astype(int)
    latest["model_threshold"] = model_threshold
    latest["actual_label"] = latest["target"].astype(int)
    latest["correct"] = latest["predicted_label"] == latest["actual_label"]

    ranked = latest.sort_values("probability", ascending=False).reset_index(drop=True)

    # Portfolio selection is stricter than model classification.
    selected = ranked[ranked["probability"] >= min_probability].head(top_n).copy()
    if selected.empty:
        selected = ranked.head(top_n).copy()

    return ranked, selected


def evaluate_predictions(df: pd.DataFrame):
    y_true = df["actual_label"].astype(int)
    y_pred = df["predicted_label"].astype(int)
    prob = df["probability"].astype(float)

    metrics = {
        "rows": int(len(df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "predicted_positive_rate": float(y_pred.mean()),
        "actual_positive_rate": float(y_true.mean()),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    metrics["auc_roc"] = float(roc_auc_score(y_true, prob)) if len(set(y_true)) > 1 else None
    return metrics


def portfolio_profit(selected: pd.DataFrame, investment_amount: float):
    if selected.empty:
        return {
            "investment_amount": float(investment_amount),
            "portfolio_return_percent": 0.0,
            "profit_pkr": 0.0,
            "ending_value_pkr": float(investment_amount),
            "win_rate": 0.0,
        }

    avg_return = float(selected["future_return"].mean())
    profit = float(investment_amount * avg_return)
    return {
        "investment_amount": float(investment_amount),
        "selected_stocks": int(len(selected)),
        "portfolio_return_percent": avg_return * 100,
        "profit_pkr": profit,
        "ending_value_pkr": float(investment_amount + profit),
        "win_rate": float((selected["future_return"] > 0).mean() * 100),
        "avg_selected_probability": float(selected["probability"].mean()),
    }


def run_single_date_backtest(
    as_of_date: str,
    investment_amount: float = 100000,
    top_n: int = 10,
    min_probability: float = 0.60,
    output_dir="reports/backtests",
):
    ranked, selected = predict_as_of_date(
        as_of_date=as_of_date,
        top_n=top_n,
        min_probability=min_probability,
    )

    all_metrics = evaluate_predictions(ranked)
    selected_metrics = evaluate_predictions(selected)
    profit = portfolio_profit(selected, investment_amount)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    date_str = pd.to_datetime(as_of_date).strftime("%Y-%m-%d")
    ranked_path = out / f"ranked_predictions_{date_str}.csv"
    selected_path = out / f"selected_portfolio_{date_str}.csv"
    report_path = out / f"backtest_report_{date_str}.json"

    cols = [
        "date", "symbol", "close", "probability", "model_threshold",
        "predicted_label", "actual_label", "future_return", "correct",
        "rsi_14", "macd_hist", "return_20d", "volume_ratio"
    ]
    cols = [c for c in cols if c in ranked.columns]

    ranked[cols].to_csv(ranked_path, index=False)
    selected[cols].to_csv(selected_path, index=False)

    report = {
        "as_of_date": date_str,
        "top_n": top_n,
        "min_probability_for_portfolio_selection": min_probability,
        "all_predictions_metrics": all_metrics,
        "selected_stocks_metrics": selected_metrics,
        "profit_simulation": profit,
        "ranked_predictions_file": str(ranked_path),
        "selected_portfolio_file": str(selected_path),
    }
    report_path.write_text(json.dumps(report, indent=2))
    return report


def run_date_range_backtest(
    start_date: str,
    end_date: str,
    step_days: int = 20,
    investment_amount: float = 100000,
    top_n: int = 10,
    min_probability: float = 0.60,
    output_dir="reports/backtests",
):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    dates = pd.date_range(start, end, freq=f"{step_days}D")

    rows = []
    for d in dates:
        try:
            report = run_single_date_backtest(
                as_of_date=str(d.date()),
                investment_amount=investment_amount,
                top_n=top_n,
                min_probability=min_probability,
                output_dir=output_dir,
            )
            p = report["profit_simulation"]
            m = report["selected_stocks_metrics"]
            rows.append({
                "date": str(d.date()),
                "selected_stocks": p.get("selected_stocks", 0),
                "portfolio_return_percent": p.get("portfolio_return_percent", 0),
                "profit_pkr": p.get("profit_pkr", 0),
                "ending_value_pkr": p.get("ending_value_pkr", investment_amount),
                "win_rate": p.get("win_rate", 0),
                "selected_accuracy": m.get("accuracy", 0),
                "selected_balanced_accuracy": m.get("balanced_accuracy", 0),
                "selected_f1": m.get("f1", 0),
            })
        except Exception as e:
            rows.append({"date": str(d.date()), "error": str(e)})

    summary_df = pd.DataFrame(rows)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / f"range_backtest_{start.date()}_{end.date()}.csv"
    summary_df.to_csv(summary_path, index=False)

    valid = summary_df.dropna(subset=["profit_pkr"]) if "profit_pkr" in summary_df else pd.DataFrame()
    summary = {
        "start_date": str(start.date()),
        "end_date": str(end.date()),
        "step_days": step_days,
        "tests_run": int(len(summary_df)),
        "valid_tests": int(len(valid)),
        "average_return_percent": float(valid["portfolio_return_percent"].mean()) if len(valid) else None,
        "total_profit_if_independent_tests_pkr": float(valid["profit_pkr"].sum()) if len(valid) else None,
        "average_selected_accuracy": float(valid["selected_accuracy"].mean()) if len(valid) else None,
        "average_selected_f1": float(valid["selected_f1"].mean()) if len(valid) else None,
        "summary_file": str(summary_path),
    }
    (out / f"range_backtest_summary_{start.date()}_{end.date()}.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Single historical date, e.g. 2014-01-15")
    parser.add_argument("--start", help="Start date for range backtest")
    parser.add_argument("--end", help="End date for range backtest")
    parser.add_argument("--step-days", type=int, default=20)
    parser.add_argument("--amount", type=float, default=100000)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--min-probability", type=float, default=0.60)
    args = parser.parse_args()

    if args.date:
        report = run_single_date_backtest(
            as_of_date=args.date,
            investment_amount=args.amount,
            top_n=args.top_n,
            min_probability=args.min_probability,
        )
        print(json.dumps(report, indent=2))
    elif args.start and args.end:
        summary = run_date_range_backtest(
            start_date=args.start,
            end_date=args.end,
            step_days=args.step_days,
            investment_amount=args.amount,
            top_n=args.top_n,
            min_probability=args.min_probability,
        )
        print(json.dumps(summary, indent=2))
    else:
        raise SystemExit("Use either --date YYYY-MM-DD or --start YYYY-MM-DD --end YYYY-MM-DD")


if __name__ == "__main__":
    main()
