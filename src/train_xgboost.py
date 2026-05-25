import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_COLUMNS_PATH, METRICS_PATH, MODELS_DIR, SCALER_PATH, XGB_MODEL_PATH

THRESHOLD_PATH = MODELS_DIR / "xgb_best_threshold.json"


def time_based_split(df: pd.DataFrame, test_size: float = 0.2):
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def train_valid_split(train_df: pd.DataFrame, valid_size: float = 0.2):
    train_df = train_df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(train_df) * (1 - valid_size))
    return train_df.iloc[:split_idx].copy(), train_df.iloc[split_idx:].copy()


def find_stable_threshold(y_true, probs):
    """
    Select a threshold that avoids one-class biased predictions.
    It maximizes F1 while penalizing thresholds that predict almost everything
    as Uptrend or almost everything as Downtrend.
    """
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)
    actual_pos_rate = float(y_true.mean())

    min_pred_rate = max(0.15, actual_pos_rate * 0.55)
    max_pred_rate = min(0.65, actual_pos_rate * 1.45)

    candidates = []

    for threshold in np.arange(0.30, 0.71, 0.01):
        preds = (probs >= threshold).astype(int)
        pred_pos_rate = float(preds.mean())

        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        bal_acc = balanced_accuracy_score(y_true, preds)

        # Reject heavily biased thresholds.
        if pred_pos_rate < min_pred_rate or pred_pos_rate > max_pred_rate:
            continue

        # Penalize predicted positive rate far from actual positive rate.
        rate_penalty = abs(pred_pos_rate - actual_pos_rate)
        score = (0.70 * f1) + (0.30 * bal_acc) - (0.20 * rate_penalty)

        candidates.append({
            "threshold": float(threshold),
            "score": float(score),
            "f1": float(f1),
            "balanced_accuracy": float(bal_acc),
            "precision": float(precision),
            "recall": float(recall),
            "predicted_positive_rate": float(pred_pos_rate),
        })

    if not candidates:
        # Safe fallback: maximize balanced accuracy in a conservative threshold range.
        for threshold in np.arange(0.40, 0.61, 0.01):
            preds = (probs >= threshold).astype(int)
            candidates.append({
                "threshold": float(threshold),
                "score": float(balanced_accuracy_score(y_true, preds)),
                "f1": float(f1_score(y_true, preds, zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
                "precision": float(precision_score(y_true, preds, zero_division=0)),
                "recall": float(recall_score(y_true, preds, zero_divion=0)) if False else float(recall_score(y_true, preds, zero_division=0)),
                "predicted_positive_rate": float(preds.mean()),
            })

    best = max(candidates, key=lambda x: x["score"])
    return best


def train_xgboost_model(df: pd.DataFrame, feature_cols: list[str]):
    train_full_df, test_df = time_based_split(df)
    train_df, valid_df = train_valid_split(train_full_df)

    X_train = train_df[feature_cols]
    y_train = train_df["target"].astype(int)
    X_valid = valid_df[feature_cols]
    y_valid = valid_df["target"].astype(int)
    X_test = test_df[feature_cols]
    y_test = test_df["target"].astype(int)

    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, SCALER_PATH)

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    # Use softened class balancing. Full neg/pos weight can push the model to
    # predict almost everything as Uptrend, so sqrt gives a safer balance.
    raw_weight = negative_count / max(positive_count, 1)
    scale_pos_weight = float(np.sqrt(raw_weight))

    model = xgb.XGBClassifier(
        n_estimators=800,
        max_depth=3,
        learning_rate=0.02,
        subsample=0.80,
        colsample_bytree=0.80,
        min_child_weight=5,
        gamma=0.2,
        reg_lambda=4.0,
        reg_alpha=0.5,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    valid_probs = model.predict_proba(X_valid)[:, 1]
    threshold_info = find_stable_threshold(y_valid, valid_probs)
    best_threshold = float(threshold_info["threshold"])

    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)

    metrics = {
        "xgboost": {
            "accuracy": float(accuracy_score(y_test, test_preds)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, test_preds)),
            "auc_roc": float(roc_auc_score(y_test, test_probs)) if len(set(y_test)) > 1 else None,
            "precision": float(precision_score(y_test, test_preds, zero_division=0)),
            "recall": float(recall_score(y_test, test_preds, zero_division=0)),
            "f1": float(f1_score(y_test, test_preds, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, test_preds).tolist(),
            "best_threshold": best_threshold,
            "validation_threshold_info": threshold_info,
            "scale_pos_weight": scale_pos_weight,
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
            "test_rows": int(len(test_df)),
            "train_positive_rate": float(y_train.mean()),
            "valid_positive_rate": float(y_valid.mean()),
            "test_positive_rate": float(y_test.mean()),
            "test_predicted_positive_rate": float(test_preds.mean()),
        }
    }

    joblib.dump(model, XGB_MODEL_PATH)
    FEATURE_COLUMNS_PATH.write_text(json.dumps(feature_cols, indent=2))
    THRESHOLD_PATH.write_text(json.dumps({"best_threshold": best_threshold}, indent=2))
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    return model, metrics
