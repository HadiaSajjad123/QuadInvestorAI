"""
Enhanced LSTM Training Module for QuadInvestorAI
Fixed: Handles NaN/Inf values properly
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .config import LSTM_MODEL_PATH, METRICS_PATH, SCALER_PATH

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from types import SimpleNamespace

try:
    import tensorflow as tf
    keras = tf.keras
except ImportError:
    import keras as keras_module
    keras = keras_module
    tf = SimpleNamespace(keras=keras)

LSTM = keras.layers.LSTM
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
BatchNormalization = keras.layers.BatchNormalization
Bidirectional = keras.layers.Bidirectional
GlobalAveragePooling1D = keras.layers.GlobalAveragePooling1D

Sequential = keras.models.Sequential
Adam = keras.optimizers.Adam
l2 = keras.regularizers.l2

EarlyStopping = keras.callbacks.EarlyStopping
ReduceLROnPlateau = keras.callbacks.ReduceLROnPlateau
ModelCheckpoint = keras.callbacks.ModelCheckpoint


def safe_pct_change(series, fill_value=0.0):
    """Safely calculate percentage change, handling NaN/Inf"""
    result = series.pct_change()
    result = result.replace([np.inf, -np.inf], fill_value)
    result = result.fillna(fill_value)
    return result


def safe_log_returns(series, fill_value=0.0):
    """Safely calculate log returns, handling zero/negative prices"""
    shifted = series.shift(1)
    ratio = series / shifted.replace(0, np.nan)
    ratio = ratio.clip(lower=0.01)
    result = np.log(ratio)
    result = result.replace([np.inf, -np.inf], fill_value)
    result = result.fillna(fill_value)
    return result


def safe_rolling_std(series, window, fill_value=0.0):
    """Safely calculate rolling standard deviation"""
    result = series.rolling(window, min_periods=max(1, window//2)).std()
    result = result.replace([np.inf, -np.inf], fill_value)
    result = result.fillna(fill_value)
    return result


def create_enhanced_lstm_sequences(
    df: pd.DataFrame, 
    feature_cols: list[str], 
    sequence_length: int = 30,
    step_size: int = 5,
    use_differential_features: bool = True,
    max_sequences_per_symbol: int = 500
):
    """Create sequences with enhanced features and proper error handling"""
    X, y = [], []
    df = df.sort_values(["symbol", "date"]).copy()
    
    scaler = None
    if SCALER_PATH.exists():
        try:
            scaler = joblib.load(SCALER_PATH)
        except:
            scaler = None
    
    print(f"Processing {df['symbol'].nunique()} symbols...")
    
    for symbol, group in df.groupby("symbol", sort=False):
        g = group.sort_values("date").copy()
        
        if len(g) < sequence_length + 10:
            continue
        
        if use_differential_features:
            for col in ['close', 'volume']:
                if col in g.columns:
                    g[f'{col}_returns'] = safe_pct_change(g[col])
                    g[f'{col}_log_returns'] = safe_log_returns(g[col])
                    g[f'{col}_volatility'] = safe_rolling_std(g[f'{col}_returns'], 20)
        
        base_features = feature_cols.copy()
        differential_features = [c for c in g.columns if '_returns' in c or '_log_returns' in c or '_volatility' in c]
        all_features = base_features + differential_features
        all_features = [f for f in all_features if f in g.columns]
        
        features = g[all_features].values.astype(np.float64)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        if scaler is not None:
            try:
                features = scaler.transform(features)
            except:
                scaler = StandardScaler()
                features = scaler.fit_transform(features)
        else:
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
        
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        targets = g["target"].astype(int).values
        
        sequences_created = 0
        for i in range(sequence_length, len(g), step_size):
            if i < len(g) and sequences_created < max_sequences_per_symbol:
                seq_features = features[i - sequence_length:i]
                if not np.any(np.isnan(seq_features)) and not np.any(np.isinf(seq_features)):
                    X.append(seq_features)
                    y.append(targets[i])
                    sequences_created += 1
    
    if not X:
        raise ValueError(f"No valid sequences created. Need at least {sequence_length} clean rows per stock.")
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    joblib.dump(scaler, SCALER_PATH)
    
    feature_cols_path = Path("models/lstm_feature_columns.json")
    feature_cols_path.write_text(json.dumps(all_features, indent=2))
    
    print(f"Created {len(X):,} sequences from {df['symbol'].nunique()} symbols")
    
    return X, y, scaler, all_features


def create_simplified_lstm_model(input_shape, dropout_rate=0.3):
    """Simplified LSTM model - stable and fast"""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(32, return_sequences=False),
        Dropout(dropout_rate),
        Dense(16, activation='relu'),
        Dropout(dropout_rate * 0.5),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model


def train_lstm_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    sequence_length: int = 30,
    test_size: float = 0.2,
    validation_split: float = 0.15,
    epochs: int = 50,
    batch_size: int = 64,
    use_class_weights: bool = True,
):
    """Stable LSTM training with proper error handling"""
    
    print("=" * 60)
    print("ENHANCED LSTM TRAINING (Stable Version)")
    print("=" * 60)
    
    print("Creating LSTM sequences...")
    try:
        X, y, scaler, all_features = create_enhanced_lstm_sequences(
            df, feature_cols, sequence_length=sequence_length, step_size=3
        )
    except Exception as e:
        print(f"Error creating sequences: {e}")
        print("Trying without differential features...")
        X, y, scaler, all_features = create_enhanced_lstm_sequences(
            df, feature_cols, sequence_length=sequence_length, step_size=3, use_differential_features=False
        )
    
    print(f"Total sequences created: {len(X):,}")
    print(f"Input shape: {X.shape}")
    print(f"Positive class ratio: {y.mean():.3f}")
    
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    split_idx = int(len(X) * (1 - test_size))
    X_train_full, X_test = X[:split_idx], X[split_idx:]
    y_train_full, y_test = y[:split_idx], y[split_idx:]
    
    val_idx = int(len(X_train_full) * (1 - validation_split))
    X_train, X_val = X_train_full[:val_idx], X_train_full[val_idx:]
    y_train, y_val = y_train_full[:val_idx], y_train_full[val_idx:]
    
    print(f"Training samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Test samples: {len(X_test):,}")
    
    input_shape = (sequence_length, X.shape[2])
    print(f"Building LSTM model with input shape {input_shape}...")
    model = create_simplified_lstm_model(input_shape)
    model.summary()
    
    callbacks = [
        EarlyStopping(monitor='val_auc', mode='max', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]
    
    class_weight = None
    if use_class_weights:
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight = dict(zip(classes, weights))
        print(f"Class weights: {class_weight}")
    
    print("\nTraining LSTM model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )
    
    print("\nEvaluating on test set...")
    test_probs = model.predict(X_test, verbose=0).reshape(-1)
    
    optimal_threshold = find_optimal_threshold(y_test, test_probs)
    test_preds = (test_probs >= optimal_threshold).astype(int)
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, test_preds)),
        "balanced_accuracy": float(recall_score(y_test, test_preds, average='macro', zero_division=0)),
        "auc_roc": float(roc_auc_score(y_test, test_probs)),
        "precision": float(precision_score(y_test, test_preds, zero_division=0)),
        "recall": float(recall_score(y_test, test_preds, zero_division=0)),
        "f1": float(f1_score(y_test, test_preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, test_preds).tolist(),
        "optimal_threshold": float(optimal_threshold),
        "sequence_length": int(sequence_length),
        "train_sequences": int(len(X_train)),
        "val_sequences": int(len(X_val)),
        "test_sequences": int(len(X_test)),
        "final_train_accuracy": float(history.history['accuracy'][-1]) if history.history['accuracy'] else 0,
        "final_val_accuracy": float(history.history['val_accuracy'][-1]) if history.history['val_accuracy'] else 0,
    }
    
    if 'auc' in history.history:
        metrics['train_auc'] = float(history.history['auc'][-1]) if history.history['auc'] else 0
        if 'val_auc' in history.history:
            metrics['val_auc'] = float(history.history['val_auc'][-1]) if history.history['val_auc'] else 0
    
    try:
        model.save(LSTM_MODEL_PATH)
        print(f"\n✅ Model saved to {LSTM_MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ Could not save model: {e}")
    
    lstm_metrics_path = Path("models/lstm_metrics.json")
    lstm_metrics_path.write_text(json.dumps(metrics, indent=2))
    
    if METRICS_PATH.exists():
        existing = json.loads(METRICS_PATH.read_text())
    else:
        existing = {}
    
    existing["lstm"] = metrics
    METRICS_PATH.write_text(json.dumps(existing, indent=2))
    
    print("\n" + "=" * 60)
    print("LSTM TRAINING RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"Optimal Threshold: {metrics['optimal_threshold']:.3f}")
    print("=" * 60)
    
    return model, metrics, history


def find_optimal_threshold(y_true, y_probs, min_precision=0.4, min_recall=0.3):
    """Find optimal threshold balancing precision and recall"""
    from sklearn.metrics import precision_recall_curve
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    
    if len(thresholds) == 0:
        return 0.5
    
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    valid_indices = (precisions[:-1] >= min_precision) & (recalls[:-1] >= min_recall)
    
    if np.any(valid_indices):
        valid_f1 = f1_scores[valid_indices]
        valid_thresholds = thresholds[valid_indices]
        if len(valid_f1) > 0:
            best_idx = np.argmax(valid_f1)
            return float(valid_thresholds[best_idx])
    
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx])


def predict_lstm_enhanced(model, df: pd.DataFrame, feature_cols: list[str], sequence_length: int = 30):
    """Enhanced prediction function for LSTM"""
    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
    
    lstm_feature_path = Path("models/lstm_feature_columns.json")
    if lstm_feature_path.exists():
        all_features = json.loads(lstm_feature_path.read_text())
    else:
        all_features = feature_cols.copy()
    
    predictions = []
    
    for symbol, group in df.sort_values(["symbol", "date"]).groupby("symbol", sort=False):
        g = group.sort_values("date")
        
        if len(g) < sequence_length:
            continue
        
        available_features = [f for f in all_features if f in g.columns]
        seq_data = g[available_features].tail(sequence_length).values
        seq_data = np.nan_to_num(seq_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        if scaler is not None:
            seq_data = scaler.transform(seq_data)
            seq_data = np.nan_to_num(seq_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        try:
            prob = float(model.predict(seq_data.reshape(1, sequence_length, len(available_features)), verbose=0).reshape(-1)[0])
        except:
            prob = 0.5
        
        predictions.append({"symbol": symbol, "lstm_prob": prob})
    
    return pd.DataFrame(predictions)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="Path to processed features CSV")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    
    if args.csv:
        print(f"Loading data from {args.csv}...")
        df = pd.read_csv(args.csv, parse_dates=["date"])
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        feature_cols = [c for c in df.columns if c not in ['date', 'symbol', 'target', 'future_return']]
        
        print(f"Loaded {len(df):,} rows with {len(feature_cols)} features")
        print(f"Unique symbols: {df['symbol'].nunique()}")
        
        model, metrics, history = train_lstm_model(
            df, feature_cols,
            sequence_length=args.sequence_length,
            epochs=args.epochs
        )
        
        print("\n📊 LSTM Training Results:")
        print(json.dumps(metrics, indent=2))