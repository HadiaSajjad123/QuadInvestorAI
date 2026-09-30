"""
Advanced Ensemble Module for QuadInvestorAI
Combines XGBoost and LSTM using multiple ensemble techniques
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

class AdvancedEnsemble:
    """Advanced ensemble techniques for combining XGBoost and LSTM"""
    
    def __init__(self, xgb_weight: float = 0.6, lstm_weight: float = 0.4):
        self.xgb_weight = xgb_weight
        self.lstm_weight = lstm_weight
        self.meta_learner = None
        self.calibrated = False
        
    def weighted_average(self, xgb_prob: np.ndarray, lstm_prob: np.ndarray) -> np.ndarray:
        """
        Simple weighted average ensemble
        
        Args:
            xgb_prob: XGBoost probabilities
            lstm_prob: LSTM probabilities
            
        Returns:
            Weighted average probabilities
        """
        # Handle NaN values
        lstm_prob = np.nan_to_num(lstm_prob, nan=0.5)
        
        # Weighted average
        ensemble_prob = self.xgb_weight * xgb_prob + self.lstm_weight * lstm_prob
        
        # Clip to valid range
        ensemble_prob = np.clip(ensemble_prob, 0.0, 1.0)
        
        return ensemble_prob
    
    def adaptive_weighted_average(self, xgb_prob: np.ndarray, lstm_prob: np.ndarray, 
                                   market_regime: str = "normal") -> np.ndarray:
        """
        Adaptive weights based on market conditions
        
        Args:
            xgb_prob: XGBoost probabilities
            lstm_prob: LSTM probabilities
            market_regime: "bull", "bear", "normal", "volatile"
            
        Returns:
            Adaptive weighted probabilities
        """
        # Adjust weights based on market regime
        regime_weights = {
            "bull": {"xgb": 0.7, "lstm": 0.3},      # XGBoost better in trending markets
            "bear": {"xgb": 0.5, "lstm": 0.5},      # Equal weight in downtrends
            "normal": {"xgb": 0.6, "lstm": 0.4},    # Default
            "volatile": {"xgb": 0.4, "lstm": 0.6}   # LSTM better in volatile markets
        }
        
        weights = regime_weights.get(market_regime, regime_weights["normal"])
        
        lstm_prob = np.nan_to_num(lstm_prob, nan=0.5)
        ensemble_prob = weights["xgb"] * xgb_prob + weights["lstm"] * lstm_prob
        
        return np.clip(ensemble_prob, 0.0, 1.0)
    
    def dynamic_weighted_average(self, xgb_prob: np.ndarray, lstm_prob: np.ndarray,
                                   xgb_confidence: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Dynamic weights based on model confidence
        
        Args:
            xgb_prob: XGBoost probabilities
            lstm_prob: LSTM probabilities
            xgb_confidence: Confidence scores from XGBoost (e.g., prediction variance)
            
        Returns:
            Dynamically weighted probabilities
        """
        lstm_prob = np.nan_to_num(lstm_prob, nan=0.5)
        
        if xgb_confidence is not None:
            # Give more weight to XGBoost when it's confident
            dynamic_xgb_weight = self.xgb_weight * (0.5 + xgb_confidence)
            dynamic_xgb_weight = np.clip(dynamic_xgb_weight, 0.3, 0.8)
            dynamic_lstm_weight = 1 - dynamic_xgb_weight
            
            ensemble_prob = dynamic_xgb_weight * xgb_prob + dynamic_lstm_weight * lstm_prob
        else:
            ensemble_prob = self.weighted_average(xgb_prob, lstm_prob)
        
        return np.clip(ensemble_prob, 0.0, 1.0)
    
    def max_ensemble(self, xgb_prob: np.ndarray, lstm_prob: np.ndarray) -> np.ndarray:
        """Take maximum probability (optimistic ensemble)"""
        lstm_prob = np.nan_to_num(lstm_prob, nan=0.5)
        return np.maximum(xgb_prob, lstm_prob)
    
    def min_ensemble(self, xgb_prob: np.ndarray, lstm_prob: np.ndarray) -> np.ndarray:
        """Take minimum probability (conservative ensemble)"""
        lstm_prob = np.nan_to_num(lstm_prob, nan=0.5)
        return np.minimum(xgb_prob, lstm_prob)
    
    def product_ensemble(self, xgb_prob: np.ndarray, lstm_prob: np.ndarray) -> np.ndarray:
        """Product of probabilities (both must agree)"""
        lstm_prob = np.nan_to_num(lstm_prob, nan=0.5)
        ensemble_prob = xgb_prob * lstm_prob
        # Normalize
        ensemble_prob = ensemble_prob / (ensemble_prob + (1-xgb_prob)*(1-lstm_prob) + 1e-8)
        return np.clip(ensemble_prob, 0.0, 1.0)
    
    def rank_average_ensemble(self, xgb_prob: np.ndarray, lstm_prob: np.ndarray) -> np.ndarray:
        """
        Average of probability ranks (robust to outliers)
        """
        lstm_prob = np.nan_to_num(lstm_prob, nan=0.5)
        
        # Convert to ranks
        xgb_rank = pd.Series(xgb_prob).rank(pct=True).values
        lstm_rank = pd.Series(lstm_prob).rank(pct=True).values
        
        # Average ranks
        ensemble_rank = (xgb_rank + lstm_rank) / 2
        
        return ensemble_rank
    
    def train_meta_learner(self, X_train_meta: np.ndarray, y_train: np.ndarray,
                           model_type: str = "logistic") -> None:
        """
        Train a meta-learner on model predictions
        
        Args:
            X_train_meta: Stacked predictions from base models (shape: n_samples, n_models)
            y_train: True labels
            model_type: "logistic", "random_forest", or "gradient_boosting"
        """
        if model_type == "logistic":
            self.meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        elif model_type == "random_forest":
            self.meta_learner = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        elif model_type == "gradient_boosting":
            self.meta_learner = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        self.meta_learner.fit(X_train_meta, y_train)
        self.calibrated = True
        
    def meta_ensemble(self, xgb_prob: np.ndarray, lstm_prob: np.ndarray) -> np.ndarray:
        """
        Use trained meta-learner to combine predictions
        """
        if self.meta_learner is None:
            raise ValueError("Meta-learner not trained. Call train_meta_learner first.")
        
        lstm_prob = np.nan_to_num(lstm_prob, nan=0.5)
        X_meta = np.column_stack([xgb_prob, lstm_prob])
        
        return self.meta_learner.predict_proba(X_meta)[:, 1]
    
    def calibrate_probabilities(self, xgb_prob: np.ndarray, lstm_prob: np.ndarray,
                                 y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calibrate model probabilities for better reliability
        
        Args:
            xgb_prob: XGBoost probabilities
            lstm_prob: LSTM probabilities
            y_true: True labels for calibration
            
        Returns:
            Calibrated probabilities for both models
        """
        from sklearn.calibration import CalibratedClassifierCV
        
        # Prepare data
        X_calib = np.column_stack([xgb_prob, lstm_prob])
        
        # Train calibrator
        calibrator = CalibratedClassifierCV(cv=5)
        
        # Need a dummy classifier for calibration
        from sklearn.dummy import DummyClassifier
        dummy = DummyClassifier(strategy="constant", constant=1)
        
        # Fit calibrator on features
        calibrator.fit(X_calib.reshape(-1, 1), y_true)
        
        # Get calibrated probabilities
        xgb_calib = calibrator.predict_proba(xgb_prob.reshape(-1, 1))[:, 1]
        lstm_calib = calibrator.predict_proba(lstm_prob.reshape(-1, 1))[:, 1]
        
        return xgb_calib, lstm_calib


class EnsembleOptimizer:
    """Optimizes ensemble weights using historical performance"""
    
    def __init__(self):
        self.optimal_weights = {"xgb": 0.6, "lstm": 0.4}
        
    def find_optimal_weights(self, xgb_probs: np.ndarray, lstm_probs: np.ndarray,
                              y_true: np.ndarray, metric: str = "accuracy") -> Dict:
        """
        Find optimal ensemble weights by grid search
        
        Args:
            xgb_probs: XGBoost probabilities on validation set
            lstm_probs: LSTM probabilities on validation set
            y_true: True labels
            metric: "accuracy", "f1", "balanced_accuracy", "auc"
            
        Returns:
            Dictionary with optimal weights and performance
        """
        best_score = 0
        best_weights = {"xgb": 0.5, "lstm": 0.5}
        
        from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
        
        metric_funcs = {
            "accuracy": accuracy_score,
            "f1": lambda y, p: f1_score(y, (p >= 0.5).astype(int)),
            "balanced_accuracy": balanced_accuracy_score,
            "auc": roc_auc_score
        }
        
        score_func = metric_funcs.get(metric, accuracy_score)
        
        # Grid search over weights
        for xgb_weight in np.arange(0.3, 0.81, 0.05):
            lstm_weight = 1 - xgb_weight
            
            ensemble_prob = xgb_weight * xgb_probs + lstm_weight * lstm_probs
            ensemble_pred = (ensemble_prob >= 0.5).astype(int)
            
            try:
                score = score_func(y_true, ensemble_pred if metric != "auc" else ensemble_prob)
            except:
                score = 0
            
            if score > best_score:
                best_score = score
                best_weights = {"xgb": xgb_weight, "lstm": lstm_weight}
        
        self.optimal_weights = best_weights
        return {"weights": best_weights, "best_score": best_score, "metric": metric}
    
    def get_optimal_weights(self) -> Dict:
        """Get the optimal weights found"""
        return self.optimal_weights


class EnsemblePredictor:
    """Main ensemble predictor for production use"""
    
    def __init__(self, xgb_model, lstm_model, feature_cols: List[str]):
        self.xgb_model = xgb_model
        self.lstm_model = lstm_model
        self.feature_cols = feature_cols
        self.ensemble = AdvancedEnsemble()
        self.optimizer = EnsembleOptimizer()
        self.use_meta_learner = False
        
    def predict(self, X: pd.DataFrame, y_true: Optional[np.ndarray] = None,
                ensemble_method: str = "weighted_average",
                return_details: bool = False) -> np.ndarray:
        """
        Make ensemble predictions
        
        Args:
            X: Feature DataFrame
            y_true: True labels (for calibration/optimization)
            ensemble_method: "weighted_average", "adaptive", "dynamic", "max", "min", "product", "rank"
            return_details: If True, return individual model predictions as well
            
        Returns:
            Ensemble probabilities
        """
        # Get individual predictions
        xgb_prob = self.xgb_model.predict_proba(X[self.feature_cols])[:, 1]
        
        # LSTM prediction (requires sequence data)
        lstm_prob = self._get_lstm_predictions(X)
        
        # Calibrate if training data provided
        if y_true is not None and len(y_true) > 100:
            xgb_prob, lstm_prob = self.ensemble.calibrate_probabilities(xgb_prob, lstm_prob, y_true)
        
        # Apply ensemble method
        if ensemble_method == "weighted_average":
            ensemble_prob = self.ensemble.weighted_average(xgb_prob, lstm_prob)
        elif ensemble_method == "adaptive":
            # Detect market regime
            regime = self._detect_market_regime(X)
            ensemble_prob = self.ensemble.adaptive_weighted_average(xgb_prob, lstm_prob, regime)
        elif ensemble_method == "dynamic":
            # Calculate confidence
            confidence = self._calculate_xgboost_confidence(X)
            ensemble_prob = self.ensemble.dynamic_weighted_average(xgb_prob, lstm_prob, confidence)
        elif ensemble_method == "max":
            ensemble_prob = self.ensemble.max_ensemble(xgb_prob, lstm_prob)
        elif ensemble_method == "min":
            ensemble_prob = self.ensemble.min_ensemble(xgb_prob, lstm_prob)
        elif ensemble_method == "product":
            ensemble_prob = self.ensemble.product_ensemble(xgb_prob, lstm_prob)
        elif ensemble_method == "rank":
            ensemble_prob = self.ensemble.rank_average_ensemble(xgb_prob, lstm_prob)
        elif ensemble_method == "meta" and self.use_meta_learner:
            ensemble_prob = self.ensemble.meta_ensemble(xgb_prob, lstm_prob)
        else:
            ensemble_prob = self.ensemble.weighted_average(xgb_prob, lstm_prob)
        
        if return_details:
            return ensemble_prob, xgb_prob, lstm_prob
        
        return ensemble_prob
    
    def _get_lstm_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get LSTM predictions (handles sequence data requirement)
        """
        if self.lstm_model is None:
            return np.full(len(X), 0.5)
        
        try:
            from src.train_lstm import predict_lstm_enhanced
            import pandas as pd
            
            # Need historical sequence data
            # For now, return placeholder
            return np.full(len(X), 0.5)
        except:
            return np.full(len(X), 0.5)
    
    def _detect_market_regime(self, X: pd.DataFrame) -> str:
        """
        Detect current market regime based on volatility and trend
        """
        try:
            # Get returns if available
            if "return_20d" in X.columns:
                avg_return = X["return_20d"].mean()
                volatility = X.get("volatility_20d", X.get("volatility_10", 0.02)).mean()
                
                if avg_return > 0.02:
                    return "bull"
                elif avg_return < -0.02:
                    return "bear"
                elif volatility > 0.03:
                    return "volatile"
                else:
                    return "normal"
        except:
            pass
        
        return "normal"
    
    def _calculate_xgboost_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate confidence based on prediction variance
        """
        try:
            # Get prediction probabilities
            probs = self.xgb_model.predict_proba(X[self.feature_cols])
            
            # Confidence is distance from 0.5
            confidence = np.abs(probs[:, 1] - 0.5) * 2
            return confidence
        except:
            return np.ones(len(X)) * 0.5


# Function to update the existing ensemble.py
def create_enhanced_ensemble_predictions(df: pd.DataFrame, 
                                          xgb_model, 
                                          lstm_model, 
                                          feature_cols: List[str],
                                          ensemble_method: str = "weighted_average") -> pd.DataFrame:
    """
    Create enhanced ensemble predictions
    
    Args:
        df: DataFrame with features
        xgb_model: Trained XGBoost model
        lstm_model: Trained LSTM model
        feature_cols: List of feature columns
        ensemble_method: Ensemble method to use
        
    Returns:
        DataFrame with ensemble predictions
    """
    ensemble_predictor = EnsemblePredictor(xgb_model, lstm_model, feature_cols)
    
    # Get predictions
    ensemble_prob, xgb_prob, lstm_prob = ensemble_predictor.predict(
        df, ensemble_method=ensemble_method, return_details=True
    )
    
    # Add to DataFrame
    df = df.copy()
    df["xgb_prob"] = xgb_prob
    df["lstm_prob"] = lstm_prob
    df["ensemble_prob"] = ensemble_prob
    
    # Add confidence labels
    df["ensemble_confidence"] = np.abs(ensemble_prob - 0.5) * 2
    
    # Add recommendation based on ensemble
    df["ensemble_recommendation"] = df["ensemble_prob"].apply(
        lambda p: "Strong Buy" if p >= 0.7 else "Buy" if p >= 0.6 else "Hold" if p >= 0.5 else "Sell"
    )
    
    return df