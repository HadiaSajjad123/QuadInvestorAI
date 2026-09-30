"""
SHAP Explainability Module for QuadInvestorAI
"""

import pandas as pd
import numpy as np


def get_shap_top_factors(model, X_sample: pd.DataFrame, top_n: int = 5):
    """
    Return top SHAP factors for one row or multiple rows.
    """
    try:
        import shap
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        
        if len(X_sample) == 1:
            values = shap_values[0]
        else:
            values = shap_values.mean(axis=0)
        
        row = X_sample.iloc[0]
        pairs = []
        
        for feature, shap_value in zip(X_sample.columns, values):
            try:
                actual_value = float(row[feature])
            except (ValueError, TypeError):
                actual_value = 0
            
            pairs.append({
                "feature": feature,
                "shap_value": float(shap_value),
                "actual_value": actual_value,
                "direction": "positive" if shap_value > 0 else "negative",
            })
        
        sorted_pairs = sorted(pairs, key=lambda x: abs(x["shap_value"]), reverse=True)
        return sorted_pairs[:top_n]
        
    except ImportError:
        print("⚠️ SHAP not installed. Install with: pip install shap")
        return _get_fallback_factors(model, X_sample, top_n)
    except Exception as e:
        print(f"⚠️ SHAP error: {e}")
        return _get_fallback_factors(model, X_sample, top_n)


def _get_fallback_factors(model, X_sample: pd.DataFrame, top_n: int = 5):
    """Fallback method when SHAP is not available."""
    try:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            return []
        
        row = X_sample.iloc[0]
        pairs = []
        
        n_features = min(len(X_sample.columns), len(importances))
        
        for feature, importance in zip(X_sample.columns[:n_features], importances[:n_features]):
            try:
                actual_value = float(row[feature])
            except (ValueError, TypeError):
                actual_value = 0
            
            pairs.append({
                "feature": feature,
                "shap_value": float(importance),
                "actual_value": actual_value,
                "direction": "positive",
            })
        
        return sorted(pairs, key=lambda x: abs(x["shap_value"]), reverse=True)[:top_n]
    except Exception as e:
        print(f"⚠️ Fallback error: {e}")
        return []


def get_shap_as_chart_data(shap_factors: list):
    """
    Convert SHAP factors into a dict ready for Plotly bar chart.
    """
    if not shap_factors:
        return {"features": [], "values": [], "colors": []}
    
    features = []
    values = []
    colors = []
    
    feature_mappings = {
        "rsi_14": "Momentum (RSI)",
        "macd_hist": "Trend Strength",
        "return_20d": "Recent Performance",
        "volume_ratio": "Trading Activity",
        "volatility_10": "Price Stability",
        "close_to_sma_20": "Price vs Avg",
        "sma_20": "Moving Average",
        "bollinger_position": "Price Range Position",
        "return_1d": "Daily Return",
        "close": "Current Price",
    }
    
    for f in shap_factors:
        raw_feature = f.get("feature", "")
        label = feature_mappings.get(raw_feature, raw_feature.replace("_", " ").title())
        v = f.get("shap_value", 0)
        features.append(label)
        values.append(v)
        colors.append("#22c55e" if v > 0 else "#ef4444")
    
    return {"features": features, "values": values, "colors": colors}


def get_lime_explanation(model, X_sample: pd.DataFrame, top_n: int = 5, num_samples: int = 500):
    """LIME explanation (optional)"""
    return []  # Return empty list if LIME not available


def get_lime_as_chart_data(lime_factors: list):
    """Convert LIME factors to chart data"""
    return {"features": [], "weights": [], "colors": []}


def factors_to_text(factors):
    """Convert SHAP factors to readable text"""
    if not factors:
        return "No explanation factors available."
    
    lines = []
    for f in factors:
        feature = f.get('feature', '')
        direction = f.get('direction', 'neutral')
        value = f.get('actual_value', 0)
        
        feature_mappings = {
            'rsi_14': 'Momentum (RSI)',
            'macd_hist': 'Trend Strength',
            'return_20d': 'Recent Performance',
            'volume_ratio': 'Trading Activity',
            'volatility_10': 'Price Stability',
        }
        display_name = feature_mappings.get(feature, feature.replace('_', ' ').title())
        
        if direction == 'positive':
            lines.append(f"✅ {display_name}: Positive impact (value: {value:.2f})")
        elif direction == 'negative':
            lines.append(f"⚠️ {display_name}: Negative impact (value: {value:.2f})")
        else:
            lines.append(f"➖ {display_name}: Neutral (value: {value:.2f})")
    
    return "\n".join(lines)


def lime_factors_to_text(lime_factors):
    """Convert LIME factors to readable text"""
    return "LIME explanations not available"


def format_shap_for_display(factors, max_factors=5):
    """Format SHAP factors for HTML display"""
    if not factors:
        return "<p>No explanation available</p>"
    
    html = "<ul style='margin: 0; padding-left: 1.2rem;'>"
    
    for f in factors[:max_factors]:
        feature = f.get('feature', '')
        direction = f.get('direction', 'neutral')
        value = f.get('actual_value', 0)
        
        feature_mappings = {
            'rsi_14': 'Momentum',
            'macd_hist': 'Trend',
            'return_20d': 'Performance',
            'volume_ratio': 'Volume',
            'volatility_10': 'Stability'
        }
        display_name = feature_mappings.get(feature, feature.replace('_', ' ').title())
        
        if direction == 'positive':
            html += f"<li>✅ <strong>{display_name}</strong>: Positive (value: {value:.2f})</li>"
        elif direction == 'negative':
            html += f"<li>⚠️ <strong>{display_name}</strong>: Negative (value: {value:.2f})</li>"
        else:
            html += f"<li>➖ <strong>{display_name}</strong>: Neutral (value: {value:.2f})</li>"
    
    html += "</ul>"
    return html


def format_lime_for_display(lime_factors, max_factors=5):
    """Format LIME factors for HTML display"""
    return "<p>LIME explanations not available for this prediction</p>"


__all__ = [
    'get_shap_top_factors',
    'get_lime_explanation',
    'get_shap_as_chart_data',
    'get_lime_as_chart_data',
    'factors_to_text',
    'lime_factors_to_text',
    'format_shap_for_display',
    'format_lime_for_display',
]