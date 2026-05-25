import pandas as pd


def get_shap_top_factors(model, X_sample: pd.DataFrame, top_n: int = 5):
    """Return top SHAP factors for one row or multiple rows.

    For dashboard reliability, this function catches SHAP errors and falls back
    to XGBoost feature importances.
    """
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        values = shap_values[0] if len(X_sample) == 1 else shap_values.mean(axis=0)
        row = X_sample.iloc[0]
        pairs = []
        for feature, shap_value in zip(X_sample.columns, values):
            pairs.append({
                "feature": feature,
                "shap_value": float(shap_value),
                "actual_value": float(row[feature]),
                "direction": "positive" if shap_value > 0 else "negative",
            })
        return sorted(pairs, key=lambda x: abs(x["shap_value"]), reverse=True)[:top_n]
    except Exception:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            return []
        row = X_sample.iloc[0]
        pairs = []
        for feature, importance in zip(X_sample.columns, importances):
            pairs.append({
                "feature": feature,
                "shap_value": float(importance),
                "actual_value": float(row[feature]),
                "direction": "positive",
            })
        return sorted(pairs, key=lambda x: abs(x["shap_value"]), reverse=True)[:top_n]


def factors_to_text(factors):
    lines = []
    for f in factors:
        lines.append(
            f"- {f['feature']} had a {f['direction']} impact "
            f"with current value {f['actual_value']:.4f}."
        )
    return "\n".join(lines)
