import numpy as np
import pandas as pd


def calculate_expected_returns(candidates: pd.DataFrame) -> pd.Series:
    """Calculate expected returns with more diversification"""
    df = candidates.copy()
    
    # Base return on probability
    df["expected_return"] = (df["uptrend_probability"] - 0.4) * 0.15
    
    # Add adjustment for recent performance
    if "return_20d" in df.columns:
        df["expected_return"] += df["return_20d"].clip(-0.1, 0.1) * 0.3
    
    # Ensure positive returns for high probability stocks
    df["expected_return"] = df["expected_return"].clip(0.02, 0.25)
    
    return df.set_index("symbol")["expected_return"]


def covariance_from_history(feature_df: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """Calculate covariance matrix from historical data"""
    if len(symbols) == 0:
        return pd.DataFrame()
    
    # Get price data
    pivot = feature_df[feature_df["symbol"].isin(symbols)].pivot_table(
        index="date",
        columns="symbol",
        values="close",
        aggfunc="last"
    ).sort_index()
    
    # Calculate returns
    returns = pivot.pct_change(fill_method=None).dropna(how="all")
    
    if returns.empty or len(returns.columns) < 2:
        # Return identity matrix if not enough data
        cov = pd.DataFrame(np.eye(len(symbols)), index=symbols, columns=symbols)
        cov = cov * 0.0004  # Scale appropriately
        return cov
    
    # Calculate covariance
    cov = returns.cov().fillna(0)
    
    # Add small regularization to ensure positive definite
    for s in symbols:
        if s in cov.index:
            cov.loc[s, s] = cov.loc[s, s] + 0.0001
    
    # Reindex to ensure all symbols are present
    cov = cov.reindex(index=symbols, columns=symbols).fillna(0)
    
    return cov


def optimize_portfolio(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    risk_profile: str = "Moderate",
    max_weight: float = 0.30,
    min_weight: float = 0.05,  # MINIMUM 5% per stock - ENSURES DIVERSIFICATION
) -> pd.Series:
    """
    Optimize portfolio allocation with diversification constraints
    """
    
    expected_returns = expected_returns.dropna()
    symbols = expected_returns.index.tolist()
    
    if len(symbols) == 0:
        return pd.Series(dtype=float)
    
    if len(symbols) == 1:
        return pd.Series([1.0], index=symbols)
    
    # Ensure covariance matrix is properly shaped
    covariance_matrix = covariance_matrix.reindex(index=symbols, columns=symbols).fillna(0)
    
    mu = expected_returns.values.astype(float)
    sigma = covariance_matrix.values.astype(float)
    
    # Risk aversion parameters
    risk_map = {
        "Conservative": 8.0,
        "Moderate": 5.0,
        "Aggressive": 3.0,
    }
    risk_aversion = risk_map.get(risk_profile, 5.0)
    
    # Calculate equal weight as baseline
    n = len(symbols)
    equal_weight = 1.0 / n
    
    # Set min_weight to ensure diversification
    effective_min_weight = max(min_weight, equal_weight * 0.5)  # At least 50% of equal weight
    effective_max_weight = min(max_weight, equal_weight * 3.0)  # At most 3x equal weight
    
    try:
        import cvxpy as cp
        
        w = cp.Variable(n)
        portfolio_return = mu @ w
        portfolio_risk = cp.quad_form(w, sigma)
        
        objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_risk)
        
        constraints = [
            cp.sum(w) == 1,
            w >= effective_min_weight,
            w <= effective_max_weight,
        ]
        
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS, verbose=False, max_iters=5000)
        
        if w.value is None:
            raise RuntimeError("CVXPY returned no solution")
        
        weights = pd.Series(np.array(w.value).reshape(-1), index=symbols)
        weights = weights.clip(lower=effective_min_weight, upper=effective_max_weight)
        
        # Normalize to sum to 1
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        return weights
        
    except Exception as e:
        print(f"Optimization failed: {e}, using equal weights")
        # Fallback to equal weights for diversification
        return pd.Series(equal_weight, index=symbols)


def create_investment_plan(
    weights: pd.Series,
    latest_prices: pd.Series,
    investment_amount: float
) -> pd.DataFrame:
    """Create investment plan from weights"""
    rows = []
    used_amount = 0.0
    
    for symbol, weight in weights.items():
        price = float(latest_prices.get(symbol, np.nan))
        amount = float(investment_amount * weight)
        
        if pd.isna(price) or price <= 0:
            shares = 0
            actual_amount = 0.0
        else:
            shares = int(amount // price)
            actual_amount = float(shares * price)
        
        used_amount += actual_amount
        
        rows.append({
            "symbol": symbol,
            "weight": float(weight),
            "allocation_percent": float(weight * 100),
            "recommended_amount_pkr": amount,
            "latest_price_pkr": price,
            "shares_to_buy": shares,
            "actual_investment_pkr": actual_amount,
        })
    
    cash = max(float(investment_amount - used_amount), 0.0)
    
    rows.append({
        "symbol": "CASH",
        "weight": cash / investment_amount if investment_amount > 0 else 0,
        "allocation_percent": cash / investment_amount * 100 if investment_amount > 0 else 0,
        "recommended_amount_pkr": cash,
        "latest_price_pkr": 1.0,
        "shares_to_buy": 0,
        "actual_investment_pkr": cash,
    })
    
    return pd.DataFrame(rows)