import numpy as np
import pandas as pd


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["symbol", "date"])
    output = []

    for symbol, group in df.groupby("symbol", sort=False):
        g = group.copy().sort_values("date")

        g["return_1d"] = g["close"].pct_change()
        g["return_5d"] = g["close"].pct_change(5)
        g["return_10d"] = g["close"].pct_change(10)
        g["return_20d"] = g["close"].pct_change(20)

        for window in [5, 10, 20, 50, 100, 200]:
            g[f"sma_{window}"] = g["close"].rolling(window).mean()
            g[f"close_to_sma_{window}"] = g["close"] / g[f"sma_{window}"] - 1

        g["ema_12"] = g["close"].ewm(span=12, adjust=False).mean()
        g["ema_26"] = g["close"].ewm(span=26, adjust=False).mean()
        g["macd"] = g["ema_12"] - g["ema_26"]
        g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()
        g["macd_hist"] = g["macd"] - g["macd_signal"]

        g["rsi_14"] = calculate_rsi(g["close"], 14)

        g["volatility_10"] = g["return_1d"].rolling(10).std()
        g["volatility_20"] = g["return_1d"].rolling(20).std()
        g["volatility_60"] = g["return_1d"].rolling(60).std()

        g["volume_sma_10"] = g["volume"].rolling(10).mean()
        g["volume_sma_20"] = g["volume"].rolling(20).mean()
        g["volume_ratio"] = g["volume"] / g["volume_sma_20"].replace(0, np.nan)

        g["bollinger_mid"] = g["close"].rolling(20).mean()
        g["bollinger_std"] = g["close"].rolling(20).std()
        g["bollinger_upper"] = g["bollinger_mid"] + 2 * g["bollinger_std"]
        g["bollinger_lower"] = g["bollinger_mid"] - 2 * g["bollinger_std"]
        g["bollinger_position"] = (
            (g["close"] - g["bollinger_lower"]) /
            (g["bollinger_upper"] - g["bollinger_lower"]).replace(0, np.nan)
        )

        g["high_low_spread"] = (g["high"] - g["low"]) / g["close"]
        g["open_close_spread"] = (g["close"] - g["open"]) / g["open"]

        output.append(g)

    final_df = pd.concat(output, ignore_index=True)
    final_df = final_df.replace([np.inf, -np.inf], np.nan)
    return final_df.dropna().reset_index(drop=True)


def create_target(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.01) -> pd.DataFrame:
    """Create binary label: 1 if future return over N days > threshold."""
    df = df.copy().sort_values(["symbol", "date"])
    output = []

    for symbol, group in df.groupby("symbol", sort=False):
        g = group.copy().sort_values("date")
        g["future_close"] = g["close"].shift(-horizon)
        g["future_return"] = (g["future_close"] - g["close"]) / g["close"]
        g["target"] = (g["future_return"] > threshold).astype(int)
        output.append(g)

    final_df = pd.concat(output, ignore_index=True)
    final_df = final_df.dropna(subset=["future_close", "future_return", "target"])
    return final_df.reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame):
    exclude = {
        "date", "symbol", "target", "future_close", "future_return",
        "sector", "listed_in", "symbol_raw", "fetched_at"
    }
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
