import pandas as pd

from .data_ingestion import clean_symbol


def normalize_historical_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    aliases = {
        "date": ["date", "time", "trading_date", "day"],
        "symbol": ["symbol", "ticker", "stock", "company", "script", "scrip", "name"],
        "open": ["open", "opening", "open_price"],
        "high": ["high", "high_price"],
        "low": ["low", "low_price"],
        "close": ["close", "closing", "close_price", "last", "price", "current"],
        "volume": ["volume", "vol", "turnover", "shares"],
    }

    rename_map = {}
    for standard, possible in aliases.items():
        for col in possible:
            if col in df.columns:
                rename_map[col] = standard
                break

    df = df.rename(columns=rename_map)

    required = ["date", "symbol", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns after normalization: {missing}. "
            f"Your CSV columns are: {list(df.columns)}"
        )

    return df[required]


def clean_historical_data(df: pd.DataFrame, min_records_per_stock: int = 300) -> pd.DataFrame:
    df = normalize_historical_columns(df)
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["symbol"] = df["symbol"].apply(clean_symbol)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = (
            df[col].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "symbol", "close"])
    df = df[df["symbol"].str.len() > 0]
    df = df.sort_values(["symbol", "date"])
    df = df.drop_duplicates(subset=["symbol", "date"], keep="last")

    numeric = ["open", "high", "low", "close", "volume"]
    df[numeric] = df.groupby("symbol")[numeric].ffill()
    df[numeric] = df.groupby("symbol")[numeric].bfill()

    counts = df.groupby("symbol")["date"].transform("count")
    df = df[counts >= min_records_per_stock]

    # Remove impossible values.
    df = df[(df["close"] > 0) & (df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0)]
    df["volume"] = df["volume"].fillna(0).clip(lower=0)

    return df.reset_index(drop=True)
