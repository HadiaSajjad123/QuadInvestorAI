from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .config import LIVE_MARKET_CSV

PSX_MARKET_WATCH_URL = "https://dps.psx.com.pk/market-watch"
PSX_EOD_URL = "https://dps.psx.com.pk/timeseries/eod/{symbol}"
PSX_INTRADAY_URL = "https://dps.psx.com.pk/timeseries/int/{symbol}"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def _clean_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("−", "-", regex=False)
        .str.strip()
    )

    cleaned = cleaned.replace({
        "": pd.NA,
        "nan": pd.NA,
        "None": pd.NA,
        "-": pd.NA
    })

    return pd.to_numeric(cleaned, errors="coerce")


def clean_symbol(symbol: str) -> str:
    if pd.isna(symbol):
        return ""

    symbol = str(symbol).strip().upper()

    # Removes extra tags if any appear as text.
    # Example: "PASL NC" -> "PASL"
    return symbol.split()[0]


def fetch_market_watch(save: bool = True) -> pd.DataFrame:
    """
    Fetch live market watch data from PSX DPS portal.

    This function DOES NOT use pd.read_html(response.text),
    because that causes FileNotFoundError in some pandas versions.
    """

    response = requests.get(
        PSX_MARKET_WATCH_URL,
        headers=HEADERS,
        timeout=30
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")

    if table is None:
        raise RuntimeError("No table found in PSX market-watch response.")

    rows = []

    for tr in table.select("tbody tr"):
        tds = tr.find_all("td")

        if len(tds) < 11:
            continue

        symbol = (
            tds[0].get("data-search")
            or tds[0].get("data-order")
            or tds[0].get_text(" ", strip=True)
        )

        def get_cell(index):
            return (
                tds[index].get("data-order")
                or tds[index].get_text(" ", strip=True)
            )

        rows.append({
            "symbol": symbol,
            "sector": get_cell(1),
            "listed_in": get_cell(2),
            "ldcp": get_cell(3),
            "open": get_cell(4),
            "high": get_cell(5),
            "low": get_cell(6),
            "current": get_cell(7),
            "change": get_cell(8),
            "change_percent": get_cell(9),
            "volume": get_cell(10),
        })

    if not rows:
        raise RuntimeError("PSX table found but no rows were parsed.")

    df = pd.DataFrame(rows)

    expected_cols = [
        "symbol",
        "sector",
        "listed_in",
        "ldcp",
        "open",
        "high",
        "low",
        "current",
        "change",
        "change_percent",
        "volume",
    ]

    for col in expected_cols:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[expected_cols]

    df["symbol_raw"] = df["symbol"].astype(str)
    df["symbol"] = df["symbol"].apply(clean_symbol)

    for col in [
        "ldcp",
        "open",
        "high",
        "low",
        "current",
        "change",
        "change_percent",
        "volume",
    ]:
        df[col] = _clean_numeric(df[col])

    df = df.dropna(subset=["symbol"])
    df = df[df["symbol"].str.len() > 0]
    df["fetched_at"] = pd.Timestamp.now()

    if save:
        LIVE_MARKET_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(LIVE_MARKET_CSV, index=False)

    return df


def fetch_eod_data(symbol: str) -> pd.DataFrame:
    symbol = clean_symbol(symbol)

    url = PSX_EOD_URL.format(symbol=symbol)

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30
    )
    response.raise_for_status()

    payload = response.json()

    if payload.get("status") != 1:
        return pd.DataFrame()

    rows = payload.get("data", [])

    if not rows:
        return pd.DataFrame()

    max_len = max(len(row) for row in rows)

    columns = ["timestamp", "close", "volume", "open"][:max_len]

    if len(columns) < max_len:
        columns += [f"raw_{i}" for i in range(len(columns), max_len)]

    df = pd.DataFrame(rows, columns=columns)

    df["symbol"] = symbol
    df["date"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        errors="coerce"
    ).dt.date

    df["date"] = pd.to_datetime(df["date"])

    for col in ["open", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("date")


def fetch_intraday_data(symbol: str) -> pd.DataFrame:
    symbol = clean_symbol(symbol)

    url = PSX_INTRADAY_URL.format(symbol=symbol)

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30
    )
    response.raise_for_status()

    payload = response.json()

    if payload.get("status") != 1:
        return pd.DataFrame()

    rows = payload.get("data", [])

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["timestamp", "price", "volume"])

    df["symbol"] = symbol
    df["datetime"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        errors="coerce"
    )

    return df.sort_values("datetime")


def load_historical_csv(path) -> pd.DataFrame:
    return pd.read_csv(path)


def safe_fetch_market_watch(fallback_csv: Optional[str] = None) -> pd.DataFrame:
    try:
        return fetch_market_watch(save=True)

    except Exception:
        if fallback_csv:
            return pd.read_csv(fallback_csv)

        if LIVE_MARKET_CSV.exists():
            return pd.read_csv(LIVE_MARKET_CSV)

        raise