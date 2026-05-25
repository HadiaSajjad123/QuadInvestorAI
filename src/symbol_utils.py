import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

PSX_MARKET_WATCH_URL = "https://dps.psx.com.pk/market-watch"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Instruments that are usually not ordinary equity shares and should normally
# be excluded from stock recommendation: TFCs, Sukuk, PIB/T-Bills/GIS, ETFs,
# preferred/right shares etc. You can relax this if your project requires ETFs.
EXCLUDE_PATTERNS = [
    r"TFC", r"ETF", r"PIB", r"GIS", r"TB", r"PFL", r"FRR", r"VRR", r"GVR",
    r"SUK", r"SC\d*", r"R\d+$", r"PS$", r"CPS", r"NCPS", r"PREF", r"RIGHT"
]


def clean_symbol(symbol: str) -> str:
    if pd.isna(symbol):
        return ""
    return str(symbol).strip().upper().split()[0]


def is_probably_equity(symbol: str) -> bool:
    """Return True for ordinary stock symbols, False for ETFs/bonds/TFCs/rights etc."""
    symbol = clean_symbol(symbol)
    if not symbol:
        return False

    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, symbol, flags=re.IGNORECASE):
            return False

    # Exclude government/debt-like symbols such as P01GIS031225, PK12TB...
    if re.match(r"^P\d{2}", symbol):
        return False
    if re.match(r"^PK\d{2}", symbol):
        return False

    return True


def get_old_dataset_symbols(csv_path="data/raw/psx_kaggle.csv", ticker_col="Ticker", equities_only=True):
    df = pd.read_csv(csv_path)
    if ticker_col not in df.columns:
        raise ValueError(f"Column '{ticker_col}' not found. Columns: {list(df.columns)}")
    symbols = sorted({clean_symbol(x) for x in df[ticker_col].dropna().unique()})
    if equities_only:
        symbols = [s for s in symbols if is_probably_equity(s)]
    return set(symbols)


def get_live_symbols(equities_only=True):
    html = requests.get(PSX_MARKET_WATCH_URL, headers=HEADERS, timeout=30).text
    soup = BeautifulSoup(html, "html.parser")
    symbols = set()
    for td in soup.select("tbody tr td:first-child"):
        sym = td.get("data-search") or td.get("data-order") or td.get_text(" ", strip=True)
        sym = clean_symbol(sym)
        if sym:
            symbols.add(sym)
    if equities_only:
        symbols = {s for s in symbols if is_probably_equity(s)}
    return symbols


def create_symbol_reports(
    csv_path="data/raw/psx_kaggle.csv",
    output_dir="data/processed",
    ticker_col="Ticker",
    equities_only=True,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    old_symbols = get_old_dataset_symbols(csv_path, ticker_col=ticker_col, equities_only=equities_only)
    live_symbols = get_live_symbols(equities_only=equities_only)

    common = sorted(old_symbols & live_symbols)
    live_only = sorted(live_symbols - old_symbols)
    old_only = sorted(old_symbols - live_symbols)

    pd.Series(common).to_csv(output_dir / "common_usable_symbols.csv", index=False, header=["symbol"])
    pd.Series(live_only).to_csv(output_dir / "live_missing_in_old_dataset.csv", index=False, header=["symbol"])
    pd.Series(old_only).to_csv(output_dir / "old_missing_in_live_dataset.csv", index=False, header=["symbol"])

    summary = {
        "old_dataset_symbols": len(old_symbols),
        "live_symbols": len(live_symbols),
        "common_usable_symbols": len(common),
        "live_missing_in_old_dataset": len(live_only),
        "old_missing_in_live_dataset": len(old_only),
        "common_file": str(output_dir / "common_usable_symbols.csv"),
        "live_only_file": str(output_dir / "live_missing_in_old_dataset.csv"),
        "old_only_file": str(output_dir / "old_missing_in_live_dataset.csv"),
    }

    pd.Series(summary).to_json(output_dir / "symbol_match_summary.json", indent=2)
    return summary


if __name__ == "__main__":
    summary = create_symbol_reports()
    print(summary)
