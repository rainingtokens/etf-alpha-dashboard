"""ETF-level metadata via yfinance Ticker.info: valuation, cost, size, liquidity.

Fetched per-ticker (slow), so callers should pass a candidate subset, not the
full universe. Results are cached to parquet for a week.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf

from app.config import CACHE_DIR, FUNDAMENTALS_CACHE_MAX_AGE_DAYS

FUND_PARQUET = CACHE_DIR / "fundamentals.parquet"

FIELDS = ["trailingPE", "priceToBook", "netExpenseRatio", "totalAssets",
          "averageVolume", "category", "longName", "yield"]


def _fetch_one(ticker: str) -> dict:
    row: dict = {"ticker": ticker}
    try:
        info = yf.Ticker(ticker).info or {}
        for f in FIELDS:
            row[f] = info.get(f)
    except Exception:
        pass
    row["fetched_at"] = time.time()
    return row


def fetch_fundamentals(tickers: list[str], refresh: bool = False,
                       max_workers: int = 8) -> pd.DataFrame:
    """Metadata frame indexed by ticker; fetches only stale/missing tickers."""
    tickers = sorted(set(tickers))
    cached = pd.read_parquet(FUND_PARQUET) if FUND_PARQUET.exists() else pd.DataFrame()

    max_age = FUNDAMENTALS_CACHE_MAX_AGE_DAYS * 86400
    to_fetch = tickers
    if not cached.empty and not refresh:
        fresh = cached[cached["fetched_at"] > time.time() - max_age]
        to_fetch = [t for t in tickers if t not in fresh.index]

    if to_fetch:
        rows = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_fetch_one, t): t for t in to_fetch}
            for fut in as_completed(futures):
                rows.append(fut.result())
        new = pd.DataFrame(rows).set_index("ticker")
        if cached.empty:
            cached = new
        else:
            cached = pd.concat([cached[~cached.index.isin(new.index)], new])
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached.to_parquet(FUND_PARQUET)

    have = [t for t in tickers if t in cached.index]
    # avoid clashing with the universe's own "category" column downstream
    return cached.loc[have].rename(columns={"category": "ms_category"})
