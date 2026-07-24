"""Daily adjusted-close prices for the universe, with a local parquet cache.

Primary source: Yahoo Finance (batched). Fallback per-ticker: Stooq.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

from app.config import CACHE_DIR, PRICE_CACHE_MAX_AGE_HOURS, PRICE_HISTORY_START

PRICES_PARQUET = CACHE_DIR / "prices.parquet"
BATCH_SIZE = 100


def _cache_age_hours(path: Path) -> float:
    if not path.exists():
        return float("inf")
    return (time.time() - path.stat().st_mtime) / 3600


def _download_yahoo(tickers: list[str], start: str) -> pd.DataFrame:
    """Batched yf.download; returns wide frame of adjusted closes."""
    frames = []
    for i in range(0, len(tickers), BATCH_SIZE):
        chunk = tickers[i : i + BATCH_SIZE]
        df = yf.download(chunk, start=start, progress=False, auto_adjust=True,
                         group_by="column", threads=True)
        if df is None or df.empty:
            continue
        close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]
        if not isinstance(df.columns, pd.MultiIndex):
            close.columns = chunk[:1]
        frames.append(close)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.sort_index()


def fetch_stooq(ticker: str, start: str = PRICE_HISTORY_START) -> pd.Series | None:
    """Per-ticker fallback: Stooq daily CSV (no key required)."""
    try:
        url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
        r = requests.get(url, timeout=15)
        if r.status_code != 200 or "Date" not in r.text[:50]:
            return None
        df = pd.read_csv(StringIO(r.text), parse_dates=["Date"]).set_index("Date")
        s = df["Close"]
        return s[s.index >= start]
    except Exception:
        return None


def fetch_prices(tickers: list[str], refresh: bool = False,
                 start: str = PRICE_HISTORY_START) -> pd.DataFrame:
    """Return wide DataFrame (date x ticker) of adjusted closes, cached locally."""
    tickers = sorted(set(tickers))
    cached = None
    if PRICES_PARQUET.exists():
        cached = pd.read_parquet(PRICES_PARQUET)

    fresh = _cache_age_hours(PRICES_PARQUET) < PRICE_CACHE_MAX_AGE_HOURS
    if cached is not None and not refresh and fresh:
        missing = [t for t in tickers if t not in cached.columns]
        if not missing:
            return cached[tickers]
    else:
        missing = tickers if cached is None else [t for t in tickers if t not in cached.columns]

    if cached is None or refresh or not fresh:
        # full (re)download for requested tickers, then merge over any extras we had
        new = _download_yahoo(tickers, start)
    else:
        new = _download_yahoo(missing, start) if missing else pd.DataFrame()

    if cached is not None and not new.empty:
        keep = [c for c in cached.columns if c not in new.columns]
        merged = pd.concat([new, cached[keep]], axis=1) if keep else new
    elif cached is not None and new.empty:
        merged = cached
    else:
        merged = new

    # Stooq fallback for tickers Yahoo returned nothing for
    still_missing = [t for t in tickers if t not in merged.columns
                     or merged[t].dropna().empty]
    for t in still_missing[:50]:  # bound fallback work per run
        s = fetch_stooq(t, start)
        if s is not None and not s.dropna().empty:
            merged[t] = s

    merged = merged.sort_index()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(PRICES_PARQUET)
    have = [t for t in tickers if t in merged.columns]
    return merged[have]


def price_cache_age_hours() -> float:
    return _cache_age_hours(PRICES_PARQUET)


def latest_price_date(prices: pd.DataFrame) -> datetime | None:
    return None if prices.empty else prices.index.max().to_pydatetime()
