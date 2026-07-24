"""Macro time series from FRED, with graceful degradation when no key is set.

Without a FRED key the regime model falls back to market-price proxies
(see app/signals/regime.py); this module simply returns what it can.
"""
from __future__ import annotations

import time

import pandas as pd
import requests

from app.config import CACHE_DIR, FRED_API_KEY

MACRO_PARQUET = CACHE_DIR / "macro.parquet"
MACRO_CACHE_MAX_AGE_HOURS = 20

FRED_SERIES = {
    "DGS10": "10Y Treasury yield",
    "DGS2": "2Y Treasury yield",
    "T10Y2Y": "10Y-2Y curve slope",
    "BAMLH0A0HYM2": "High-yield OAS",
    "T10YIE": "10Y breakeven inflation",
    "DTWEXBGS": "Trade-weighted dollar",
    "UNRATE": "Unemployment rate",
    "INDPRO": "Industrial production",
    "CPIAUCSL": "CPI (SA)",
}


def _fetch_series(series_id: str) -> pd.Series | None:
    try:
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": series_id,
                "api_key": FRED_API_KEY,
                "file_type": "json",
                "observation_start": "2010-01-01",
            },
            timeout=20,
        )
        if r.status_code != 200:
            return None
        obs = r.json().get("observations", [])
        s = pd.Series(
            {o["date"]: float(o["value"]) for o in obs if o["value"] not in (".", "")},
            dtype=float,
        )
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    except Exception:
        return None


def get_macro(refresh: bool = False) -> pd.DataFrame:
    """Wide frame of FRED series (empty if no key / all fetches fail)."""
    if MACRO_PARQUET.exists() and not refresh:
        age_h = (time.time() - MACRO_PARQUET.stat().st_mtime) / 3600
        if age_h < MACRO_CACHE_MAX_AGE_HOURS:
            return pd.read_parquet(MACRO_PARQUET)

    if not FRED_API_KEY:
        return pd.DataFrame()

    data = {}
    for sid in FRED_SERIES:
        s = _fetch_series(sid)
        if s is not None and not s.empty:
            data[sid] = s
    df = pd.DataFrame(data)
    if not df.empty:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(MACRO_PARQUET)
    return df
