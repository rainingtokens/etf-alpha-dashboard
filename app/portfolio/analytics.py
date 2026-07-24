"""Benchmark-relative risk statistics for a proposed portfolio."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import BENCHMARK

TRADING_YEAR = 252


def portfolio_stats(weights: pd.Series, prices: pd.DataFrame,
                    benchmark: str = BENCHMARK, lookback_days: int = 504) -> dict:
    tickers = [t for t in weights.index if t in prices.columns]
    if benchmark not in prices.columns or not tickers:
        return {}

    px = prices[tickers + [benchmark]].ffill().dropna(how="all").tail(lookback_days + 1)
    rets = px.pct_change().dropna(how="all")
    w = weights.reindex(tickers).fillna(0)
    w = w / w.sum()

    port_ret = (rets[tickers] * w).sum(axis=1)
    bench_ret = rets[benchmark]
    both = pd.concat([port_ret, bench_ret], axis=1, keys=["port", "bench"]).dropna()
    if len(both) < 60:
        return {}

    active = both["port"] - both["bench"]
    cov = np.cov(both["port"], both["bench"])
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else np.nan

    def ann_ret(s: pd.Series) -> float:
        return (1 + s).prod() ** (TRADING_YEAR / len(s)) - 1

    return {
        "beta_vs_spy": beta,
        "correlation": both["port"].corr(both["bench"]),
        "tracking_error": active.std() * np.sqrt(TRADING_YEAR),
        "port_vol": both["port"].std() * np.sqrt(TRADING_YEAR),
        "bench_vol": both["bench"].std() * np.sqrt(TRADING_YEAR),
        "port_ann_return": ann_ret(both["port"]),
        "bench_ann_return": ann_ret(both["bench"]),
        "active_ann_return": ann_ret(both["port"]) - ann_ret(both["bench"]),
        "lookback_days": len(both),
    }
