"""Medium-term momentum and trend signals from daily adjusted closes."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = {"1m": 21, "6m": 126, "12m": 252}


def _zscore(s: pd.Series) -> pd.Series:
    sd = s.std()
    if not np.isfinite(sd) or sd == 0:
        return s * 0.0
    return ((s - s.mean()) / sd).clip(-3, 3)


def momentum_frame(prices: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Per-ticker: mom_12_1, mom_6, above_200dma, and a blended momentum z-score."""
    px = prices if as_of is None else prices.loc[:as_of]
    px = px.ffill(limit=5)
    if len(px) < TRADING_DAYS["12m"] + TRADING_DAYS["1m"] + 5:
        raise ValueError("insufficient price history for momentum")

    last = px.iloc[-1]
    m1 = px.iloc[-TRADING_DAYS["1m"] - 1]
    m6 = px.iloc[-TRADING_DAYS["6m"] - 1]
    m12 = px.iloc[-TRADING_DAYS["12m"] - 1]

    mom_12_1 = m1 / m12 - 1          # classic 12-1 (skip last month)
    mom_6 = last / m6 - 1
    ma200 = px.rolling(200).mean().iloc[-1]
    above_200 = (last > ma200).astype(float)

    out = pd.DataFrame({
        "mom_12_1": mom_12_1,
        "mom_6": mom_6,
        "above_200dma": above_200,
    })
    out["momentum"] = 0.6 * _zscore(out["mom_12_1"]) + 0.4 * _zscore(out["mom_6"])
    return out
