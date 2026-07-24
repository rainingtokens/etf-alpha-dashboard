"""Blend momentum, valuation, regime fit, and vehicle quality into one score."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import SIGNAL_WEIGHTS
from app.signals.momentum import momentum_frame
from app.signals.regime import regime_fit_scores
from app.signals.valuation import valuation_scores


def _z(s: pd.Series) -> pd.Series:
    sd = s.std()
    if not np.isfinite(sd) or sd == 0:
        return s * 0.0
    return ((s - s.mean()) / sd).clip(-3, 3)


def quality_scores(fundamentals: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    """Vehicle quality: cheap, big, liquid. Neutral 0 where data is missing."""
    if fundamentals.empty:
        return pd.Series(dtype=float, name="quality")
    f = fundamentals.copy()
    last_px = prices.ffill().iloc[-1]
    f["dollar_vol"] = f["averageVolume"] * last_px.reindex(f.index)

    z_exp = _z(np.log(f["netExpenseRatio"].where(f["netExpenseRatio"] > 0)))
    z_aum = _z(np.log(f["totalAssets"].where(f["totalAssets"] > 0)))
    z_dv = _z(np.log(f["dollar_vol"].where(f["dollar_vol"] > 0)))

    q = (-0.4 * z_exp.fillna(0) + 0.3 * z_aum.fillna(0) + 0.3 * z_dv.fillna(0))
    q.name = "quality"
    return q


def build_scores(universe: pd.DataFrame, prices: pd.DataFrame,
                 fundamentals: pd.DataFrame, regime: dict) -> pd.DataFrame:
    """Score table for US-listed tickers with enough price history."""
    uni = universe[universe["listed_us"]].copy()
    have_px = [t for t in uni["ticker"] if t in prices.columns
               and prices[t].dropna().shape[0] > 300]
    uni = uni[uni["ticker"].isin(have_px)]

    mom = momentum_frame(prices[have_px])
    val = valuation_scores(uni, fundamentals)
    reg = regime_fit_scores(uni, regime)
    qual = quality_scores(fundamentals, prices)

    df = uni.set_index("ticker")[["name", "category", "core_eligible", "exclusion_reason"]]
    df = df.join(mom[["mom_12_1", "mom_6", "above_200dma", "momentum"]], how="left")
    df["valuation"] = val.reindex(df.index).fillna(0)
    df["regime_fit"] = reg.reindex(df.index).fillna(0)
    df["quality"] = qual.reindex(df.index).fillna(0)

    w = SIGNAL_WEIGHTS
    df["composite"] = (
        w["momentum"] * df["momentum"].fillna(0)
        + w["valuation"] * df["valuation"]
        + w["regime_fit"] * df["regime_fit"]
        + w["quality"] * df["quality"]
    )
    return df.sort_values("composite", ascending=False)
