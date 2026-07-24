"""Cross-sectional valuation score from holdings-based P/E and P/B.

Cheapness is measured *within category* (value vs value, EM vs EM) so the
score rewards the cheaper vehicle for a given exposure, and separately at the
category level vs SPY so cheap regions/styles get a mild portfolio-level tilt.
Tickers without data get a neutral 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _within_group_z(values: pd.Series, groups: pd.Series) -> pd.Series:
    def z(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        sd = s.std()
        if not np.isfinite(sd) or sd == 0 or s.notna().sum() < 3:
            return s * 0.0
        return ((s - s.mean()) / sd).clip(-3, 3)
    return values.groupby(groups).transform(z)


def valuation_scores(universe: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.Series:
    """Series indexed by ticker: higher = cheaper (better)."""
    df = universe.set_index("ticker").join(fundamentals, how="left")

    log_pe = np.log(df["trailingPE"].where(df["trailingPE"] > 0))
    log_pb = np.log(df["priceToBook"].where(df["priceToBook"] > 0))

    z_pe = _within_group_z(log_pe, df["category"])
    z_pb = _within_group_z(log_pb, df["category"])
    within = -(z_pe.fillna(0) * 0.6 + z_pb.fillna(0) * 0.4)

    # category-level tilt: median category P/E vs SPY-proxy (us_core median)
    cat_pe = np.log(df["trailingPE"].where(df["trailingPE"] > 0)).groupby(df["category"]).median()
    base = cat_pe.get("us_core", cat_pe.median())
    cat_tilt = (-(cat_pe - base)).clip(-1, 1)  # cheap category -> positive
    across = df["category"].map(cat_tilt).fillna(0)

    score = (0.7 * within + 0.3 * across).fillna(0.0)
    score.name = "valuation"
    return score
