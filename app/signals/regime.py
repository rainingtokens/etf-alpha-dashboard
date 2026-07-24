"""Macro regime classification (growth x inflation quadrant) and category fit.

Growth and inflation trends come from FRED when available, otherwise from
market-price proxies computable from the universe itself:
  growth  ~ 6m relative return of industrials (XLI) vs consumer staples (KXI)
  inflation ~ 6m change of the TIP/IEF ratio (breakeven proxy)

Prediction-market inputs (recession probability, expected Fed cuts) adjust the
defensive tilt but never pick ETFs directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

QUADRANTS = {
    ("up", "down"): "Goldilocks (growth up, inflation down)",
    ("up", "up"): "Reflation (growth up, inflation up)",
    ("down", "down"): "Slowdown (growth down, inflation down)",
    ("down", "up"): "Stagflation (growth down, inflation up)",
}

# category tilt per quadrant, -1 (avoid) .. +1 (favor)
CATEGORY_TILTS = {
    "Goldilocks (growth up, inflation down)": {
        "us_core": 0.6, "us_factor": 0.5, "us_small_mid": 0.8, "em_equity": 0.6,
        "intl_equity": 0.4, "sector_thematic": 0.5, "real_estate": 0.3,
        "commodity": -0.4, "bond_treasury": -0.4, "bond_ig": -0.2, "bond_tips": -0.4,
        "bond_hy": 0.2, "preferred": 0.1, "us_equity_other": 0.3,
    },
    "Reflation (growth up, inflation up)": {
        "us_core": 0.3, "us_factor": 0.6, "us_small_mid": 0.6, "em_equity": 0.7,
        "intl_equity": 0.6, "sector_thematic": 0.3, "real_estate": 0.2,
        "commodity": 0.8, "bond_treasury": -0.8, "bond_ig": -0.5, "bond_tips": 0.3,
        "bond_hy": 0.1, "preferred": -0.2, "us_equity_other": 0.3,
    },
    "Slowdown (growth down, inflation down)": {
        "us_core": 0.5, "us_factor": 0.3, "us_small_mid": -0.4, "em_equity": -0.4,
        "intl_equity": 0.0, "sector_thematic": -0.2, "real_estate": 0.2,
        "commodity": -0.6, "bond_treasury": 0.8, "bond_ig": 0.5, "bond_tips": 0.0,
        "bond_hy": -0.4, "preferred": 0.2, "us_equity_other": 0.1,
    },
    "Stagflation (growth down, inflation up)": {
        "us_core": -0.2, "us_factor": 0.3, "us_small_mid": -0.5, "em_equity": 0.0,
        "intl_equity": 0.3, "sector_thematic": -0.3, "real_estate": -0.2,
        "commodity": 0.9, "bond_treasury": -0.5, "bond_ig": -0.4, "bond_tips": 0.6,
        "bond_hy": -0.5, "preferred": -0.3, "us_equity_other": 0.0,
    },
}

# style keywords refine the tilt within equity categories
STYLE_ADJUST = {
    "Goldilocks (growth up, inflation down)": {"growth": 0.2, "momentum": 0.2, "value": 0.0, "min_vol": -0.3, "dividend": -0.1, "quality": 0.1},
    "Reflation (growth up, inflation up)": {"growth": -0.2, "momentum": 0.1, "value": 0.3, "min_vol": -0.2, "dividend": 0.2, "quality": 0.0},
    "Slowdown (growth down, inflation down)": {"growth": 0.2, "momentum": -0.1, "value": -0.1, "min_vol": 0.3, "dividend": 0.2, "quality": 0.3},
    "Stagflation (growth down, inflation up)": {"growth": -0.3, "momentum": 0.0, "value": 0.3, "min_vol": 0.3, "dividend": 0.3, "quality": 0.2},
}

STYLE_PATTERNS = {
    "growth": r"Growth",
    "value": r"Value|RAFI|Fundamental|Cash Cows|Shareholder Yield|Revenue",
    "momentum": r"Momentum",
    "min_vol": r"Low Vol|Min Vol|Minimum Vol|Managed Vol|Volatility Wtd",
    "dividend": r"Dividend|Yield Focus|High Yield Equity",
    "quality": r"Quality|Moat|Capital Strength|Profitability|Free Cash Flow",
}


def _trend_from_series(s: pd.Series, months: int = 6) -> str:
    s = s.dropna()
    if len(s) < 30:
        return "flat"
    now = s.iloc[-1]
    then = s.iloc[max(0, len(s) - months * 21)] if s.index.freq is None else s.shift(months).iloc[-1]
    return "up" if now > then else "down"


def classify_regime(macro: pd.DataFrame, prices: pd.DataFrame,
                    pm_inputs: dict) -> dict:
    used = []
    growth = inflation = None

    if not macro.empty and "INDPRO" in macro:
        s = macro["INDPRO"].dropna()
        if len(s) > 8:
            growth = "up" if s.iloc[-1] > s.iloc[-7] else "down"   # 6m change (monthly series)
            used.append("FRED INDPRO 6m change")
    if not macro.empty and "T10YIE" in macro:
        s = macro["T10YIE"].dropna()
        if len(s) > 130:
            inflation = "up" if s.iloc[-1] > s.iloc[-127] else "down"
            used.append("FRED 10Y breakeven 6m change")

    if growth is None and {"XLI", "KXI"}.issubset(prices.columns):
        ratio = (prices["XLI"] / prices["KXI"]).dropna()
        growth = _trend_from_series(ratio)
        used.append("XLI/KXI cyclical-defensive ratio (proxy)")
    if inflation is None and {"TIP", "IEF"}.issubset(prices.columns):
        ratio = (prices["TIP"] / prices["IEF"]).dropna()
        inflation = _trend_from_series(ratio)
        used.append("TIP/IEF breakeven proxy")

    growth = growth or "up"
    inflation = inflation or "down"
    quadrant = QUADRANTS[(growth, inflation)]

    recession_prob = pm_inputs.get("recession_prob")
    expected_cuts = pm_inputs.get("expected_fed_cuts")
    defensive_shift = 0.0
    if recession_prob is not None and recession_prob > 0.35:
        defensive_shift = min((recession_prob - 0.35) * 2, 0.5)

    return {
        "growth": growth,
        "inflation": inflation,
        "quadrant": quadrant,
        "recession_prob": recession_prob,
        "expected_fed_cuts": expected_cuts,
        "defensive_shift": defensive_shift,
        "inputs_used": used,
    }


def regime_fit_scores(universe: pd.DataFrame, regime: dict) -> pd.Series:
    """Series indexed by ticker: how well each ETF fits the current regime."""
    quadrant = regime["quadrant"]
    tilts = CATEGORY_TILTS[quadrant]
    styles = STYLE_ADJUST[quadrant]

    base = universe["category"].map(tilts).fillna(0.0)

    adj = pd.Series(0.0, index=universe.index)
    for style, pat in STYLE_PATTERNS.items():
        mask = universe["name"].str.contains(pat, case=False, regex=True, na=False)
        adj[mask] += styles.get(style, 0.0)

    score = base + adj

    # prediction-market defensive shift: haircut risk-on categories, boost defensives
    shift = regime.get("defensive_shift", 0.0)
    if shift:
        risk_on = universe["category"].isin(["us_small_mid", "em_equity", "sector_thematic", "bond_hy"])
        defensive = universe["category"].isin(["bond_treasury", "bond_ig"]) | \
            universe["name"].str.contains(STYLE_PATTERNS["min_vol"], case=False, na=False)
        score[risk_on] -= shift
        score[defensive] += shift

    out = pd.Series(score.values, index=universe["ticker"].values, name="regime_fit")
    return out.clip(-1.5, 1.5)
