"""Central configuration: paths, keys, model parameters."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
UNIVERSE_CSV = DATA_DIR / "universe.csv"
PORTFOLIO_JSON = DATA_DIR / "my_portfolio.json"

load_dotenv(PROJECT_ROOT / ".env")

FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip()
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()

PRICE_HISTORY_START = "2013-01-01"
PRICE_CACHE_MAX_AGE_HOURS = 20
FUNDAMENTALS_CACHE_MAX_AGE_DAYS = 7
PM_CACHE_MAX_AGE_MINUTES = 15

BENCHMARK = "SPY"

# Composite signal weights
SIGNAL_WEIGHTS = {
    "momentum": 0.30,
    "valuation": 0.25,
    "regime_fit": 0.25,
    "quality": 0.20,
}

# Liquidity floors for core portfolio picks
MIN_AUM = 200e6
MIN_DOLLAR_VOLUME = 2e6

# Portfolio construction buckets: categories, weight band, max picks
BUCKETS = {
    "us_core": {"categories": ["us_core"], "band": (0.25, 0.45), "picks": 2},
    "us_factor": {"categories": ["us_factor", "us_equity_other"], "band": (0.10, 0.25), "picks": 2},
    "us_small_mid": {"categories": ["us_small_mid"], "band": (0.05, 0.15), "picks": 1},
    "intl": {"categories": ["intl_equity"], "band": (0.10, 0.25), "picks": 2},
    "em": {"categories": ["em_equity"], "band": (0.00, 0.10), "picks": 1},
    "satellite": {
        "categories": [
            "sector_thematic", "commodity", "real_estate", "preferred",
            "bond_treasury", "bond_tips", "bond_ig", "bond_hy",
        ],
        "band": (0.00, 0.15),
        "picks": 2,
    },
}

MAX_POSITION = 0.25
MIN_POSITION = 0.03
MAX_MONTHLY_TURNOVER = 0.30
# a challenger must beat the incumbent's composite by this margin to replace it
INCUMBENT_SCORE_MARGIN = 0.25

DISCLAIMER = (
    "Decision support only — not investment advice. All signals are derived from "
    "public market data. You remain responsible for confirming any security is "
    "eligible for you to trade and for your own long-term-investing and holding-"
    "period discipline. This tool never places trades."
)


def load_universe() -> pd.DataFrame:
    df = pd.read_csv(UNIVERSE_CSV)
    df["exclusion_reason"] = df["exclusion_reason"].fillna("")
    return df
