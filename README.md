# ETF Alpha Dashboard

Live decision-support dashboard for long-only ETF positioning across a broad
US-listed ETF universe, benchmarked against SPY over a 2–5 year horizon.

> ⚖️ Decision support only — not investment advice. Signals are derived from
> public market data. You are responsible for confirming any security is
> eligible for you to trade and for your own holding-period discipline. The
> app never places trades.

## Run

```bash
.venv/bin/streamlit run dashboard.py
```

First run downloads ~1,000 tickers of price history (≈2 min) and ETF metadata
for candidates (≈2 min); both are cached under `data/cache/` (prices refresh
after 20h, metadata after 7 days, prediction markets after 15 min).

Optional (recommended): copy `.env.example` to `.env` and add a free
[FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) for proper
macro series. Without it the regime model uses market-price proxies
(XLI/KXI for growth, TIP/IEF for inflation).

## How it works

- **Universe** — `scripts/build_universe.py` parses a source ETF list (PDF or
  CSV) into `data/universe.csv` (~1,080 rows; ~1,014 US-listed). Non-US listings
  and option-overlay (covered-call) funds are excluded from the core portfolio.
  Point it at an updated source list to refresh the universe.
- **Signals** (per-ETF composite): momentum 30% (12-1m + 6m, 200dma gate),
  valuation 25% (holdings P/E, P/B vs category peers), regime fit 25%
  (growth×inflation quadrant from macro data, shifted by Polymarket/Kalshi
  recession & Fed odds), vehicle quality 20% (expense, AUM, liquidity).
- **Portfolio** — top picks per bucket (US core / factor / small-mid / intl /
  EM / satellite) with weight bands, 25% position cap, liquidity floors,
  near-duplicate dedupe, incumbent stickiness, and a 30% monthly turnover cap
  vs your saved portfolio (`data/my_portfolio.json`, saved from the UI).
- **Backtest** — monthly-rebalance sanity check of the historically computable
  sleeve (momentum/trend/buckets) vs SPY since 2016. It validates the engine;
  it is not a forecast of the live composite.

## Design stances

- Prediction markets resolve in weeks–months → used only as regime inputs,
  never to pick ETFs.
- 2–5y alpha comes from valuation spreads, factor premia, momentum, and macro
  positioning — not from concentrated thematic bets; the portfolio targets
  ~3–7% tracking error with beta near 1.
