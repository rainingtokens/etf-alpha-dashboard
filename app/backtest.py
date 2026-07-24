"""Monthly-rebalance sanity-check backtest of the momentum/trend sleeve vs SPY.

Honest scope: only signals computable historically (momentum, 200dma trend,
bucket structure) are used. Valuation and prediction-market inputs have no
point-in-time history here, so live composite scores will differ. Treat results
as a sanity check of the engine, not a forecast.

Run: .venv/bin/python -m app.backtest
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import BENCHMARK, BUCKETS, CACHE_DIR, load_universe
from app.data.prices import fetch_prices

TXN_COST = 0.0005  # 5 bps per side
START = "2016-01-01"
EQUITY_CURVE_CSV = CACHE_DIR / "backtest_equity.csv"


def _bucket_midweights() -> dict[str, float]:
    mids = {b: (cfg["band"][0] + cfg["band"][1]) / 2 for b, cfg in BUCKETS.items()}
    total = sum(mids.values())
    return {b: w / total for b, w in mids.items()}


def run_backtest(prices: pd.DataFrame | None = None,
                 verbose: bool = True) -> dict:
    uni = load_universe()
    core = uni[uni["core_eligible"]]
    if prices is None:
        prices = fetch_prices(sorted(set(core["ticker"]) | {BENCHMARK}))
    px = prices.ffill(limit=5)

    cat_by_ticker = core.set_index("ticker")["category"]
    bucket_of: dict[str, str] = {}
    for b, cfg in BUCKETS.items():
        for c in cfg["categories"]:
            bucket_of[c] = b
    mid_w = _bucket_midweights()

    month_end = px.resample("ME").last()
    ma200 = px.rolling(200).mean().resample("ME").last()
    mom_12_1 = month_end.shift(1) / month_end.shift(12) - 1
    mom_6 = month_end / month_end.shift(6) - 1
    signal = mom_12_1.rank(axis=1, pct=True) * 0.6 + mom_6.rank(axis=1, pct=True) * 0.4
    above = month_end > ma200

    daily_ret = px.pct_change()
    dates = month_end.loc[START:].index
    weights_prev = pd.Series(dtype=float)
    port_curve, spy_curve, port_dates = [], [], []
    nav, spy_nav = 1.0, 1.0

    for i, d in enumerate(dates[:-1]):
        sig = signal.loc[d].dropna()
        ok = above.loc[d]
        sig = sig[ok.reindex(sig.index).fillna(False)]
        sig = sig[sig.index.isin(cat_by_ticker.index)]
        if sig.empty:
            continue

        target: dict[str, float] = {}
        by_bucket: dict[str, list[str]] = {}
        for t in sig.sort_values(ascending=False).index:
            b = bucket_of.get(cat_by_ticker[t])
            if b is None:
                continue
            picks = by_bucket.setdefault(b, [])
            if len(picks) < BUCKETS[b]["picks"]:
                picks.append(t)
        for b, picks in by_bucket.items():
            for t in picks:
                target[t] = mid_w[b] / len(picks)
        w = pd.Series(target)
        w /= w.sum()

        turnover = (w.reindex(w.index.union(weights_prev.index)).fillna(0)
                    - weights_prev.reindex(w.index.union(weights_prev.index)).fillna(0)
                    ).abs().sum() / 2
        cost = turnover * 2 * TXN_COST
        weights_prev = w

        nxt = dates[i + 1]
        period = daily_ret.loc[d + pd.Timedelta(days=1): nxt]
        pr = (period[w.index].fillna(0) * w).sum(axis=1)
        sr = period[BENCHMARK].fillna(0)
        nav *= float((1 + pr).prod()) * (1 - cost)
        spy_nav *= float((1 + sr).prod())
        port_curve.append(nav)
        spy_curve.append(spy_nav)
        port_dates.append(nxt)

    curve = pd.DataFrame({"strategy": port_curve, "spy": spy_curve},
                         index=pd.DatetimeIndex(port_dates, name="date"))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    curve.to_csv(EQUITY_CURVE_CSV)

    def stats(series: pd.Series) -> dict:
        rets = series.pct_change().dropna()
        years = (series.index[-1] - series.index[0]).days / 365.25
        cagr = series.iloc[-1] ** (1 / years) - 1
        vol = rets.std() * np.sqrt(12)
        dd = (series / series.cummax() - 1).min()
        return {"CAGR": cagr, "vol": vol, "sharpe": cagr / vol if vol else np.nan,
                "max_drawdown": dd}

    s_strat, s_spy = stats(curve["strategy"]), stats(curve["spy"])
    active = curve["strategy"].pct_change().dropna() - curve["spy"].pct_change().dropna()
    te = active.std() * np.sqrt(12)
    result = {
        "strategy": s_strat, "spy": s_spy,
        "tracking_error": te,
        "info_ratio": (s_strat["CAGR"] - s_spy["CAGR"]) / te if te else np.nan,
        "n_months": len(curve),
        "curve": curve,
    }

    if verbose:
        print(f"Backtest {curve.index[0].date()} → {curve.index[-1].date()} "
              f"({result['n_months']} months, monthly rebalance, {TXN_COST*1e4:.0f}bps/side)")
        row = "{:<12} {:>8} {:>8} {:>8} {:>8}"
        print(row.format("", "CAGR", "Vol", "Sharpe", "MaxDD"))
        for label, s in (("strategy", s_strat), ("SPY", s_spy)):
            print(row.format(label, f"{s['CAGR']:.1%}", f"{s['vol']:.1%}",
                             f"{s['sharpe']:.2f}", f"{s['max_drawdown']:.1%}"))
        print(f"tracking error {te:.1%}  |  info ratio {result['info_ratio']:.2f}")
    return result


if __name__ == "__main__":
    run_backtest()
