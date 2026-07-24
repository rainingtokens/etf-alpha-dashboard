"""Turn the score table into a long-only model portfolio with constraints."""
from __future__ import annotations

import json
import time

import pandas as pd

from app.config import (BUCKETS, INCUMBENT_SCORE_MARGIN, MAX_MONTHLY_TURNOVER,
                        MAX_POSITION, MIN_AUM, MIN_DOLLAR_VOLUME, MIN_POSITION,
                        PORTFOLIO_JSON)


def _liquidity_ok(row: pd.Series, fundamentals: pd.DataFrame,
                  last_px: pd.Series) -> bool:
    t = row.name
    if t not in fundamentals.index:
        return True  # unknown -> allow; quality score already penalizes
    f = fundamentals.loc[t]
    aum = f.get("totalAssets")
    vol = f.get("averageVolume")
    px = last_px.get(t)
    if pd.notna(aum) and aum < MIN_AUM:
        return False
    if pd.notna(vol) and pd.notna(px) and vol * px < MIN_DOLLAR_VOLUME:
        return False
    return True


def select_portfolio(scores: pd.DataFrame, fundamentals: pd.DataFrame,
                     prices: pd.DataFrame,
                     previous: dict[str, float] | None = None) -> pd.DataFrame:
    """Pick top ETFs per bucket, weight by score within bucket bands."""
    last_px = prices.ffill().iloc[-1]
    eligible = scores[scores["core_eligible"]].copy()
    eligible = eligible[eligible["above_200dma"].fillna(0) > 0]  # trend gate for adds

    picks: list[dict] = []
    for bucket, cfg in BUCKETS.items():
        pool = eligible[eligible["category"].isin(cfg["categories"])]
        pool = pool[pool.apply(_liquidity_ok, axis=1,
                               fundamentals=fundamentals, last_px=last_px)]
        if pool.empty:
            continue

        # incumbent stickiness: keep a currently-held name unless a challenger
        # beats it by a clear margin (keeps turnover and taxes down)
        chosen: list[str] = []
        if previous:
            incumbents = [t for t in previous if t in pool.index]
            for inc in sorted(incumbents, key=lambda t: -pool.loc[t, "composite"]):
                if len(chosen) >= cfg["picks"]:
                    break
                best = pool.index[0]
                if pool.loc[best, "composite"] - pool.loc[inc, "composite"] < INCUMBENT_SCORE_MARGIN:
                    chosen.append(inc)
        # avoid holding two near-identical vehicles (e.g. IVV + SPYM)
        rets_1y = prices[pool.index.intersection(prices.columns)].ffill().tail(252).pct_change()
        for t in pool.index:
            if len(chosen) >= cfg["picks"]:
                break
            if t in chosen:
                continue
            duplicate = any(
                t in rets_1y.columns and c in rets_1y.columns
                and rets_1y[t].corr(rets_1y[c]) > 0.98
                for c in chosen
            )
            if not duplicate:
                chosen.append(t)

        lo, hi = cfg["band"]
        top_score = pool.loc[chosen, "composite"].max()
        # optional buckets (lo == 0) are skipped when their best pick is weak
        if lo == 0 and top_score < 0.05:
            continue
        # bucket weight scales within its band by score strength
        strength = min(max(top_score, 0) / 0.8, 1.0)
        bucket_w = lo + (hi - lo) * strength

        sub = pool.loc[chosen]
        rel = sub["composite"].clip(lower=0.01)
        for t, w in (rel / rel.sum() * bucket_w).items():
            picks.append({"ticker": t, "bucket": bucket, "weight": w})

    port = pd.DataFrame(picks).set_index("ticker")
    port = port.join(scores[["name", "category", "composite", "momentum",
                             "valuation", "regime_fit", "quality",
                             "mom_12_1", "mom_6"]])

    # normalize, cap positions, drop dust
    port["weight"] /= port["weight"].sum()
    port["weight"] = port["weight"].clip(upper=MAX_POSITION)
    port = port[port["weight"] >= MIN_POSITION]
    port["weight"] /= port["weight"].sum()

    # turnover throttle vs previous portfolio
    if previous:
        prev = pd.Series(previous, dtype=float)
        cur = port["weight"]
        allt = cur.index.union(prev.index)
        delta = (cur.reindex(allt).fillna(0) - prev.reindex(allt).fillna(0))
        turnover = delta.abs().sum() / 2
        if turnover > MAX_MONTHLY_TURNOVER:
            scale = MAX_MONTHLY_TURNOVER / turnover
            blended = prev.reindex(allt).fillna(0) + delta * scale
            blended = blended[blended > MIN_POSITION / 2]
            port = port.reindex(blended.index.intersection(port.index))
            extra = blended.index.difference(port.index)
            port["weight"] = blended.reindex(port.index)
            port.attrs["turnover_capped"] = True
            port.attrs["kept_from_previous"] = list(extra)
        port.attrs["turnover"] = min(turnover, MAX_MONTHLY_TURNOVER)

    port["weight"] /= port["weight"].sum()
    return port.sort_values("weight", ascending=False)


def build_rationales(port: pd.DataFrame, regime: dict) -> pd.Series:
    """One-sentence-per-signal explanation for each position."""
    out = {}
    for t, r in port.iterrows():
        parts = []
        if r["momentum"] > 0.3:
            parts.append(f"strong medium-term momentum (12-1m {r['mom_12_1']:+.0%})")
        elif r["momentum"] < -0.3:
            parts.append("weak momentum (held for diversification/valuation)")
        if r["valuation"] > 0.3:
            parts.append("attractively valued vs peers")
        if r["regime_fit"] > 0.3:
            parts.append(f"fits the current regime ({regime['quadrant'].split(' (')[0]})")
        if r["quality"] > 0.2:
            parts.append("cheap, large and liquid vehicle")
        if not parts:
            parts.append("balanced contributor across signals")
        out[t] = f"{r['bucket'].replace('_', ' ')} sleeve: " + "; ".join(parts) + "."
    return pd.Series(out, name="rationale")


def diff_vs_previous(port: pd.DataFrame,
                     previous: dict[str, float] | None) -> pd.DataFrame:
    prev = pd.Series(previous or {}, dtype=float)
    allt = port.index.union(prev.index)
    cur = port["weight"].reindex(allt).fillna(0)
    old = prev.reindex(allt).fillna(0)
    diff = pd.DataFrame({"current": cur, "previous": old, "change": cur - old})
    diff["action"] = "hold"
    diff.loc[(diff.previous == 0) & (diff.current > 0), "action"] = "BUY (new)"
    diff.loc[(diff.previous > 0) & (diff.current == 0), "action"] = "SELL (exit)"
    diff.loc[(diff.change > 0.01) & (diff.previous > 0), "action"] = "add"
    diff.loc[(diff.change < -0.01) & (diff.current > 0), "action"] = "trim"
    return diff.sort_values("change", ascending=False)


def save_portfolio(port: pd.DataFrame) -> None:
    payload = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "weights": port["weight"].round(4).to_dict(),
    }
    PORTFOLIO_JSON.write_text(json.dumps(payload, indent=1))


def load_previous() -> dict[str, float] | None:
    if not PORTFOLIO_JSON.exists():
        return None
    try:
        return json.loads(PORTFOLIO_JSON.read_text())["weights"]
    except Exception:
        return None
