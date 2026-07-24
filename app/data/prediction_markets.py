"""Live macro odds from Polymarket (Gamma API) and Kalshi — both keyless for data.

Used only as regime inputs (Fed path, recession, inflation), never as direct
ETF-selection signals: these markets resolve in weeks-months, far short of the
2-5y investment horizon.
"""
from __future__ import annotations

import json
import time
from typing import Any

import requests

from app.config import CACHE_DIR, PM_CACHE_MAX_AGE_MINUTES

PM_CACHE = CACHE_DIR / "prediction_markets.json"

POLYMARKET_TAGS = ["fed-rates", "economy", "recession", "inflation"]
KALSHI_SERIES = {
    "KXFED": "Fed funds rate",
    "KXFEDDECISION": "Fed decision",
    "KXRECSSNBER": "US recession (NBER)",
    "KXCPIYOY": "CPI y/y",
    "KXU3": "Unemployment rate",
}
MACRO_KEYWORDS = ("fed", "rate", "recession", "inflation", "cpi", "gdp",
                  "unemployment", "tariff", "treasury", "economy")


def _fetch_polymarket() -> list[dict[str, Any]]:
    events: dict[str, dict] = {}
    for tag in POLYMARKET_TAGS:
        try:
            r = requests.get(
                "https://gamma-api.polymarket.com/events",
                params={"closed": "false", "order": "volume", "ascending": "false",
                        "limit": 8, "tag_slug": tag},
                timeout=15,
            )
            if r.status_code != 200:
                continue
            for e in r.json():
                title = e.get("title", "")
                if title in events:
                    continue
                if not any(k in title.lower() for k in MACRO_KEYWORDS):
                    continue
                outcomes = []
                for m in e.get("markets", [])[:12]:
                    try:
                        names = json.loads(m.get("outcomes", "[]"))
                        prices = [float(p) for p in json.loads(m.get("outcomePrices", "[]"))]
                    except Exception:
                        continue
                    q = m.get("groupItemTitle") or m.get("question", "")
                    if len(names) == 2 and names[0] == "Yes":
                        outcomes.append({"name": q, "prob": prices[0] if prices else None})
                    else:
                        for n, p in zip(names, prices):
                            outcomes.append({"name": f"{q}: {n}" if q else n, "prob": p})
                outcomes = [o for o in outcomes if o["prob"] is not None]
                outcomes.sort(key=lambda o: -o["prob"])
                if outcomes:
                    events[title] = {
                        "source": "Polymarket", "title": title,
                        "volume": float(e.get("volume") or 0),
                        "outcomes": outcomes[:8],
                        "url": f"https://polymarket.com/event/{e.get('slug','')}",
                    }
        except Exception:
            continue
    return sorted(events.values(), key=lambda e: -e["volume"])


def _fetch_kalshi() -> list[dict[str, Any]]:
    out = []
    for series, label in KALSHI_SERIES.items():
        try:
            r = requests.get(
                "https://api.elections.kalshi.com/trade-api/v2/markets",
                params={"status": "open", "limit": 20, "series_ticker": series},
                timeout=15,
            )
            if r.status_code != 200:
                continue
            markets = r.json().get("markets", [])
            # keep the nearest-dated, most-traded markets
            markets.sort(key=lambda m: (m.get("close_time") or "", -(m.get("volume") or 0)))
            for m in markets[:6]:
                # current API exposes prices as "*_dollars" strings; older
                # deployments used integer cents — support both
                prob = None
                for field, scale in (("last_price_dollars", 1.0),
                                     ("yes_bid_dollars", 1.0),
                                     ("last_price", 0.01), ("yes_bid", 0.01)):
                    v = m.get(field)
                    if v not in (None, ""):
                        prob = float(v) * scale
                        break
                if prob is None:
                    continue
                out.append({
                    "source": "Kalshi", "series": label,
                    "title": m.get("title", ""),
                    "subtitle": m.get("yes_sub_title", ""),
                    "prob": prob,
                    "volume": float(m.get("volume_fp") or m.get("volume") or 0),
                    "close_time": m.get("close_time", ""),
                })
        except Exception:
            continue
    return out


def get_prediction_markets(refresh: bool = False) -> dict[str, Any]:
    """{'polymarket': [...], 'kalshi': [...], 'fetched_at': ts} with 15-min cache."""
    if PM_CACHE.exists() and not refresh:
        age_min = (time.time() - PM_CACHE.stat().st_mtime) / 60
        if age_min < PM_CACHE_MAX_AGE_MINUTES:
            try:
                return json.loads(PM_CACHE.read_text())
            except Exception:
                pass

    data = {
        "polymarket": _fetch_polymarket(),
        "kalshi": _fetch_kalshi(),
        "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if data["polymarket"] or data["kalshi"]:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        PM_CACHE.write_text(json.dumps(data, indent=1))
    return data


def derive_regime_inputs(pm: dict[str, Any]) -> dict[str, float | None]:
    """Extract recession probability and expected Fed easing from raw markets."""
    recession_prob = None
    for m in pm.get("kalshi", []):
        if "recession" in m.get("title", "").lower():
            recession_prob = m["prob"]
            break
    if recession_prob is None:
        for e in pm.get("polymarket", []):
            if "recession" in e["title"].lower() and "us" in e["title"].lower():
                ys = [o for o in e["outcomes"] if o["name"].lower().startswith("yes")]
                if ys:
                    recession_prob = ys[0]["prob"]
                break

    # expected number of Fed cuts this year from a Polymarket bucket market
    expected_cuts = None
    for e in pm.get("polymarket", []):
        t = e["title"].lower()
        if "rate cut" in t and "how many" in t:
            num, den = 0.0, 0.0
            for o in e["outcomes"]:
                name = o["name"].lower()
                digits = [int(c) for c in name if c.isdigit()]
                if not digits:
                    continue
                n = digits[0] + (0.5 if "+" in name or "more" in name else 0)
                num += n * o["prob"]
                den += o["prob"]
            if den > 0.5:
                expected_cuts = num / den
            break

    return {"recession_prob": recession_prob, "expected_fed_cuts": expected_cuts}
