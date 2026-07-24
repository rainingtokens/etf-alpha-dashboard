"""ETF Alpha Dashboard — long-only recommendations across a broad US-listed ETF universe.

Run: .venv/bin/streamlit run dashboard.py
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.config import (BENCHMARK, DISCLAIMER, FRED_API_KEY, SIGNAL_WEIGHTS,
                        load_universe)
from app.data.fundamentals import fetch_fundamentals
from app.data.macro import get_macro
from app.data.prediction_markets import derive_regime_inputs, get_prediction_markets
from app.data.prices import fetch_prices, price_cache_age_hours
from app.portfolio.analytics import portfolio_stats
from app.portfolio.constructor import (build_rationales, diff_vs_previous,
                                       load_previous, save_portfolio,
                                       select_portfolio)
from app.signals.composite import build_scores
from app.signals.momentum import momentum_frame
from app.signals.regime import classify_regime

st.set_page_config(page_title="ETF Alpha Dashboard", layout="wide")

ALWAYS_FETCH = {BENCHMARK, "XLI", "KXI", "TIP", "IEF"}  # benchmark + regime proxies


@st.cache_data(ttl=3600, show_spinner="Loading universe…")
def _universe() -> pd.DataFrame:
    return load_universe()


@st.cache_data(ttl=3600, show_spinner="Downloading prices (first run takes a few minutes)…")
def _prices(tickers: tuple[str, ...], refresh: bool) -> pd.DataFrame:
    return fetch_prices(list(tickers), refresh=refresh)


@st.cache_data(ttl=3600, show_spinner="Fetching ETF metadata (first run takes a few minutes)…")
def _fundamentals(tickers: tuple[str, ...], refresh: bool) -> pd.DataFrame:
    return fetch_fundamentals(list(tickers), refresh=refresh)


@st.cache_data(ttl=3600)
def _macro(refresh: bool) -> pd.DataFrame:
    return get_macro(refresh=refresh)


@st.cache_data(ttl=900)
def _pm(refresh: bool) -> dict:
    return get_prediction_markets(refresh=refresh)


def fmt_pct(x: float | None, digits: int = 1) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:.{digits}%}"


# ---------------------------------------------------------------- sidebar
st.sidebar.title("ETF Alpha Dashboard")
refresh = st.sidebar.button("🔄 Refresh live data")
if refresh:
    st.cache_data.clear()

uni = _universe()
us_tickers = tuple(sorted(set(uni.loc[uni.listed_us, "ticker"]) | ALWAYS_FETCH))
prices = _prices(us_tickers, refresh)

age_h = price_cache_age_hours()
last_date = prices.index.max().date() if not prices.empty else "—"
st.sidebar.caption(
    f"**Prices:** {prices.shape[1]} tickers through {last_date} "
    f"(cache {age_h:.1f}h old)")
st.sidebar.caption(f"**FRED key:** {'✅ set' if FRED_API_KEY else '⚠️ not set — using market proxies'}")

# candidate set for slow per-ticker metadata: top momentum per category + majors
mom_all = momentum_frame(prices)
cand = set()
uni_px = uni[uni.ticker.isin(mom_all.index)]
for cat, grp in uni_px.groupby("category"):
    ranked = mom_all.loc[grp.ticker.tolist(), "momentum"].sort_values(ascending=False)
    cand.update(ranked.head(30).index)
prev_weights = load_previous()
if prev_weights:
    cand.update(prev_weights)
cand.update(ALWAYS_FETCH & set(prices.columns))
fund = _fundamentals(tuple(sorted(cand)), refresh)

macro = _macro(refresh)
pm = _pm(refresh)
pm_inputs = derive_regime_inputs(pm)
regime = classify_regime(macro, prices, pm_inputs)

scores = build_scores(uni, prices, fund, regime)
port = select_portfolio(scores, fund, prices, previous=prev_weights)
port["rationale"] = build_rationales(port, regime)
stats = portfolio_stats(port["weight"], prices)

st.sidebar.caption(f"**Prediction markets:** fetched {pm.get('fetched_at', '—')}")
st.sidebar.divider()
st.sidebar.markdown(f"**Regime:** {regime['quadrant']}")
if regime["recession_prob"] is not None:
    st.sidebar.markdown(f"**Recession odds (PM):** {regime['recession_prob']:.0%}")
if regime["expected_fed_cuts"] is not None:
    st.sidebar.markdown(f"**Expected Fed cuts:** {regime['expected_fed_cuts']:.1f}")

# ---------------------------------------------------------------- tabs
tab_rec, tab_sig, tab_macro, tab_data, tab_bt, tab_uni = st.tabs(
    ["📌 Recommendations", "🔎 Signals", "🌍 Macro & prediction markets",
     "🛰 Live data", "📈 Backtest", "🗂 Universe"])

with tab_rec:
    st.warning(DISCLAIMER, icon="⚖️")
    st.subheader("Model portfolio (2–5y horizon, benchmark SPY)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Positions", len(port))
    c2.metric("Beta vs SPY", f"{stats.get('beta_vs_spy', float('nan')):.2f}" if stats else "—")
    c3.metric("Est. tracking error", fmt_pct(stats.get("tracking_error")) if stats else "—")
    c4.metric("2y active return (ann.)", fmt_pct(stats.get("active_ann_return")) if stats else "—")

    show = port.reset_index()[["ticker", "name", "bucket", "weight", "composite",
                               "momentum", "valuation", "regime_fit", "quality",
                               "rationale"]]
    st.dataframe(
        show.style.format({"weight": "{:.1%}", "composite": "{:+.2f}",
                           "momentum": "{:+.2f}", "valuation": "{:+.2f}",
                           "regime_fit": "{:+.2f}", "quality": "{:+.2f}"}),
        use_container_width=True, hide_index=True)

    fig = px.pie(port.reset_index(), values="weight", names="ticker", hole=0.45,
                 title="Target weights")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Changes vs saved portfolio")
    if prev_weights:
        diff = diff_vs_previous(port, prev_weights)
        moves = diff[diff.action != "hold"]
        if moves.empty:
            st.info("No changes recommended vs your saved portfolio.")
        else:
            st.dataframe(moves.style.format({"current": "{:.1%}", "previous": "{:.1%}",
                                             "change": "{:+.1%}"}),
                         use_container_width=True)
        turn = port.attrs.get("turnover")
        if turn is not None:
            st.caption(f"Implied turnover: {turn:.0%}"
                       + (" (capped at monthly limit)" if port.attrs.get("turnover_capped") else ""))
    else:
        st.info("No saved portfolio yet — save the current recommendation to track "
                "changes and enable turnover control at the next refresh.")
    if st.button("💾 Save this portfolio as my current holdings"):
        save_portfolio(port)
        st.success("Saved. Future runs will show buy/trim/hold changes vs this.")

with tab_sig:
    st.subheader("Scored universe")
    cats = ["(all)"] + sorted(scores["category"].unique())
    pick_cat = st.selectbox("Category", cats)
    view = scores if pick_cat == "(all)" else scores[scores.category == pick_cat]
    st.caption(f"Signal weights: {SIGNAL_WEIGHTS}")
    st.dataframe(
        view.reset_index()[["ticker", "name", "category", "composite", "momentum",
                            "valuation", "regime_fit", "quality", "mom_12_1",
                            "mom_6", "above_200dma", "core_eligible"]]
        .style.format({"composite": "{:+.2f}", "momentum": "{:+.2f}",
                       "valuation": "{:+.2f}", "regime_fit": "{:+.2f}",
                       "quality": "{:+.2f}", "mom_12_1": "{:+.1%}", "mom_6": "{:+.1%}"}),
        use_container_width=True, hide_index=True, height=420)

    st.subheader("ETF detail")
    sel = st.selectbox("Ticker", scores.index.tolist())
    if sel:
        c1, c2 = st.columns([3, 1])
        with c1:
            rel = (prices[[sel, BENCHMARK]].dropna().ffill())
            rel = rel / rel.iloc[0]
            figd = go.Figure()
            for col in rel.columns:
                figd.add_trace(go.Scatter(x=rel.index, y=rel[col], name=col))
            figd.update_layout(title=f"{sel} vs {BENCHMARK} (normalized)", height=350)
            st.plotly_chart(figd, use_container_width=True)
        with c2:
            r = scores.loc[sel]
            st.metric("Composite", f"{r.composite:+.2f}")
            for k in ["momentum", "valuation", "regime_fit", "quality"]:
                st.metric(k, f"{r[k]:+.2f}")
            if sel in fund.index:
                f = fund.loc[sel]
                st.caption(f"P/E {f.get('trailingPE') or '—'} · expense "
                           f"{f.get('netExpenseRatio') or '—'}% · AUM "
                           f"${(f.get('totalAssets') or 0)/1e9:.1f}B")

with tab_macro:
    st.subheader(f"Current regime: {regime['quadrant']}")
    st.caption("Inputs used: " + "; ".join(regime["inputs_used"]))
    cols = st.columns(3)
    cols[0].metric("Growth trend", regime["growth"])
    cols[1].metric("Inflation trend", regime["inflation"])
    cols[2].metric("PM recession odds",
                   fmt_pct(regime["recession_prob"], 0) if regime["recession_prob"] is not None else "—")

    if not macro.empty:
        pick = st.multiselect("FRED series", list(macro.columns),
                              default=[c for c in ["T10Y2Y", "BAMLH0A0HYM2", "T10YIE"]
                                       if c in macro.columns])
        if pick:
            figm = go.Figure()
            for c in pick:
                figm.add_trace(go.Scatter(x=macro.index, y=macro[c], name=c))
            figm.update_layout(height=350)
            st.plotly_chart(figm, use_container_width=True)
    else:
        st.info("Set FRED_API_KEY in .env for macro charts (free key). "
                "Regime is currently computed from market-price proxies.")

    st.subheader("Prediction markets (macro) — regime inputs only")
    st.caption("These markets resolve in weeks–months; they inform the regime tilt, "
               "never direct ETF picks.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Polymarket**")
        for e in pm.get("polymarket", [])[:6]:
            with st.expander(f"{e['title']}  (${e['volume']/1e6:.0f}M vol)"):
                for o in e["outcomes"][:6]:
                    st.write(f"- {o['name']}: **{o['prob']:.0%}**")
    with right:
        st.markdown("**Kalshi**")
        kdf = pd.DataFrame(pm.get("kalshi", []))
        if not kdf.empty:
            kdf = kdf[["series", "title", "subtitle", "prob", "volume"]]
            st.dataframe(kdf.style.format({"prob": "{:.0%}"}),
                         use_container_width=True, hide_index=True, height=400)

with tab_data:
    st.subheader("Every live input feeding the recommendation engine")

    # ---- 1. feed status ----
    fund_latest = (pd.to_datetime(fund["fetched_at"], unit="s").max().strftime("%Y-%m-%d %H:%M")
                   if not fund.empty else "—")
    feeds = pd.DataFrame([
        {"feed": "Yahoo Finance daily prices",
         "coverage": f"{prices.shape[1]} tickers × {prices.shape[0]} days",
         "latest data": str(prices.index.max().date()),
         "cache": f"{age_h:.1f}h old (refreshes >20h)",
         "feeds into": "momentum 30%, 200dma trend gate, regime proxies, beta/TE analytics"},
        {"feed": "yfinance ETF metadata",
         "coverage": f"{len(fund)} candidate ETFs "
                     f"(P/E {fund['trailingPE'].notna().mean():.0%}, "
                     f"expense {fund['netExpenseRatio'].notna().mean():.0%} coverage)",
         "latest data": fund_latest,
         "cache": "≤7 days",
         "feeds into": "valuation 25% (P/E, P/B), quality 20% (expense, AUM, liquidity)"},
        {"feed": "Polymarket (Gamma API, keyless)",
         "coverage": f"{len(pm.get('polymarket', []))} macro events",
         "latest data": pm.get("fetched_at", "—"),
         "cache": "≤15 min",
         "feeds into": "regime 25%: expected Fed cuts, recession odds"},
        {"feed": "Kalshi (trade-api v2, keyless)",
         "coverage": f"{len(pm.get('kalshi', []))} macro markets",
         "latest data": pm.get("fetched_at", "—"),
         "cache": "≤15 min",
         "feeds into": "regime 25%: recession probability (nearest-dated market)"},
        {"feed": "FRED macro series",
         "coverage": (f"{macro.shape[1]} series" if not macro.empty
                      else "not active — no API key, using market proxies"),
         "latest data": (str(macro.index.max().date()) if not macro.empty else "—"),
         "cache": "≤20h",
         "feeds into": "regime 25%: growth (INDPRO) & inflation (breakevens) trends"},
    ])
    st.dataframe(feeds, use_container_width=True, hide_index=True)

    # ---- 2. regime inputs ----
    st.subheader("Regime classifier inputs")
    c = st.columns(5)
    c[0].metric("Growth trend", regime["growth"])
    c[1].metric("Inflation trend", regime["inflation"])
    c[2].metric("PM recession odds",
                fmt_pct(regime["recession_prob"], 0) if regime["recession_prob"] is not None else "—")
    c[3].metric("Expected Fed cuts",
                f"{regime['expected_fed_cuts']:.1f}" if regime["expected_fed_cuts"] is not None else "—")
    c[4].metric("Defensive shift", f"{regime['defensive_shift']:+.2f}")
    st.caption("Sources used: " + "; ".join(regime["inputs_used"]))

    pcol1, pcol2 = st.columns(2)
    if {"XLI", "KXI"}.issubset(prices.columns):
        ratio_g = (prices["XLI"] / prices["KXI"]).dropna().tail(504)
        figg = go.Figure(go.Scatter(x=ratio_g.index, y=ratio_g, name="XLI/KXI"))
        figg.update_layout(title="Growth proxy: XLI/KXI cyclicals vs staples (2y)", height=280)
        pcol1.plotly_chart(figg, use_container_width=True)
    if {"TIP", "IEF"}.issubset(prices.columns):
        ratio_i = (prices["TIP"] / prices["IEF"]).dropna().tail(504)
        figi = go.Figure(go.Scatter(x=ratio_i.index, y=ratio_i, name="TIP/IEF"))
        figi.update_layout(title="Inflation proxy: TIP/IEF breakeven ratio (2y)", height=280)
        pcol2.plotly_chart(figi, use_container_width=True)

    # ---- 3. prediction-market raw feeds ----
    st.subheader("Prediction markets — raw odds")
    pm_rows = [{"event": e["title"], "outcome": o["name"], "prob": o["prob"],
                "volume ($M)": e["volume"] / 1e6}
               for e in pm.get("polymarket", []) for o in e["outcomes"]]
    lcol, rcol = st.columns(2)
    with lcol:
        st.markdown(f"**Polymarket** — {len(pm_rows)} outcomes")
        if pm_rows:
            st.dataframe(pd.DataFrame(pm_rows).style.format(
                {"prob": "{:.0%}", "volume ($M)": "{:,.0f}"}),
                use_container_width=True, hide_index=True, height=350)
    with rcol:
        kdf_raw = pd.DataFrame(pm.get("kalshi", []))
        st.markdown(f"**Kalshi** — {len(kdf_raw)} markets")
        if not kdf_raw.empty:
            st.dataframe(kdf_raw[["series", "title", "prob", "volume", "close_time"]]
                         .style.format({"prob": "{:.0%}", "volume": "{:,.0f}"}),
                         use_container_width=True, hide_index=True, height=350)

    # ---- 4. ETF metadata ----
    st.subheader(f"ETF metadata (valuation & quality inputs) — {len(fund)} candidates")
    meta = fund[["longName", "trailingPE", "priceToBook", "netExpenseRatio",
                 "totalAssets", "averageVolume", "ms_category"]].copy()
    meta["totalAssets ($B)"] = meta.pop("totalAssets") / 1e9
    st.dataframe(meta.sort_values("totalAssets ($B)", ascending=False)
                 .style.format({"trailingPE": "{:.1f}", "priceToBook": "{:.1f}",
                                "netExpenseRatio": "{:.2f}", "totalAssets ($B)": "{:,.1f}",
                                "averageVolume": "{:,.0f}"}, na_rep="—"),
                 use_container_width=True, height=350)

    # ---- 5. price panel ----
    st.subheader("Price panel (momentum inputs)")
    last2 = prices.ffill().iloc[-2:]
    snap = pd.DataFrame({
        "last close": last2.iloc[-1],
        "1d %": last2.iloc[-1] / last2.iloc[-2] - 1,
    }).join(mom_all[["mom_12_1", "mom_6", "above_200dma"]])
    snap.index.name = "ticker"
    show_all = st.checkbox("Show full universe", value=False,
                           help="Unchecked: current portfolio + benchmark only")
    view_px = snap if show_all else snap.loc[snap.index.intersection(
        list(port.index) + [BENCHMARK])]
    st.dataframe(view_px.sort_values("mom_12_1", ascending=False)
                 .style.format({"last close": "{:,.2f}", "1d %": "{:+.2%}",
                                "mom_12_1": "{:+.1%}", "mom_6": "{:+.1%}",
                                "above_200dma": "{:.0f}"}, na_rep="—"),
                 use_container_width=True, height=350)

with tab_bt:
    st.subheader("Momentum-sleeve sanity check (monthly rebalance)")
    st.caption("Only historically computable signals (momentum, trend, bucket structure). "
               "Valuation/regime inputs lack point-in-time history — this validates the "
               "engine, it does not forecast the live composite.")
    if st.button("Run backtest (≈1 min)") or "bt" in st.session_state:
        if "bt" not in st.session_state:
            from app.backtest import run_backtest
            with st.spinner("Running backtest…"):
                st.session_state["bt"] = run_backtest(prices=prices, verbose=False)
        bt = st.session_state["bt"]
        curve = bt["curve"]
        figb = go.Figure()
        for c in curve.columns:
            figb.add_trace(go.Scatter(x=curve.index, y=curve[c], name=c))
        figb.update_layout(title="Growth of $1", height=380)
        st.plotly_chart(figb, use_container_width=True)
        srow = pd.DataFrame({
            "strategy": bt["strategy"], "SPY": bt["spy"],
        }).T
        st.dataframe(srow.style.format({"CAGR": "{:.1%}", "vol": "{:.1%}",
                                        "sharpe": "{:.2f}", "max_drawdown": "{:.1%}"}),
                     use_container_width=True)
        st.caption(f"Tracking error {bt['tracking_error']:.1%} · "
                   f"information ratio {bt['info_ratio']:.2f} · {bt['n_months']} months")

with tab_uni:
    st.subheader("Parsed universe")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total rows", len(uni))
    c2.metric("US-listed", int(uni.listed_us.sum()))
    c3.metric("Core-eligible", int(uni.core_eligible.sum()))
    only_excl = st.checkbox("Show only exclusions")
    view_u = uni[uni.exclusion_reason != ""] if only_excl else uni
    st.dataframe(view_u, use_container_width=True, hide_index=True, height=500)
