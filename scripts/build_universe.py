"""Parse a source ETF list (two-column ticker/name PDF) into a tagged universe CSV.

Usage: python scripts/build_universe.py [path-to-pdf]

Set the source list via the argument, or the ETF_SOURCE_LIST env var, or drop a
file named etf_source_list.pdf in the project root.

Output: data/universe.csv with columns
    ticker, name, category, listed_us, core_eligible, exclusion_reason
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd
import pdfplumber

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = PROJECT_ROOT / "data" / "universe.csv"


def resolve_source() -> Path:
    """Locate the source ETF list: CLI arg, env var, generic name, or lone root PDF."""
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    env = os.getenv("ETF_SOURCE_LIST")
    if env:
        return Path(env)
    generic = PROJECT_ROOT / "etf_source_list.pdf"
    if generic.exists():
        return generic
    root_pdfs = list(PROJECT_ROOT.glob("*.pdf"))
    if len(root_pdfs) == 1:
        return root_pdfs[0]
    raise SystemExit(
        "No source list found. Pass a path, set ETF_SOURCE_LIST, or place "
        "etf_source_list.pdf in the project root."
    )

# x-position that separates the ticker column from the name column
TICKER_COL_MAX_X = 130

US_TICKER_RE = re.compile(r"^[A-Z]{1,5}$")

# short tickers that look US but are foreign exchange lines (verified: no US quote)
KNOWN_NON_US = {"AHYG", "AJAC", "IFFF", "IGWD", "IMEU", "IWDE", "VDU", "VUSA", "XAXD"}

# Names that indicate the row is not a US-listed line we can trade via a US broker
NON_US_NAME_PAT = re.compile(
    r"UCITS|Hang Seng|Nomura|Nippon India|FIC FI|Commercial Paper|SSE Pledge"
    r"|\(Australia\)|US Treasury 5 Year Note$",
    re.IGNORECASE,
)

# Option-overlay / defined-outcome structures: structurally capped upside, cannot
# beat SPY over 2-5y by construction. Kept in the screener, excluded from core.
OPTION_OVERLAY_PAT = re.compile(
    r"BuyWrite|Covered Call|Premium Income|Premium Yield|Target Income"
    r"|Option Income|Overlay|Hedged Equity|Managed Floor|Collared|Buffer"
    r"|Equity Premium|Option ETF|Hedged Eqty",
    re.IGNORECASE,
)

CATEGORY_RULES: list[tuple[str, re.Pattern]] = [
    # bonds first: many bond fund names also contain words like "value"/"income"
    ("bond_muni", re.compile(r"Muni|Municipal|Tax-Exempt|Tax Exempt|Tax Aware|Tax-Aware", re.I)),
    ("bond_treasury", re.compile(r"Treasury|T-Bill|Govt Bond|Government Bond|Zero Coupon", re.I)),
    ("bond_tips", re.compile(r"TIPS|Inflation-Protected|Inflation Protected", re.I)),
    ("bond_hy", re.compile(r"High Yield|Fallen Angel|BB-|CCC|B-BBB|BBB-B", re.I)),
    ("bond_clo_loan", re.compile(r"CLO|Senior Loan|Floating Rate|Bank Loan|Loan ETF", re.I)),
    ("bond_ig", re.compile(
        r"\bBond\b|Aggregate|Corporate|Credit|MBS|Mortgage|Securitized|Fixed Income"
        r"|Duration|Short Maturity|Ultra[- ]?Short|Income ETF|Total Return|Core Plus"
        r"|GNMA|CMBS|Convertible|Sukuk|Govt|iBonds|BulletShares", re.I)),
    ("preferred", re.compile(r"Preferred", re.I)),
    ("commodity", re.compile(
        r"Commodity|Gold|Silver|Copper|Natural Resources|Miners|Upstream|GSCI", re.I)),
    ("real_estate", re.compile(r"REIT|Real Estate", re.I)),
    # emerging markets before intl (many names contain both)
    ("em_equity", re.compile(
        r"Emerging|China|India|Brazil|Thailand|Turkey|A[- ]?Shares|Golden Dragon"
        r"|Asia ex[- ]?Japan|Frontier|Nifty|CSI \d", re.I)),
    ("intl_equity", re.compile(
        r"International|EAFE|Developed|Europe|European|Japan|Pacific|Canada"
        r"|United Kingdom|FTSE 250|Eurozone|EURO STOXX|STOXX|ex[- ]?US|ex U\.S\."
        r"|Overseas|World|Global|ACWI|Nikkei|JPX|France|Germany|Israel|Far East"
        r"|Foreign|Korea|Australia", re.I)),
    ("sector_thematic", re.compile(
        r"Technology|Tech |Tech-|Software|Semiconductor|Artificial Intelligence"
        r"|\bAI\b|Robotics|FinTech|Internet|Retail|Bank|Community Bank|Insurance"
        r"|Water|Climate|Carbon|Clean|Solar|Infrastructure|Industrial|Consumer"
        r"|Staples|Health|Biotech|Quantum|Electric Vehicles|Self-Driving|EV and"
        r"|Online|Data Sharing|Disruptors|Innovat|New Economies|Financial|Energy"
        r"|Private Equity|Merger|Crypto|Blockchain|Transformational", re.I)),
    ("us_small_mid", re.compile(
        r"Small|SMID|Mid[- ]?Cap|Micro[- ]?Cap|Russell 2000|S&P 600|S&P 400"
        r"|Extended Market|Next 500|Mid Cap", re.I)),
    ("us_factor", re.compile(
        r"Value|Growth|Momentum|Quality|Dividend|Low Vol|Min Vol|Minimum Vol"
        r"|Equal Weight|Factor|Multifactor|Multi-Factor|Buyback|BuyBack|Moat"
        r"|Cash Cows|Free Cash Flow|Shareholder Yield|GARP|Revenue|Pure"
        r"|High Beta|Aristocrat|Achievers|Capital Strength|AlphaDEX|Earnings", re.I)),
    ("us_core", re.compile(
        r"S&P 500|Total Stock Market|Russell [13]000|Large[- ]?Cap|Broad Market"
        r"|US Equity|U\.S\. Equity|1000 Index|Nasdaq[- ]?100|NASDAQ-100|QQQ"
        r"|Total US|Core Equity|S&P 100|Top 200|1500|Transform 500|500 ETF"
        r"|Strive 500|Mega Cap|Dow Jones U\.S\.", re.I)),
]


def classify(name: str) -> str:
    for category, pat in CATEGORY_RULES:
        if pat.search(name):
            return category
    return "us_equity_other"


def parse_pdf(pdf_path: Path) -> list[tuple[str, str]]:
    """Extract (ticker_cell, name) rows from the two-column table pages."""
    rows: list[tuple[str, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[1:]:  # page 1 is guidelines text
            words = page.extract_words()
            lines: dict[int, list] = {}
            for w in words:
                key = int(round(w["top"] / 3))  # group words into lines by y
                lines.setdefault(key, []).append(w)
            for _, ws in sorted(lines.items()):
                ws.sort(key=lambda w: w["x0"])
                ticker_cell = " ".join(w["text"] for w in ws if w["x0"] < TICKER_COL_MAX_X)
                name = " ".join(w["text"] for w in ws if w["x0"] >= TICKER_COL_MAX_X)
                if not ticker_cell or not name or ticker_cell.lower() == "ticker":
                    continue
                rows.append((ticker_cell.strip(), name.strip()))
    return rows


def build(pdf_path: Path) -> pd.DataFrame:
    records = []
    for ticker_cell, name in parse_pdf(pdf_path):
        # rows like "EMB, IEMB, IUS7, S" list dual listings; first is the US line
        primary = ticker_cell.split(",")[0].strip()
        # rows like "VB U" / "PY U" are truncated "<ticker> US" suffixes
        primary = primary.split()[0] if primary else primary

        listed_us = (bool(US_TICKER_RE.match(primary))
                     and primary not in KNOWN_NON_US
                     and not NON_US_NAME_PAT.search(name))
        option_overlay = bool(OPTION_OVERLAY_PAT.search(name))
        category = "option_overlay" if option_overlay else classify(name)

        reason = ""
        if not listed_us:
            reason = "non-US listing / not tradable on US exchange"
        elif option_overlay:
            reason = "option-overlay structure caps upside vs SPY"

        records.append({
            "ticker": primary,
            "name": name,
            "category": category,
            "listed_us": listed_us,
            "core_eligible": listed_us and not option_overlay,
            "exclusion_reason": reason,
        })

    # a ticker can appear both as a UCITS line and a US line: keep the US one
    df = pd.DataFrame(records)
    df = df.sort_values(["ticker", "listed_us"], ascending=[True, False])
    df = df.drop_duplicates(subset="ticker", keep="first").reset_index(drop=True)
    return df


def main() -> None:
    df = build(resolve_source())
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df)} tickers to {OUT_CSV}")
    print(f"  US-listed: {df.listed_us.sum()}  core-eligible: {df.core_eligible.sum()}")
    print(df.category.value_counts().to_string())


if __name__ == "__main__":
    main()
