#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:23:53 2026

@author: teunwesseling
"""

"""
enrich_finbert_outputs.py
--------------------------
Enriches already-generated FinBERT parquet files for all 11 stocks.

For the ARTICLE-level file:
  - Drops the redundant 'text' column (identical to full_text)

For the DAILY-level file:
  - Drops  : sum_sentiment
  - Adds   : neu_share, std_sentiment, max_neg_prob, max_pos_prob,
             pre_market_count, during_market_count, post_market_count

Run from terminal:
  python enrich_finbert_outputs.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------------------------------------------
# SETTINGS — adjust paths and ticker list as needed
# ------------------------------------------------------------------

import os
BASE       = Path(os.environ.get("THESIS_BASE", Path(__file__).resolve().parent.parent))
INPUT_DIR  = BASE                 # folder with the step-2 FinBERT parquets
OUTPUT_DIR = BASE / "enriched"    # where enriched files go

TICKERS = ["GS", "JPM", "CVX", "XOM", "JNJ", "PFE", "WMT", "KO", "AMZN", "AAPL"]

# Column name patterns — adjust if your files are named differently
ARTICLE_SUFFIX = "_finbert_article.parquet"
DAILY_SUFFIX   = "_finbert_daily.parquet"

# Time column in the article file (assumed UTC)
TIME_COL = "time_utc"

# US Eastern timezone (handles DST automatically)
ET = "America/New_York"
MARKET_OPEN  = pd.Timestamp("09:30:00").time()
MARKET_CLOSE = pd.Timestamp("16:00:00").time()

# ------------------------------------------------------------------


def enrich_article(df: pd.DataFrame) -> pd.DataFrame:
    """Drop redundant text column from article-level dataframe."""
    if "text" in df.columns and "full_text" in df.columns:
        df = df.drop(columns=["text"])
        print("    Dropped column: text")
    return df


def compute_market_session(series_utc: pd.Series) -> pd.Series:
    """
    Given a UTC datetime series, return a string series indicating
    which market session each article belongs to:
      'pre'    : before 09:30 ET
      'during' : 09:30–16:00 ET
      'post'   : after 16:00 ET
    """
    et_times = series_utc.dt.tz_localize("UTC").dt.tz_convert(ET).dt.time
    conditions = [
        et_times < MARKET_OPEN,
        et_times >= MARKET_CLOSE,
    ]
    choices = ["pre", "post"]
    return np.select(conditions, choices, default="during")


def enrich_daily(article_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich the daily dataframe using the article-level dataframe.
    """

    # --- 1. Parse datetime ---
    article_df[TIME_COL] = pd.to_datetime(article_df[TIME_COL])
    article_df["date"]   = article_df[TIME_COL].dt.date

    # --- 2. Market session label ---
    article_df["session"] = compute_market_session(article_df[TIME_COL])

    # --- 3. Compute new daily aggregates from article file ---
    extra = (
        article_df.groupby(["ric", "date"])
        .agg(
            std_sentiment  = ("sentiment_score", "std"),   # disagreement
            max_neg_prob   = ("prob_neg",        "max"),   # worst article of day
            max_pos_prob   = ("prob_pos",        "max"),   # best article of day
            neu_share      = ("prob_neu",        "mean"),  # neutral share
        )
        .reset_index()
    )

    # --- 4. Session counts (pre / during / post market) ---
    session_counts = (
        article_df.groupby(["ric", "date", "session"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={
            "pre":    "pre_market_count",
            "during": "during_market_count",
            "post":   "post_market_count",
        })
        .reset_index()
    )

    # Ensure all three columns exist even if a session had zero articles
    for col in ["pre_market_count", "during_market_count", "post_market_count"]:
        if col not in session_counts.columns:
            session_counts[col] = 0

    # --- 5. Merge into daily ---
    daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date  # align dtype

    daily_df = daily_df.merge(extra,          on=["ric", "date"], how="left")
    daily_df = daily_df.merge(session_counts, on=["ric", "date"], how="left")

    # --- 6. Drop sum_sentiment ---
    if "sum_sentiment" in daily_df.columns:
        daily_df = daily_df.drop(columns=["sum_sentiment"])
        print("    Dropped column: sum_sentiment")

    # --- 7. Fill NaN in std_sentiment for days with only 1 article (std=NaN → 0) ---
    daily_df["std_sentiment"] = daily_df["std_sentiment"].fillna(0)

    # --- 8. Reorder columns cleanly ---
    preferred_order = [
        "ric", "date",
        "news_count", "log_news_count",
        "pre_market_count", "during_market_count", "post_market_count",
        "avg_sentiment", "std_sentiment",
        "pos_share", "neg_share", "neu_share",
        "max_pos_prob", "max_neg_prob",
    ]
    existing = [c for c in preferred_order if c in daily_df.columns]
    remaining = [c for c in daily_df.columns if c not in existing]
    daily_df = daily_df[existing + remaining]

    return daily_df


# ------------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for ticker in TICKERS:
        print(f"\n{'='*50}")
        print(f"Processing: {ticker}")
        print(f"{'='*50}")

        article_path = INPUT_DIR / f"{ticker}{ARTICLE_SUFFIX}"
        daily_path   = INPUT_DIR / f"{ticker}{DAILY_SUFFIX}"

        # --- Check files exist ---
        if not article_path.exists():
            print(f"  WARNING: Article file not found — {article_path}, skipping.")
            continue
        if not daily_path.exists():
            print(f"  WARNING: Daily file not found — {daily_path}, skipping.")
            continue

        # --- Load ---
        print("  Loading files...")
        article_df = pd.read_parquet(article_path)
        daily_df   = pd.read_parquet(daily_path)
        print(f"  Articles: {len(article_df):,} rows | Daily: {len(daily_df):,} rows")

        # --- Enrich article ---
        print("  Enriching article file...")
        article_df = enrich_article(article_df)

        # --- Enrich daily ---
        print("  Enriching daily file...")
        daily_df = enrich_daily(article_df, daily_df)

        # --- Save ---
        out_article = OUTPUT_DIR / f"{ticker}{ARTICLE_SUFFIX}"
        out_daily   = OUTPUT_DIR / f"{ticker}{DAILY_SUFFIX}"

        article_df.to_parquet(out_article, index=False)
        daily_df.to_parquet(out_daily,   index=False)

        print(f"  Saved article → {out_article}")
        print(f"  Saved daily   → {out_daily}")
        print(f"  Daily columns : {list(daily_df.columns)}")

    print(f"\n{'='*50}")
    print("All done. Enriched files saved to:")
    print(f"  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()