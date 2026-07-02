#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_clean_taq.py
=================================================================
Step 4 of the pipeline (independent of steps 1-3). Cleans the raw WRDS TAQ
trade files into one tidy intraday file per stock:

  raw data/<ticker>_raw.csv + <ticker>_raw2.csv   (two sample periods)
        -> clean data/<ticker>_cleaned.csv

Filtering, per chunk (memory-safe):
  - keep only normal trades (TR_SCOND consisting solely of @ E F, and non-empty)
  - keep corrected-trade flag TR_CORR == 0
  - drop zero prices
Then: build a microsecond datetime, floor to the second, and keep the FIRST
trade per (symbol, second). One cleaned CSV per ticker.

>>> ACCESS REQUIRED. The raw TAQ files come from WRDS and are very large; a
    grader without WRDS access cannot reproduce this step. The cleaned files are
    the starting point for the volatility computations in 06_main_models.py.
"""
import os
from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
BASE = Path(os.environ.get("THESIS_BASE",
            Path(__file__).resolve().parent.parent))
RAW_DIR = BASE / "raw data"
CLEAN_DIR = BASE / "clean data"

TICKERS = ["gs", "jpm", "cvx", "xom", "jnj", "pfe", "wmt", "ko", "amzn", "aapl"]
CHUNKSIZE = 2_000_000          # lower to 1_000_000 on low-RAM machines
USE_COLS = ["DATE", "TIME_M", "SYM_ROOT", "TR_SCOND", "PRICE", "TR_CORR"]


def load_and_filter_chunked(path, chunksize):
    """Read in chunks and filter each chunk immediately."""
    parts = []
    reader = pd.read_csv(path, usecols=USE_COLS, chunksize=chunksize, low_memory=False)
    for i, chunk in enumerate(reader, 1):
        s = chunk["TR_SCOND"].fillna("").astype(str)
        keep = s.str.replace(r"[@EF]", "", regex=True).eq("") & s.ne("")
        chunk = chunk[keep & (chunk["TR_CORR"] == 0) & (chunk["PRICE"] != 0)]
        parts.append(chunk)
        print(f"    chunk {i}: {len(chunk):,} rows kept "
              f"(total {sum(len(p) for p in parts):,})")
    return pd.concat(parts, ignore_index=True)


def load_raw_chunked(ticker, chunksize):
    raw1 = RAW_DIR / f"{ticker}_raw.csv"
    raw2 = RAW_DIR / f"{ticker}_raw2.csv"
    dfs = []
    for path in [raw1, raw2]:
        if path.exists():
            print(f"  -> {path.name}")
            dfs.append(load_and_filter_chunked(path, chunksize))
        else:
            print(f"  ! not found (skipped): {path.name}")
    if not dfs:
        raise FileNotFoundError(f"No raw files for {ticker.upper()}.")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  combined rows after filter: {len(combined):,}")
    return combined


def finish_cleaning(df, ticker):
    time_us = df["TIME_M"].astype(str).str.replace(r"(\.\d{6})\d+", r"\1", regex=True)
    df = df.assign(datetime=pd.to_datetime(
        df["DATE"].astype(str) + " " + time_us,
        format="%Y-%m-%d %H:%M:%S.%f", errors="coerce"))
    df = df[df["datetime"].notna()]
    df["sec_floor"] = df["datetime"].dt.floor("s")
    df_cleaned = df.drop_duplicates(subset=["SYM_ROOT", "sec_floor"], keep="first")
    print(f"  rows after dedup: {len(df_cleaned):,}")
    return df_cleaned.reset_index(drop=True)


def clean_one(ticker, chunksize=CHUNKSIZE):
    print("=" * 50, f"\nCleaning: {ticker.upper()}\n", "=" * 50, sep="")
    df_raw = load_raw_chunked(ticker, chunksize)
    df_cleaned = finish_cleaning(df_raw, ticker)
    out = CLEAN_DIR / f"{ticker}_cleaned.csv"
    df_cleaned.to_csv(out, index=False)
    print(f"  saved {len(df_cleaned):,} rows -> {out.name}\n")


if __name__ == "__main__":
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    for tk in TICKERS:
        try:
            clean_one(tk)
        except FileNotFoundError as err:
            print(f"  ! {err}\n")
    print("Done. Next: run 06_main_models.py (and 05_descriptives.py for figures).")
