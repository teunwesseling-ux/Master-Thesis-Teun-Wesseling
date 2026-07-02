#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_export_news.py
=================================================================
Step 1 of the pipeline. Exports Reuters (RTRS) news for every stock from the
LSEG / Refinitiv workstation, one RIC at a time, and saves two parquet files
per stock:

  articles_<ric>_reuters_full.parquet   full-text articles (kept if >= 50 chars)
  headlines_<ric>_reuters.parquet       the full headline universe (no text)

For each month it pulls the headlines, stores the headline universe, then
fetches and cleans the body text (HTML stripped) for the de-duplicated stories.

>>> ACCESS REQUIRED. This step only runs on a machine with an authenticated
    LSEG session (`lseg.data`). A grader without LSEG access CANNOT reproduce
    this step; the resulting parquet files are provided as the starting point
    for step 2. See README.

After this step you MANUALLY rename each stock's full-text file
`articles_<ric>_reuters_full.parquet` to `<ticker>.parquet` (e.g.
`articles_xom_n_reuters_full.parquet` -> `xom.parquet`) before running
step 2 (02_finbert_sentiment.py).
"""
from datetime import datetime
import time
import pandas as pd
import lseg.data as ld
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
# RICs for the ten stocks used in the thesis. Nasdaq names end in .O, NYSE .N.
# Verify these against your LSEG instrument lookup if a query returns nothing.
RICS = [
    "GS.N", "JPM.N", "CVX.N", "XOM.N", "JNJ.N",
    "PFE.N", "WMT.N", "KO.N", "AMZN.O", "AAPL.O",
]

START = "2024-12-01"
END = datetime.today().strftime("%Y-%m-%d")
MIN_TEXT_LEN = 50          # drop bodies shorter than this many characters
HEADLINES_PER_MONTH = 1000  # LSEG page size; 1000 hits => consider weekly batching


def export_one_ric(ric: str) -> None:
    query = f"R:{ric} AND Language:LEN AND Source:RTRS"
    months = pd.date_range(START, END, freq="MS")

    all_rows = []           # full-text kept articles
    all_headlines_rows = []  # ALL headlines (universe)
    seen_text = set()
    seen_headlines = set()

    for s in months:
        e = s + pd.offsets.MonthBegin(1)
        print(f"  [{ric}] headlines {s.date()} -> {e.date()}")
        headlines = ld.news.get_headlines(
            query=query, start=s.to_pydatetime(), end=e.to_pydatetime(),
            count=HEADLINES_PER_MONTH,
        )
        if len(headlines) == 0:
            continue

        # normalise the timestamp into a 'time_utc' column
        if headlines.index.name == "versionCreated":
            headlines = headlines.reset_index().rename(columns={"versionCreated": "time_utc"})
        elif "versionCreated" in headlines.columns:
            headlines = headlines.rename(columns={"versionCreated": "time_utc"})
        else:
            headlines = headlines.reset_index().rename(columns={"index": "time_utc"})
        headlines["time_utc"] = pd.to_datetime(headlines["time_utc"], errors="coerce")

        if len(headlines) >= HEADLINES_PER_MONTH:
            print(f"  ! WARNING: {HEADLINES_PER_MONTH} headlines this month; "
                  f"consider weekly batching for {s.date()}.")

        # 1) headline universe (no text needed)
        for _, r in headlines.iterrows():
            sid = r["storyId"]
            if sid in seen_headlines:
                continue
            seen_headlines.add(sid)
            t = r["time_utc"]
            all_headlines_rows.append({
                "ric": ric, "time_utc": t,
                "date": t.date() if pd.notnull(t) else None,
                "headline": r.get("headline", None),
                "storyId": sid, "sourceCode": r.get("sourceCode", None),
            })

        # 2) fetch + clean body text (de-duplicated)
        for _, r in headlines.iterrows():
            sid = r["storyId"]
            if sid in seen_text:
                continue
            seen_text.add(sid)
            try:
                try:
                    txt = ld.news.get_story(sid, format="text")
                except TypeError:
                    txt = ld.news.get_story(sid)
                if isinstance(txt, str) and "<" in txt and ">" in txt:
                    txt = BeautifulSoup(txt, "html.parser").get_text("\n")
                if txt is None or len(str(txt).strip()) < MIN_TEXT_LEN:
                    continue
                t = r["time_utc"]
                all_rows.append({
                    "ric": ric, "time_utc": t,
                    "date": t.date() if pd.notnull(t) else None,
                    "headline": r.get("headline", None),
                    "storyId": sid, "sourceCode": r.get("sourceCode", None),
                    "text": txt,
                })
                time.sleep(0.05)
            except Exception:
                continue

    headlines_df = (pd.DataFrame(all_headlines_rows)
                    .drop_duplicates(subset="storyId").sort_values("time_utc"))
    articles_df = (pd.DataFrame(all_rows)
                   .drop_duplicates(subset="storyId").sort_values("time_utc"))

    print(f"  [{ric}] headline universe : {len(headlines_df):,}")
    print(f"  [{ric}] full-text kept    : {len(articles_df):,}")

    tag = ric.replace(".", "_").lower()
    headlines_out = f"headlines_{tag}_reuters.parquet"
    articles_out = f"articles_{tag}_reuters_full.parquet"
    headlines_df.to_parquet(headlines_out, index=False)
    articles_df.to_parquet(articles_out, index=False)
    print(f"  [{ric}] saved {headlines_out} + {articles_out}")


if __name__ == "__main__":
    ld.open_session()
    try:
        for ric in RICS:
            print("=" * 60, f"\nEXPORTING NEWS: {ric}\n", "=" * 60, sep="")
            export_one_ric(ric)
    finally:
        ld.close_session()
    print("\nDone. Next: rename each articles_<ric>_reuters_full.parquet to "
          "<ticker>.parquet, then run 02_finbert_sentiment.py.")
