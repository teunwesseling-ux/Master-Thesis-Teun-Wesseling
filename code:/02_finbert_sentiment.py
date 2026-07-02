#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_finbert_sentiment.py
=================================================================
Step 2 of the pipeline. Scores every news article with FinBERT
(ProsusAI/finbert) and produces, per stock:

  <TICKER>_finbert_article.parquet   article-level FinBERT probabilities + score
  <TICKER>_finbert_daily.parquet     daily aggregates (count, avg sentiment, ...)

Input per stock is <ticker>.parquet (lower-case), i.e. the renamed full-text
news file from step 1 (articles_<ric>_reuters_full.parquet -> <ticker>.parquet).

Signed sentiment per article:  s = p_pos - p_neg.
Runs on CPU or GPU automatically. Loops over all tickers in one go.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
BASE = Path(os.environ.get("THESIS_BASE",
            Path(__file__).resolve().parent.parent))
MODEL_NAME = "ProsusAI/finbert"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Each entry maps the input <ticker>.parquet (lower) to the output stub (UPPER).
TICKERS = ["GS", "JPM", "CVX", "XOM", "JNJ", "PFE", "WMT", "KO", "AMZN", "AAPL"]


def load_model():
    print("Loading FinBERT model on", DEVICE, "...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, use_safetensors=True, revision="refs/pr/28")
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def score_articles(df, tokenizer, model):
    results = []
    for text in tqdm(df["full_text"], desc="  FinBERT"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        prob_pos, prob_neg, prob_neu = probs[0], probs[1], probs[2]
        results.append([prob_pos, prob_neg, prob_neu, prob_pos - prob_neg])
    return np.array(results)


def process_ticker(ticker, tokenizer, model):
    parquet_file = BASE / f"{ticker.lower()}.parquet"
    out_article = BASE / f"{ticker}_finbert_article.parquet"
    out_daily = BASE / f"{ticker}_finbert_daily.parquet"

    if not parquet_file.exists():
        print(f"  ! {parquet_file.name} not found — skipping {ticker}.")
        return

    df = pd.read_parquet(parquet_file)
    print(f"  articles loaded: {len(df):,}")
    df["time_utc"] = pd.to_datetime(df["time_utc"])
    df["full_text"] = df["headline"].fillna("") + ". " + df["text"].fillna("")

    res = score_articles(df, tokenizer, model)
    df["prob_pos"], df["prob_neg"] = res[:, 0], res[:, 1]
    df["prob_neu"], df["sentiment_score"] = res[:, 2], res[:, 3]

    df.to_parquet(out_article)

    df["date"] = df["time_utc"].dt.date
    daily = (df.groupby(["ric", "date"])
             .agg(news_count=("storyId", "count"),
                  avg_sentiment=("sentiment_score", "mean"),
                  sum_sentiment=("sentiment_score", "sum"),
                  neg_share=("prob_neg", "mean"),
                  pos_share=("prob_pos", "mean"))
             .reset_index())
    daily["log_news_count"] = np.log1p(daily["news_count"])
    daily.to_parquet(out_daily)
    print(f"  saved {out_article.name} + {out_daily.name}")


if __name__ == "__main__":
    tokenizer, model = load_model()
    for ticker in TICKERS:
        print("=" * 50, f"\nFinBERT: {ticker}\n", "=" * 50, sep="")
        process_ticker(ticker, tokenizer, model)
    print("\nDone. Next: run 03_enrich_news.py.")
