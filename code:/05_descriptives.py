#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_descriptives.py
=================================================================
Step 5: the four descriptive figures used in the thesis. This script depends
ONLY on the canonical results backbone (06_main_models.py) — it reuses that
file's volatility code so every number matches the estimation sample exactly.

Figures produced, for each measure (RK primary, RV robustness):
  fig_logtransform_<M>.pdf   distribution of RV vs log-RV (justifies the log)
  fig_acf_<M>.pdf            mean ACF of log-vol with HAR lags 1/5/22 marked
  fig_returns_grid_<M>.pdf   daily open-to-close returns, 5x2 grid per stock
  fig_logrv_grid_<M>.pdf     log realized variance, 5x2 grid per stock

Saved to <BASE>/results/figures/ as vector PDFs.

Run:  python 05_descriptives.py      (needs clean data/, like 06 does)
"""
import os
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Load the canonical backbone (06_main_models.py) as a module, reusing its
# paths and volatility functions. Importing it does NOT run the analysis
# (that is guarded by __main__).
# ---------------------------------------------------------------------------
_BACKBONE = Path(__file__).with_name("06_main_models.py")
_spec = importlib.util.spec_from_file_location("thesis_model", _BACKBONE)
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
MEASURES = ["RK", "RV"]          # RK is primary; RV is the robustness measure
SECTOR = {"GS": "Financials", "JPM": "Financials", "CVX": "Energy", "XOM": "Energy",
          "JNJ": "Healthcare", "PFE": "Healthcare", "WMT": "Consumer", "KO": "Consumer",
          "AMZN": "Tech", "AAPL": "Tech"}
GRID = (5, 2)                    # 5 rows x 2 cols = 10 panels
FIG_DIR = T.BASE / "results" / "figures"


def _save(fig, stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{stem}.pdf"
    fig.savefig(path, bbox_inches="tight")
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# PANEL CONSTRUCTION  (rv/lrv from the chosen measure + open-to-close return)
# ---------------------------------------------------------------------------
def _daily_return(df, price_col, ts_col):
    """Open-to-close daily log return, matching the open-to-close RV window."""
    d = df[[ts_col, price_col]].copy()
    d[ts_col] = pd.to_datetime(d[ts_col]); d = d.dropna()
    d = d[d[price_col] > 0]
    d = (d.sort_values(ts_col).set_index(ts_col)
         .between_time(T.SESSION[0], T.SESSION[1]))
    out = []
    for day, g in d.groupby(d.index.normalize()):
        p = g[price_col]; med = p.median()
        p = p[(p > med * (1 - T.PRICE_DEV)) & (p < med * (1 + T.PRICE_DEV))]
        if len(p) < 2:
            continue
        out.append((day.normalize(), float(np.log(p.iloc[-1]) - np.log(p.iloc[0]))))
    return pd.DataFrame(out, columns=["date", "ret"])


def _vol_series(df, price_col, ts_col, measure):
    """Daily volatility series named 'rv' (RV via 5-min, or RK per day)."""
    if measure.upper() == "RV":
        return T.compute_rv_from_intraday(df, price_col, ts_col)
    # realized kernel, one value per day
    d = df[[ts_col, price_col]].copy()
    d[ts_col] = pd.to_datetime(d[ts_col]); d = d.dropna()
    d = d[d[price_col] > 0]
    d = (d.sort_values(ts_col).set_index(ts_col)
         .between_time(T.SESSION[0], T.SESSION[1]))
    rows = []
    for day, g in d.groupby(d.index.normalize()):
        val = T.realized_kernel_day(g[price_col])
        if np.isfinite(val) and val > 0:
            rows.append((day.normalize(), val))
    return pd.DataFrame(rows, columns=["date", "rv"])


def build_panel(measure):
    frames = []
    for tk in T.TICKERS:
        df = T._read_any(T.CLEAN_DIR / f"{tk.lower()}{T.CLEAN_SUFFIX}")
        ts = T._first_present(df.columns, T.TS_COL_CANDIDATES)
        px = T._first_present(df.columns, T.PRICE_COL_CANDIDATES)
        if ts is None or px is None:
            raise ValueError(f"[{tk}] need (timestamp, price); got {list(df.columns)}")
        vol = _vol_series(df, px, ts, measure).sort_values("date")
        vol["rv"] = vol["rv"].clip(lower=T.RV_FLOOR)
        vol["lrv"] = np.log(vol["rv"])
        ret = _daily_return(df, px, ts)
        s = vol.merge(ret, on="date", how="left")
        s["ticker"] = tk
        s["sector"] = SECTOR.get(tk, "")
        frames.append(s)
    panel = pd.concat(frames, ignore_index=True)
    return panel.dropna(subset=["lrv"]).sort_values(["ticker", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# FIGURES
# ---------------------------------------------------------------------------
def plot_logtransform(panel, measure):
    """Histogram of RV vs log-RV — justifies the log transform."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(panel["rv"], bins=80, color="#4C72B0", edgecolor="white")
    ax[0].set_title(f"Realized variance (levels, {measure})"); ax[0].set_xlabel(measure)
    ax[1].hist(panel["lrv"], bins=80, color="#55A868", edgecolor="white")
    ax[1].set_title(f"log realized variance ({measure})"); ax[1].set_xlabel(f"log {measure}")
    fig.suptitle("Distribution: RV vs log-RV (justifies the log transform)")
    fig.tight_layout()
    _save(fig, f"fig_logtransform_{measure}")
    plt.close(fig)


def plot_acf(panel, measure, nlags=66):
    """Mean ACF of log-vol across stocks, HAR lags 1/5/22 highlighted."""
    acfs = [acf(s["lrv"].values, nlags=nlags, fft=True)
            for _, s in panel.groupby("ticker")]
    m = np.mean(acfs, axis=0)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(m)), m, color="#BBBBBB", width=0.85)
    for lag, c in [(1, "#C44E52"), (5, "#4C72B0"), (22, "#55A868")]:
        ax.bar(lag, m[lag], color=c, width=0.85, label=f"HAR lag {lag}")
    ci = 1.96 / np.sqrt(panel.groupby("ticker").size().mean())
    ax.axhline(ci, ls="--", c="k", lw=0.7); ax.axhline(-ci, ls="--", c="k", lw=0.7)
    ax.set_title(f"Mean ACF of log-RV ({measure}) — justifies HAR daily/weekly/monthly")
    ax.set_xlabel("lag (trading days)"); ax.set_ylabel("autocorrelation")
    ax.legend(); fig.tight_layout()
    _save(fig, f"fig_acf_{measure}")
    plt.close(fig)


def plot_returns_grid(panel, measure):
    """Daily open-to-close returns per stock (5x2)."""
    fig, axes = plt.subplots(*GRID, figsize=(11, 12), sharex=True)
    for ax, tk in zip(axes.ravel(), list(T.TICKERS)):
        s = panel[panel["ticker"] == tk].sort_values("date")
        ax.plot(s["date"], 100 * s["ret"], lw=0.6, color="#4C72B0")
        ax.axhline(0, lw=0.5, color="k", alpha=0.4)
        ax.set_title(f"{tk} ({SECTOR.get(tk, '')})", fontsize=9)
        ax.set_ylabel("ret (%)", fontsize=8); ax.tick_params(labelsize=7)
    fig.suptitle("Daily open-to-close returns by stock", y=0.995)
    fig.tight_layout()
    _save(fig, f"fig_returns_grid_{measure}")
    plt.close(fig)


def plot_logrv_grid(panel, measure):
    """log realized variance time series per stock (5x2)."""
    fig, axes = plt.subplots(*GRID, figsize=(11, 12), sharex=True, sharey=True)
    for ax, tk in zip(axes.ravel(), list(T.TICKERS)):
        s = panel[panel["ticker"] == tk].sort_values("date")
        ax.plot(s["date"], s["lrv"], lw=0.7, color="#55A868")
        ax.set_title(f"{tk} ({SECTOR.get(tk, '')})  mean={s['lrv'].mean():.2f}", fontsize=9)
        ax.set_ylabel("log-RV", fontsize=8); ax.tick_params(labelsize=7)
    fig.suptitle(f"log realized variance by stock ({measure})", y=0.995)
    fig.tight_layout()
    _save(fig, f"fig_logrv_grid_{measure}")
    plt.close(fig)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for measure in MEASURES:
        print("=" * 60, f"\nDESCRIPTIVE FIGURES — measure = {measure}\n", "=" * 60, sep="")
        panel = build_panel(measure)
        print(f"{panel['ticker'].nunique()} stocks, {panel['date'].nunique()} dates, "
              f"{len(panel)} stock-days")
        plot_logtransform(panel, measure)
        plot_acf(panel, measure)
        plot_returns_grid(panel, measure)
        plot_logrv_grid(panel, measure)
    print(f"\nDone. Figures written to {FIG_DIR}")
