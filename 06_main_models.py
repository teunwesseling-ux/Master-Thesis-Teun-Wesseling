"""
@author: teunwesseling
"""

"""
News analytics & volatility forecasting — full results script.
=================================================================

Multivariate HAR system over 10 stocks. The dependent variable
log(RK) (realized kernel) is the PRIMARY measure; log(RV) (5-min realized
variance) is run in parallel as a robustness measure. Every table is produced
twice, once per measure, with RK as the primary column.

Estimation : OLS on the stacked stock-day data with stock-specific intercepts
             (10 dummies) and slopes shared across all series.
Std errors : Driscoll-Kraay HAC (aggregate the regression scores per day over
             the cross-section, then a Newey-West Bartlett kernel with
             truncation lag  m = floor(4 * (T/100)^(2/9))).

Pipeline (run top to bottom):
  1. RV (5-min) and RK (realized kernel) per ticker from cleaned intraday data
  2. news variables from the per-article FinBERT file (overnight + full-day placebo)
  3. log-HAR terms (lags 1, 5, 22) for both measures; assemble the stacked panel
  4. in-sample model grid M0-M15 (+ placebo twins): LR test, AIC, BIC, R^2
  5. out-of-sample expanding-window forecasts: MSE-log, QLIKE, DM* vs M0 (HLN)
  6. Model Confidence Set (Hansen-Lunde-Nason) over M0-M15 under both losses
  7. write every result to CSV + LaTeX, one set for RK and one for RV

"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, chi2, t as student_t
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
BASE = Path(os.environ.get("THESIS_BASE",
            Path(__file__).resolve().parent.parent))   # default: the folder that contains code/
CLEAN_DIR = BASE / "clean data"          # cleaned intraday: <lowerticker>_cleaned.{csv,parquet}
NEWS_DIR = BASE / "enriched"             # per-article FinBERT: TICKER_finbert_article.{parquet,csv}
OUT_DIR = BASE / "results"               # tables are written here
OUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["GS", "JPM", "CVX", "XOM", "JNJ", "PFE", "WMT", "KO", "AMZN", "AAPL"]

# file naming -----------------------------------------------------------
CLEAN_SUFFIX = "_cleaned"                 # cleaned intraday file stub: f"{ticker.lower()}{CLEAN_SUFFIX}"
ARTICLE_SUFFIX = "_finbert_article"       # per-article file stub:      f"{ticker}{ARTICLE_SUFFIX}"

# intraday column auto-detection
TS_COL_CANDIDATES = ["datetime", "timestamp", "ts", "DT", "TIME_M"]
PRICE_COL_CANDIDATES = ["PRICE", "price", "trade_price", "mid", "close"]

# per-article FinBERT columns
ARTICLE_DATE_COL = "date"        # calendar date/time of the article (from the enrich step)
ARTICLE_SESSION_COL = "session"  # 'pre'/'post'/'during' market label (from the enrich step)
PROB_POS_COL = "prob_pos"        # FinBERT positive probability p_pos
PROB_NEG_COL = "prob_neg"        # FinBERT negative probability p_neg

ARTICLE_DATE_CANDIDATES = ["date", "datetime", "timestamp", "published", "published_at", "time", "DATE"]
ARTICLE_SESSION_CANDIDATES = ["session", "window", "market_session", "period", "session_window"]
PROB_POS_CANDIDATES = ["prob_pos", "p_pos", "pos_prob", "positive", "prob_positive",
                       "finbert_pos", "positive_prob", "score_pos", "pos"]
PROB_NEG_CANDIDATES = ["prob_neg", "p_neg", "neg_prob", "negative", "prob_negative",
                       "finbert_neg", "negative_prob", "score_neg", "neg"]


# estimation / forecasting parameters 
RV_SAMPLING = "5min"
SESSION = ("09:30", "16:00")
RET_CAP = 0.10            # |5-min log return| cap (bad-tick guard, RV only)
PRICE_DEV = 0.20         # drop prints > 20% away from the daily median
RV_FLOOR = 1e-12
WEEK, MONTH = 5, 22
INITIAL_TRAIN_FRAC = 0.60
EXTREME_THR = 0.5        # |s| > 0.5 counts as an "extreme" article (c_s)

# ---- realized-kernel parameters (BNHLS 2009, Parzen, end-point jittering) ---
RK_Q = 25                # subsample offsets for omega^2 (noise variance)
RK_IV_FREQ = "20min"     # sparse sampling for the IV proxy (bandwidth selection)
RK_JITTER = 2            # m: prices averaged at each end (jittering)
RK_HMIN, RK_HCAP, RK_HDEF = 1, 150, 60   # bandwidth clamp and fallback

# ---- Model Confidence Set ---------------------------------------------------
MCS_B = 1000             # bootstrap replications
MCS_BLOCK = None         # block length; None -> round(T_oos ** (1/3))
MCS_ALPHA = 0.10         # models with MCS p-value < alpha are eliminated
MCS_SEED = 0
RUN_MCS = True           # the slow part; set False to skip while iterating


# =============================================================================
# IO HELPERS
# =============================================================================
def _read_any(stub: Path) -> pd.DataFrame:
    if stub.exists() and stub.suffix:
        return pd.read_parquet(stub) if stub.suffix == ".parquet" else pd.read_csv(stub)
    for ext in (".parquet", ".csv"):
        p = stub.with_suffix(ext)
        if p.exists():
            return pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
    raise FileNotFoundError(f"No parquet/csv for {stub}")


def _first_present(cols, candidates, override=None):
    if override is not None:
        return override
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
        if c.lower() in low:
            return low[c.lower()]
    return None


# =============================================================================
# REALIZED VARIANCE (5-min)
# =============================================================================
def compute_rv_from_intraday(df, price_col, ts_col, freq=RV_SAMPLING,
                             session=SESSION, ret_cap=RET_CAP, price_dev=PRICE_DEV):
    d = df[[ts_col, price_col]].copy()
    d[ts_col] = pd.to_datetime(d[ts_col])
    d = d.dropna()
    d = d[d[price_col] > 0]
    d = d.sort_values(ts_col).set_index(ts_col).between_time(session[0], session[1])
    out = []
    for day, g in d.groupby(d.index.normalize()):
        p = g[price_col]
        med = p.median()
        p = p[(p > med * (1 - price_dev)) & (p < med * (1 + price_dev))]
        if len(p) < 10:
            continue
        grid = p.resample(freq).last().ffill().dropna()
        if len(grid) < 3:
            continue
        ret = np.log(grid).diff().dropna()
        ret = ret[ret.abs() <= ret_cap]
        if len(ret) < 3:
            continue
        out.append((day.normalize(), float((ret ** 2).sum())))
    return pd.DataFrame(out, columns=["date", "rv"])


# =============================================================================
# REALIZED KERNEL (Barndorff-Nielsen, Hansen, Lunde & Shephard 2009)
# =============================================================================
def _parzen_weights(H):
    h = np.arange(1, H + 1)
    x = h / (H + 1.0)
    return np.where(x <= 0.5, 1 - 6 * x ** 2 + 6 * x ** 3,
                    np.where(x <= 1.0, 2 * (1 - x) ** 3, 0.0))


def _omega2_hat(r, q=RK_Q):
    """Noise variance via q subsample offsets; drop zero returns (price unchanged)."""
    r = r[r != 0.0]
    if r.size < q + 5:
        return np.nan
    vals = []
    for off in range(int(q)):
        sub = r[off::int(q)]
        if sub.size >= 5:
            vals.append(float(sub @ sub) / (2.0 * sub.size))
    return float(np.mean(vals)) if vals else np.nan


def _iv_sparse(logp_series, freq=RK_IV_FREQ):
    """Sparse IV proxy (only for bandwidth selection)."""
    lp = logp_series.resample(freq).last().dropna()
    r = lp.diff().dropna().to_numpy()
    return float(r @ r) if r.size > 1 else np.nan


def _H_star(n, omega2, IV):
    if not (np.isfinite(omega2) and np.isfinite(IV)) or omega2 <= 0 or IV <= 0 or n <= 0:
        return np.nan
    xi = np.sqrt(omega2 / IV)
    return 3.5134 * (xi ** 0.8) * (n ** 0.6)      # c* = 3.5134 (Parzen)


def _jitter(lp, m=RK_JITTER):
    """End-point jittering: average of the first m and last m log prices."""
    n = len(lp)
    if n < 2 * m + 2:
        return lp
    return np.concatenate([[lp[:m].mean()], lp[m:n - m], [lp[-m:].mean()]])


def realized_kernel_day(price_series, price_dev=PRICE_DEV, q=RK_Q,
                        iv_freq=RK_IV_FREQ, m=RK_JITTER,
                        h_min=RK_HMIN, h_cap=RK_HCAP, default_h=RK_HDEF):
    """RK = gamma0 + 2*sum_h k(h/(H+1)) gamma_h, Parzen, on jittered tick returns."""
    p = price_series
    med = p.median()
    p = p[(p > med * (1 - price_dev)) & (p < med * (1 + price_dev))]
    if len(p) < 30:
        return np.nan
    logp = np.log(p)
    r_dense = np.diff(logp.to_numpy())
    omega2 = _omega2_hat(r_dense, q)
    IV = _iv_sparse(logp, iv_freq)
    H_raw = _H_star(r_dense.size, omega2, IV)
    H = default_h if not np.isfinite(H_raw) else int(np.ceil(H_raw))
    H = min(max(H, h_min), h_cap)
    r = np.diff(_jitter(logp.to_numpy(), m))      # jittered returns
    if r.size < 10:
        return np.nan
    H = min(H, r.size - 1)
    w = _parzen_weights(H)
    rk = float(r @ r)                             # gamma0
    for h in range(1, H + 1):
        rk += 2.0 * w[h - 1] * float(r[h:] @ r[:-h])
    return max(rk, 0.0)


def load_both_measures(ticker):
    """Read the cleaned intraday file once; return date, rv (5-min) and rk (kernel)."""
    df = _read_any(CLEAN_DIR / f"{ticker.lower()}{CLEAN_SUFFIX}")
    ts = _first_present(df.columns, TS_COL_CANDIDATES)
    px = _first_present(df.columns, PRICE_COL_CANDIDATES)
    if ts is None or px is None:
        raise ValueError(f"[{ticker}] need (timestamp, price) columns; got {list(df.columns)}")

    rv = compute_rv_from_intraday(df, px, ts)

    d = df[[ts, px]].copy()
    d[ts] = pd.to_datetime(d[ts])
    d = d.dropna()
    d = d[d[px] > 0]
    d = d.sort_values(ts).set_index(ts).between_time(SESSION[0], SESSION[1])
    rk_rows = []
    for day, g in d.groupby(d.index.normalize()):
        val = realized_kernel_day(g[px])
        if np.isfinite(val) and val > 0:
            rk_rows.append((day.normalize(), val))
    rk = pd.DataFrame(rk_rows, columns=["date", "rk"])

    out = rv.merge(rk, on="date", how="outer").sort_values("date").reset_index(drop=True)
    return out


# =============================================================================
# NEWS VARIABLES  (from the per-article FinBERT file)
# =============================================================================
# Signed score per article:  s = p_pos - p_neg.
# Overnight window for trading day t = post-close articles of t-1 + pre-open of t.
# Full-day placebo window for day t = all articles dated on the previous trading day.
def _detect_article_cols(cols):
    dcol = _first_present(cols, ARTICLE_DATE_CANDIDATES, ARTICLE_DATE_COL)
    scol = _first_present(cols, ARTICLE_SESSION_CANDIDATES, ARTICLE_SESSION_COL)
    pcol = _first_present(cols, PROB_POS_CANDIDATES, PROB_POS_COL)
    ncol = _first_present(cols, PROB_NEG_CANDIDATES, PROB_NEG_COL)
    miss = [n for n, c in [("date", dcol), ("session", scol),
                           ("p_pos", pcol), ("p_neg", ncol)] if c is None]
    if miss:
        raise ValueError(
            f"Per-article file missing columns for {miss}. "
            f"Set ARTICLE_DATE_COL / ARTICLE_SESSION_COL / PROB_POS_COL / PROB_NEG_COL "
            f"at the top of the script. Available columns: {list(cols)}")
    return dcol, scol, pcol, ncol


def _normalize_session(series):
    """Map free-text session labels onto 'pre' / 'post' / 'during'."""
    s = series.astype(str).str.lower()
    out = np.where(s.str.contains("pre"), "pre",
          np.where(s.str.contains("post") | s.str.contains("after"), "post", "during"))
    return out


def assign_overnight_day(art, trading_days):
    """pre -> open of that same trading day; post -> the next trading day."""
    td = pd.DatetimeIndex(sorted(pd.unique(pd.to_datetime(trading_days)))).normalize()
    a = art[art["session_norm"].isin(["pre", "post"])].copy()
    a["d"] = pd.to_datetime(a["adate"]).dt.normalize()
    dv = a["d"].values.astype("datetime64[ns]")
    i_pre = td.searchsorted(dv, side="left")
    i_post = td.searchsorted(dv, side="right")
    idx = np.where(a["session_norm"].eq("post").values, i_post, i_pre)
    on = np.full(len(a), np.datetime64("NaT"), "datetime64[ns]")
    ok = idx < len(td)
    on[ok] = td[idx[ok]].values
    a["overnight_day"] = on
    return a.dropna(subset=["overnight_day"])


def _tone_aggregates(g):
    """Compute the spec's tone block from a group of articles (one window)."""
    s = (g["p_pos"] - g["p_neg"]).to_numpy()
    n = len(s)
    return pd.Series({
        "count":  n,
        "m_sent": s.mean(),                                   # direction
        "P_pos":  g["p_pos"].mean(),                          # asymmetry (+)
        "P_neg":  g["p_neg"].mean(),                          # asymmetry (-)
        "a_sent": np.abs(s).mean(),                           # strength
        "d_sent": s.std(ddof=1) if n > 1 else 0.0,            # disagreement (across articles)
        "G":      (2.0 * np.minimum(g["p_pos"], g["p_neg"])).mean(),   # within-article conflict
        "c_n":    int(np.sum(np.abs(s) > EXTREME_THR)),       # extreme count (-> c_s = log1p)
    })


def build_news_vars(ticker, trading_days):
    """Return a per-trading-day frame with overnight + full-day-placebo news vars."""
    art = _read_any(NEWS_DIR / f"{ticker}{ARTICLE_SUFFIX}")
    dcol, scol, pcol, ncol = _detect_article_cols(art.columns)
    art = art.rename(columns={dcol: "adate", pcol: "p_pos", ncol: "p_neg"})
    art["session_norm"] = _normalize_session(art[scol])
    art = art.dropna(subset=["p_pos", "p_neg"])

    td = pd.DatetimeIndex(sorted(pd.unique(pd.to_datetime(trading_days)))).normalize()

    # ---- overnight window ----------------------------------------------------
    on = assign_overnight_day(art, trading_days)
    if len(on):
        ov = on.groupby("overnight_day").apply(_tone_aggregates).reset_index()
        ov = ov.rename(columns={"overnight_day": "date"})
    else:
        ov = pd.DataFrame(columns=["date", "count", "m_sent", "P_pos", "P_neg",
                                   "a_sent", "d_sent", "G", "c_n"])
    ov["date"] = pd.to_datetime(ov["date"]).dt.normalize()
    ov["x_int"] = np.log1p(ov["count"])
    ov["c_s"] = np.log1p(ov["c_n"])
    ov = ov.drop(columns=["count", "c_n"])

    # ---- full-day placebo window (all articles, by their own calendar day) ---
    art["cal_day"] = pd.to_datetime(art["adate"]).dt.normalize()
    fd = art.groupby("cal_day").apply(_tone_aggregates).reset_index()
    fd = fd.rename(columns={"cal_day": "date"})
    fd["date"] = pd.to_datetime(fd["date"]).dt.normalize()
    fd["x_int_full_raw"] = np.log1p(fd["count"])
    fd["c_s_f_raw"] = np.log1p(fd["c_n"])
    fd = fd.rename(columns={"m_sent": "m_sent_f", "P_pos": "P_pos_f", "P_neg": "P_neg_f",
                            "a_sent": "a_sent_f", "d_sent": "d_sent_f", "G": "G_f"})
    fd = fd.drop(columns=["count", "c_n"])

    # align full-day series onto the trading-day grid, then lag one trading day
    fd_grid = (fd.set_index("date").reindex(td).fillna(0.0))
    placebo = fd_grid.shift(1)                       # day t gets day t-1's full-day news
    placebo.columns = ["m_sent_f", "P_pos_f", "P_neg_f", "a_sent_f", "d_sent_f",
                       "G_f", "x_int_full", "c_s_f"]
    placebo = placebo.reset_index().rename(columns={"index": "date"})

    out = pd.DataFrame({"date": td}).merge(ov, on="date", how="left")
    out = out.merge(placebo, on="date", how="left")
    # zero-news stock-days -> 0 everywhere (spec)
    out = out.fillna(0.0)
    out["ticker"] = ticker
    return out


# =============================================================================
# PANEL ASSEMBLY
# =============================================================================
HAR_RK = ["rk_d", "rk_w", "rk_m"]
HAR_RV = ["rv_d", "rv_w", "rv_m"]

# every regressor that any model uses, so the in-sample sample is constant
NEWS_OVERNIGHT = ["x_int", "x_int_full", "m_sent", "P_pos", "P_neg",
                  "a_sent", "d_sent", "G", "c_s"]
NEWS_PLACEBO = ["m_sent_f", "P_pos_f", "P_neg_f", "a_sent_f", "d_sent_f", "G_f", "c_s_f"]
ALL_NEWS = NEWS_OVERNIGHT + NEWS_PLACEBO


def build_panel_ticker(ticker):
    m = load_both_measures(ticker)
    m["rv"] = m["rv"].clip(lower=RV_FLOOR)
    m["rk"] = m["rk"].clip(lower=RV_FLOOR)
    m["lrv"] = np.log(m["rv"])
    m["lrk"] = np.log(m["rk"])
    for tag, src in [("rk", "lrk"), ("rv", "lrv")]:
        m[f"{tag}_d"] = m[src].shift(1)
        m[f"{tag}_w"] = m[src].shift(1).rolling(WEEK).mean()
        m[f"{tag}_m"] = m[src].shift(1).rolling(MONTH).mean()

    news = build_news_vars(ticker, m["date"].values)
    df = m.merge(news, on="date", how="left")
    df[ALL_NEWS] = df[ALL_NEWS].fillna(0.0)
    df["ticker"] = ticker
    return df


def assemble_panel():
    frames = [build_panel_ticker(t) for t in TICKERS]
    panel = pd.concat(frames, ignore_index=True)
    missing = set(TICKERS) - set(panel["ticker"].unique())
    assert not missing, f"Missing tickers (path/suffix wrong?): {missing}"
    need = ["lrk", "lrv"] + HAR_RK + HAR_RV + ALL_NEWS
    panel = panel.dropna(subset=need).sort_values(["ticker", "date"]).reset_index(drop=True)
    return panel


# =============================================================================
# ESTIMATION CORE  (stacked OLS + stock dummies + Driscoll-Kraay SE)
# =============================================================================
def build_design(df, xcols, tickers):
    """Stock-specific intercepts (10 dummies, no global constant) + xcols."""
    D = pd.get_dummies(df["ticker"], prefix="fe").astype(float)
    D = D.reindex(columns=[f"fe_{t}" for t in tickers], fill_value=0.0)
    X = pd.concat([D.reset_index(drop=True),
                   df[xcols].astype(float).reset_index(drop=True)], axis=1)
    return X.values, list(X.columns)


def fit_ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(resid @ resid)
    return beta, resid, rss


def driscoll_kraay_se(X, resid, days, series_ids=None):
    """DK HAC: aggregate scores per day, Newey-West Bartlett, m=floor(4*(T/100)^(2/9)).
    Robust to cross-sectional dependence + serial correlation. series_ids is ignored
    (kept so DK and the per-series alternative share one call signature)."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    scores = X * resid[:, None]                       # n x k
    H = (pd.DataFrame(scores).assign(_d=np.asarray(days))
         .groupby("_d").sum().sort_index().values)    # T x k (daily score sums)
    T = H.shape[0]
    m = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
    S = H.T @ H                                        # Gamma_0
    for j in range(1, m + 1):
        w = 1.0 - j / (m + 1.0)                        # Bartlett
        G = H[j:].T @ H[:-j]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return np.sqrt(np.maximum(np.diag(V), 0.0)), m


def newey_west_by_series_se(X, resid, days, series_ids):
    """Robustness alternative to Driscoll-Kraay: a Newey-West HAC computed WITHIN
    each series (Bartlett kernel, lag m=floor(4*(T_i/100)^(2/9))) and summed across
    series. Robust to serial correlation/heteroskedasticity within a stock, but
    assumes cross-sectional independence -- the opposite dependence assumption to DK.
    Point estimates are identical; only the standard errors differ."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    scores = X * resid[:, None]
    df = pd.DataFrame(scores).assign(_s=np.asarray(series_ids), _d=np.asarray(days))
    S = np.zeros((k, k))
    m_used = 0
    for _, g in df.groupby("_s"):
        Hi = g.sort_values("_d").drop(columns=["_s", "_d"]).values   # T_i x k
        Ti = Hi.shape[0]
        m = int(np.floor(4.0 * (Ti / 100.0) ** (2.0 / 9.0)))
        m_used = max(m_used, m)
        Si = Hi.T @ Hi
        for j in range(1, m + 1):
            w = 1.0 - j / (m + 1.0)
            G = Hi[j:].T @ Hi[:-j]
            Si += w * (G + G.T)
        S += Si
    V = XtX_inv @ S @ XtX_inv
    return np.sqrt(np.maximum(np.diag(V), 0.0)), m_used


def gaussian_loglik(n, rss):
    """Concentrated normal log-likelihood."""
    return -0.5 * n * (np.log(2 * np.pi) + np.log(rss / n) + 1.0)


def aic_bic(n, rss, n_params):
    """p = #coefficients (dummies + slopes) + 1 for sigma^2."""
    ll = gaussian_loglik(n, rss)
    p = n_params + 1
    return -2 * ll + 2 * p, -2 * ll + p * np.log(n), ll


def lr_test(n, rss_restricted, rss_unrestricted, df):
    """LR = n*log(RSS_r/RSS_u) ~ chi2(df)."""
    if df <= 0:
        return np.nan, df, np.nan
    stat = n * np.log(rss_restricted / rss_unrestricted)
    return stat, df, float(1 - chi2.cdf(stat, df))


# =============================================================================
# MODEL GRID   (label, news regressors, benchmark-model-label for the LR test)
# =============================================================================
def make_models():
    main = [
        ("M0",  [],                                   None),
        ("M1",  ["x_int_full"],                        "M0"),
        ("M2",  ["x_int"],                             "M0"),
        ("M3",  ["m_sent"],                            "M0"),
        ("M4",  ["P_pos"],                             "M0"),
        ("M5",  ["P_neg"],                             "M0"),
        ("M6",  ["P_pos", "P_neg"],                    "M0"),
        ("M7",  ["a_sent"],                            "M0"),
        ("M8",  ["d_sent"],                            "M0"),
        ("M9",  ["G"],                                 "M0"),
        ("M10", ["a_sent", "d_sent", "G"],             "M0"),
        ("M11", ["c_s"],                               "M0"),
        ("M12", ["x_int", "m_sent"],                   "M2"),
        ("M13", ["x_int", "P_pos", "P_neg"],           "M2"),
        ("M14", ["x_int", "a_sent", "d_sent", "G"],    "M2"),
        ("M15", ["x_int", "c_s"],                      "M2"),
    ]
    # placebo twins: tone block uses the full-day versions; combo models use
    # full-day intensity so the whole news block is full-day.
    placebo = [
        ("M3d",  ["m_sent_f"],                                 "M0"),
        ("M4d",  ["P_pos_f"],                                  "M0"),
        ("M5d",  ["P_neg_f"],                                  "M0"),
        ("M6d",  ["P_pos_f", "P_neg_f"],                       "M0"),
        ("M7d",  ["a_sent_f"],                                 "M0"),
        ("M8d",  ["d_sent_f"],                                 "M0"),
        ("M9d",  ["G_f"],                                      "M0"),
        ("M10d", ["a_sent_f", "d_sent_f", "G_f"],              "M0"),
        ("M11d", ["c_s_f"],                                    "M0"),
        ("M12d", ["x_int_full", "m_sent_f"],                   "M1"),
        ("M13d", ["x_int_full", "P_pos_f", "P_neg_f"],         "M1"),
        ("M14d", ["x_int_full", "a_sent_f", "d_sent_f", "G_f"], "M1"),
        ("M15d", ["x_int_full", "c_s_f"],                      "M1"),
    ]
    return main, placebo


# =============================================================================
# IN-SAMPLE  (full panel: R^2, LR vs benchmark, AIC, BIC + coefficient detail)
# =============================================================================
def fit_all_models(panel, measure, models):
    """Fit every model on the full panel for one measure. Returns label -> dict."""
    har = HAR_RK if measure == "rk" else HAR_RV
    y = panel[f"l{measure}"].values
    days = panel["date"].values
    tickers = sorted(panel["ticker"].unique())
    fits = {}
    for label, news, bench in models:
        X, names = build_design(panel, har + news, tickers)
        beta, resid, rss = fit_ols(X, y)
        fits[label] = dict(news=news, bench=bench, X=X, names=names,
                           beta=beta, resid=resid, rss=rss, n=len(y),
                           k=X.shape[1], days=days, har=har)
    return fits


def _collect_coefs(models, fits, se_func, series_ids):
    """Build a long coef frame (model, block, var, coef, p) under a given SE estimator.
    Point estimates are shared across SE methods; only p changes with se_func."""
    rows = []
    for label, news, _ in models:
        f = fits[label]
        se, _m = se_func(f["X"], f["resid"], f["days"], series_ids)
        se_map = dict(zip(f["names"], se))
        b_map = dict(zip(f["names"], f["beta"]))
        har = f["har"]
        har_label = {har[0]: "HAR_daily", har[1]: "HAR_weekly", har[2]: "HAR_monthly"}
        for c in har + news:
            b, s = b_map[c], se_map[c]
            tval = b / s if s > 0 else np.nan
            rows.append(dict(model=label, block="HAR" if c in har else "news",
                             var=har_label.get(c, c), coef=b,
                             p=2 * (1 - norm.cdf(abs(tval))) if np.isfinite(tval) else np.nan))
    return pd.DataFrame(rows)


def run_insample(panel, measure, models, tag, fits, news_map):
    """fits: label -> fit dict spanning the whole model universe (main+placebo+benchmarks).
       news_map: label -> news regressor list, also universe-wide (for LR df)."""
    y = panel[f"l{measure}"].values
    tss = float(((y - y.mean()) ** 2).sum())

    # ---- summary table (SE-independent: R^2, AIC, BIC, LR) -------------------
    summ = []
    for label, news, bench in models:
        f = fits[label]
        r2 = 1.0 - f["rss"] / tss
        aic, bic, ll = aic_bic(f["n"], f["rss"], f["k"])
        if bench is None:
            lr, lrdf, lrp = np.nan, 0, np.nan
        else:
            added = [c for c in news if c not in news_map.get(bench, [])]
            lr, lrdf, lrp = lr_test(f["n"], fits[bench]["rss"], f["rss"], len(added))
        summ.append(dict(model=label, news="+".join(news) if news else "(HAR only)",
                         R2=r2, AIC=aic, BIC=bic, loglik=ll,
                         LR_vs=bench or "", LR_stat=lr, LR_df=lrdf, LR_p=lrp))
    summ = pd.DataFrame(summ)
    write_table(summ, f"insample_{tag}", measure)

    series_ids = panel["ticker"].values
    har_vars = ["HAR_daily", "HAR_weekly", "HAR_monthly"]
    set_label = "main models" if tag == "main" else "full-day placebo twins"

    # ---- wide coefficient matrix, Driscoll-Kraay SE (primary) ----------------
    coefs = _collect_coefs(models, fits, driscoll_kraay_se, series_ids)
    news_vars = [v for v in NEWS_TABLE_ORDER if v in set(coefs["var"])]
    write_wide_coefs(coefs, har_vars + news_vars, f"coefs_wide_{tag}",
                     measure, set_label, COEF_NOTE)

    # ---- robustness: per-series Newey-West SE (main set only) -----------------
    if tag == "main":
        coefs_nw = _collect_coefs(models, fits, newey_west_by_series_se, series_ids)
        write_wide_coefs(coefs_nw, har_vars + news_vars, f"coefs_wide_{tag}_nw",
                         measure, set_label + ", per-series Newey-West SE", COEF_NOTE_NW)
    return summ, coefs


# =============================================================================
# CONTEMPORANEOUS RESIDUAL CORRELATION ACROSS STOCKS  (Breusch-Pagan 1980 LM)
# -----------------------------------------------------------------------------
# Justifies the Driscoll-Kraay standard errors: it tests whether the HAR-baseline
# (M0) residuals of the different stocks are correlated on the same trading day.
# For each of the N(N-1)/2 stock pairs we take the correlation of their residuals
# over the days both are observed; LM = sum_{i<j} T_ij * rho_ij^2 ~ chi^2 with
# (#pairs) degrees of freedom. A rejection means the residuals move together
# across stocks, so DK (robust to exactly that) is the appropriate choice and
# plain/series-clustered errors would understate the true sampling variation.
# =============================================================================
def residual_crosscorr_lm(panel, measure, fits):
    f = fits["M0"]
    res = pd.DataFrame({"date": panel["date"].values,
                        "ticker": panel["ticker"].values,
                        "e": f["resid"]})
    R = res.pivot(index="date", columns="ticker", values="e")   # days x stocks
    cols = list(R.columns)
    N = len(cols)

    lm = 0.0
    rhos = []
    n_pairs = 0
    for a in range(N):
        for b in range(a + 1, N):
            ea, eb = R.iloc[:, a].values, R.iloc[:, b].values
            ok = np.isfinite(ea) & np.isfinite(eb)
            t_ij = int(ok.sum())
            if t_ij < 5:
                continue
            sa, sb = ea[ok].std(), eb[ok].std()
            if sa == 0 or sb == 0:
                continue
            rho = np.corrcoef(ea[ok], eb[ok])[0, 1]
            if not np.isfinite(rho):
                continue
            lm += t_ij * rho ** 2
            rhos.append(rho)
            n_pairs += 1

    df_chi = n_pairs                       # one df per included stock pair
    p = float(1 - chi2.cdf(lm, df_chi)) if df_chi > 0 else np.nan
    rhos = np.array(rhos)
    out = pd.DataFrame([dict(
        measure=f"log({measure.upper()})",
        n_stocks=N, n_pairs=n_pairs,
        LM_stat=lm, df=df_chi, p_value=p,
        mean_corr=float(rhos.mean()) if len(rhos) else np.nan,
        mean_abs_corr=float(np.abs(rhos).mean()) if len(rhos) else np.nan,
        max_abs_corr=float(np.abs(rhos).max()) if len(rhos) else np.nan,
    )])
    write_table(out, "resid_crosscorr_lm", measure)
    return out


# =============================================================================
# OUT-OF-SAMPLE  (expanding window, MSE-log + QLIKE, DM* vs M0 with HLN)
# =============================================================================
def qlike(actual_level, h):
    r = max(actual_level, RV_FLOOR) / max(h, RV_FLOOR)
    return r - np.log(r) - 1.0


def run_oos(panel, measure, models):
    har = HAR_RK if measure == "rk" else HAR_RV
    level_col = measure                       # 'rk' or 'rv'
    ycol = f"l{measure}"
    tickers = sorted(panel["ticker"].unique())
    dates = np.sort(panel["date"].unique())
    start = dates[int(len(dates) * INITIAL_TRAIN_FRAC)]
    test_days = dates[dates >= start]

    rows = []
    for tday in test_days:
        tr = panel[panel["date"] < tday]
        te = panel[panel["date"] == tday]
        if len(tr) < 50 or te.empty:
            continue
        ytr = tr[ycol].values
        for label, news, bench in models:
            Xtr, _ = build_design(tr, har + news, tickers)
            Xte, _ = build_design(te, har + news, tickers)
            beta, resid, rss = fit_ols(Xtr, ytr)
            sig2 = rss / max(len(ytr) - Xtr.shape[1], 1)    # in-sample residual variance
            yhat = Xte @ beta
            h = np.exp(yhat + 0.5 * sig2)                   # log-normal correction
            for j, (_, r) in enumerate(te.iterrows()):
                rows.append(dict(date=tday, ticker=r["ticker"], model=label,
                                 se_log=(r[ycol] - yhat[j]) ** 2,
                                 qlike=qlike(r[level_col], h[j])))
    loss = pd.DataFrame(rows)

    # ---- DM* vs M0 (daily-aggregated differentials, HLN small-sample fix) ----
    # Sign convention: d_t = loss(M0) - loss(model)  =>  DM_star > 0 means the
    # model BEATS M0. We report three things so direction is never ambiguous:
    #   p_two_sided : H1 "model differs from M0" (|DM*|); same in either direction
    #   p_better    : one-sided H1 "model beats M0"  (small p only when DM* > 0)
    #   verdict     : 'better' / 'worse' read straight off the sign of DM*
    # A large negative DM* (model worse) now gets a SMALL p_two_sided but a
    # p_better ~ 1, so it can no longer be mistaken for a good result.
    dm_rows = []
    daily = {}      # loss -> model -> per-day mean-loss series (for MCS)
    for lname in ("se_log", "qlike"):
        piv = loss.pivot_table(index=["date", "ticker"], columns="model", values=lname)
        day_mean = piv.groupby(level="date").mean()        # T_oos x models
        daily[lname] = day_mean
        T = day_mean.shape[0]
        base = day_mean["M0"]
        for label, news, bench in models:
            mloss = day_mean[label].mean()
            if label == "M0":
                dm_rows.append(dict(loss=lname, model=label, mean_loss=mloss,
                                    DM_star=np.nan, p_two_sided=np.nan,
                                    p_better=np.nan, verdict="benchmark"))
                continue
            d = (base - day_mean[label]).dropna().values   # >0 => model better
            v = d.var(ddof=1)
            dm = d.mean() / np.sqrt(v / T) if v > 0 else np.nan
            dm_star = dm * np.sqrt((T - 1) / T) if np.isfinite(dm) else np.nan
            if np.isfinite(dm_star):
                p_two = 2 * (1 - student_t.cdf(abs(dm_star), df=T - 1))
                p_better = 1 - student_t.cdf(dm_star, df=T - 1)   # H1: model beats M0
                verdict = "better" if dm_star > 0 else "worse"
            else:
                p_two = p_better = np.nan
                verdict = "n/a"
            dm_rows.append(dict(loss=lname, model=label, mean_loss=mloss,
                                DM_star=dm_star, p_two_sided=p_two,
                                p_better=p_better, verdict=verdict))
    dm_tab = pd.DataFrame(dm_rows)
    return loss, dm_tab, daily


# =============================================================================
# MODEL CONFIDENCE SET  (Hansen-Lunde-Nason: range statistic + block bootstrap)
# =============================================================================
def _block_bootstrap_idx(T, B, block, rng):
    nblocks = int(np.ceil(T / block))
    out = np.empty((B, nblocks * block), dtype=int)
    starts = rng.integers(0, T, size=(B, nblocks))
    offs = np.arange(block)
    for b in range(B):
        out[b] = ((starts[b][:, None] + offs[None, :]) % T).ravel()   # circular blocks
    return out[:, :T]


def model_confidence_set(day_loss, model_labels, B=MCS_B, block=MCS_BLOCK,
                         alpha=MCS_ALPHA, seed=MCS_SEED):
    """day_loss: T_oos x M array of per-day mean losses (lower=better)."""
    L = np.asarray(day_loss, float)
    T, M = L.shape
    block = block or max(1, int(round(T ** (1.0 / 3.0))))
    rng = np.random.default_rng(seed)
    boot_idx = _block_bootstrap_idx(T, B, block, rng)
    boot_means_full = np.stack([L[boot_idx[b]].mean(0) for b in range(B)])   # B x M

    alive = list(range(M))
    mcs_p = {}
    running = 0.0
    while len(alive) > 1:
        sub = np.array(alive)
        Lbar = L[:, sub].mean(0)                       # m
        bm = boot_means_full[:, sub]                    # B x m
        diff = Lbar[:, None] - Lbar[None, :]            # m x m
        bdiff = bm[:, :, None] - bm[:, None, :]         # B x m x m
        var = bdiff.var(0, ddof=1)
        np.fill_diagonal(var, np.inf)                   # ignore diagonal
        tstat = diff / np.sqrt(var)
        TR = np.nanmax(np.abs(tstat))
        bt = (bdiff - diff[None]) / np.sqrt(var)[None]
        TR_boot = np.nanmax(np.abs(bt).reshape(B, -1), axis=1)
        pval = float(np.mean(TR_boot >= TR))
        running = max(running, pval)
        # eliminate the worst model: largest standardized average loss advantage to others
        t_i = np.nanmax(tstat, axis=1)                  # per surviving model
        worst = sub[int(np.nanargmax(t_i))]
        mcs_p[worst] = running
        if pval >= alpha:                               # whole set survives -> stop
            for s in sub:
                mcs_p.setdefault(s, running)
            break
        alive.remove(worst)
    for s in alive:
        mcs_p.setdefault(s, 1.0)
    return pd.DataFrame({"model": model_labels,
                         "MCS_p": [mcs_p[i] for i in range(M)],
                         "in_MCS": [mcs_p[i] >= alpha for i in range(M)]})


def run_mcs(daily, main_models, measure):
    labels = [m[0] for m in main_models]
    for lname in ("se_log", "qlike"):
        dm = daily[lname][labels].dropna()
        tab = model_confidence_set(dm.values, labels)
        write_table(tab, f"mcs_{lname}", measure)


# =============================================================================
# TABLE OUTPUT  (CSV + LaTeX, one set per measure)
# =============================================================================
def write_table(df, name, measure):
    base = OUT_DIR / f"{name}_{measure.upper()}"
    df.to_csv(base.with_suffix(".csv"), index=False)
    try:
        df.to_latex(base.with_suffix(".tex"), index=False, float_format="%.4f")
    except Exception as e:
        print(f"  [latex skip] {name}: {e}")
    print(f"  wrote {base.name}.csv / .tex   ({len(df)} rows)")


# fixed left-to-right ordering of the news columns in coefficient tables
NEWS_TABLE_ORDER = ["x_int", "x_int_full", "m_sent", "P_pos", "P_neg",
                    "a_sent", "d_sent", "G", "c_s",
                    "m_sent_f", "P_pos_f", "P_neg_f", "a_sent_f", "d_sent_f",
                    "G_f", "c_s_f"]

# short LaTeX column headers (edit these if you prefer other abbreviations)
VAR_TEX = {
    "HAR_daily": r"HAR$_{d}$", "HAR_weekly": r"HAR$_{w}$", "HAR_monthly": r"HAR$_{m}$",
    "x_int": r"x\_int", "x_int_full": r"x\_full",
    "m_sent": r"m\_s", "P_pos": r"P$_{+}$", "P_neg": r"P$_{-}$",
    "a_sent": r"a\_s", "d_sent": r"d\_s", "G": r"G", "c_s": r"c\_s",
    "m_sent_f": r"m\_s$_{f}$", "P_pos_f": r"P$_{+,f}$", "P_neg_f": r"P$_{-,f}$",
    "a_sent_f": r"a\_s$_{f}$", "d_sent_f": r"d\_s$_{f}$", "G_f": r"G$_{f}$",
    "c_s_f": r"c\_s$_{f}$",
}
COEF_NOTE = (r"Driscoll-Kraay HAC standard errors (truncation lag m=5). "
             r"p-values in parentheses. */**/*** = significant at 10/5/1\%.")
COEF_NOTE_NW = (r"Per-series Newey-West HAC standard errors (Bartlett kernel, "
                r"per-series truncation lag m=5; assumes cross-sectional independence). "
                r"Robustness check on the Driscoll-Kraay results; point estimates are "
                r"identical. p-values in parentheses. */**/*** = significant at 10/5/1\%.")


def _stars(p):
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def _tex_escape(s):
    return str(s).replace("_", r"\_")


def write_wide_coefs(coefs, var_order, name, measure, set_label, note=COEF_NOTE):
    """Wide matrix in the thesis layout: full table env, resizebox to \\textwidth,
    coef* on top and (p) underneath, rounded to 2 decimals. CSV keeps full-precision
    numeric coef/p pairs for your own use. The \\label is derived from `name` so the
    Driscoll-Kraay and Newey-West tables get distinct labels."""
    models = list(dict.fromkeys(coefs["model"]))
    cmat = coefs.pivot(index="model", columns="var", values="coef").reindex(index=models, columns=var_order)
    pmat = coefs.pivot(index="model", columns="var", values="p").reindex(index=models, columns=var_order)
    base = OUT_DIR / f"{name}_{measure.upper()}"

    # ---- CSV: one row per model, full-precision numeric coef + p columns -----
    rows = []
    for m in models:
        row = {"model": m}
        for v in var_order:
            row[v] = cmat.loc[m, v]
            row[f"{v}_p"] = pmat.loc[m, v]
        rows.append(row)
    pd.DataFrame(rows).to_csv(base.with_suffix(".csv"), index=False)

    # ---- TeX: full table environment, resizebox, 2-decimal stacked layout ----
    headers = " & ".join(VAR_TEX.get(v, _tex_escape(v)) for v in var_order)
    caption = f"Coefficient matrix of the HAR models --- log({measure.upper()}), {set_label}."
    label = f"tab:{name}_{measure}"
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{" + caption + "}", r"\label{" + label + "}",
         r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{l" + "r" * len(var_order) + "}",
         r"\toprule",
         "Model & " + headers + r" \\",
         r"\midrule"]
    for m in models:
        c_cells, p_cells = [], []
        for v in var_order:
            c, p = cmat.loc[m, v], pmat.loc[m, v]
            if pd.isna(c):
                c_cells.append("")
                p_cells.append("")
            else:
                c_cells.append(f"{c:.2f}{_stars(p)}")
                p_cells.append(f"({p:.2f})")
        L.append(_tex_escape(m) + " & " + " & ".join(c_cells) + r" \\")
        L.append(" & " + " & ".join(p_cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}%", r"}",
          r"\smallskip", r"\small",
          r"\noindent " + note, r"\end{table}"]
    base.with_suffix(".tex").write_text("\n".join(L))
    print(f"  wrote {base.name}.csv / .tex   ({len(models)} models x {len(var_order)} vars)")


# =============================================================================
# ORCHESTRATION
# =============================================================================
def run_full_analysis(panel, measure):
    main, placebo = make_models()
    universe = main + placebo
    news_map = {lab: news for lab, news, _ in universe}
    primary = " (PRIMARY)" if measure == "rk" else " (robustness)"
    print("\n" + "=" * 72)
    print(f"MEASURE = log({measure.upper()}){primary}")
    print("=" * 72)

    fits = fit_all_models(panel, measure, universe)
    print("-- in-sample: main grid M0-M15")
    run_insample(panel, measure, main, "main", fits, news_map)
    print("-- in-sample: full-day placebo twins (appendix)")
    run_insample(panel, measure, placebo, "placebo", fits, news_map)

    print("-- contemporaneous residual correlation across stocks (Breusch-Pagan LM)")
    residual_crosscorr_lm(panel, measure, fits)

    print("-- out-of-sample + DM* vs M0: main grid")
    _, dm_main, daily = run_oos(panel, measure, main)
    write_table(dm_main, "oos_dm_main", measure)
    print("-- out-of-sample + DM* vs M0: placebo twins (appendix)")
    _, dm_plac, _ = run_oos(panel, measure, [main[0]] + placebo)   # main[0] == M0 (DM benchmark)
    dm_plac = dm_plac[dm_plac["model"] != "M0"]
    write_table(dm_plac, "oos_dm_placebo", measure)

    if RUN_MCS:
        print("-- model confidence set over M0-M15 (both losses)")
        run_mcs(daily, main, measure)


if __name__ == "__main__":
    print("=" * 72)
    print("ASSEMBLING PANEL")
    print("=" * 72)
    panel = assemble_panel()
    print(f"{panel['ticker'].nunique()} tickers, {panel['date'].nunique()} dates, "
          f"{len(panel)} stock-day obs")

    print("\n=== RV/RK HEALTH CHECK (per ticker) ===")
    print(panel.groupby("ticker")[["lrk", "lrv"]].agg(["count", "mean", "std"]).round(3))

    run_full_analysis(panel, "rk")     # primary
    run_full_analysis(panel, "rv")     # robustness
    print(f"\nDone. All tables written to: {OUT_DIR}")