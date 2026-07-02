# Master-Thesis-Teun-Wesseling
This github page contains all coding and data I used for my master thesis. 

## Folder layout

```
Master Thesis/
├── README.md
├── requirements.txt
├── code/                          # the scripts 
│   ├── 01_export_news.py ... 06_main_models.py
│   └── checks/vif_collinearity.py
├── <ticker>.parquet               # per-stock news input for step 2
├── <TICKER>_finbert_*.parquet     # step 2 output (article + daily)
├── raw data/                      # WRDS TAQ trade files (not in dropbox)
├── raw news/                      # raw LSEG news (not in dropbox)
├── clean data/                    # cleaned intraday, step 4 output
├── enriched/                      # final news files, step 3 output
└── results/                       # tables (step 6) + figures/ (step 5)
```



Python 3.10+. FinBERT (step 2) uses the GPU if available, otherwise the CPU.

# Pipeline
Steps 1 and 4 need restricted data access
(LSEG workstation and WRDS); everything after that reproduces from the included
data.

1. **`01_export_news.py`** — export Reuters news per stock (LSEG). Afterwards,
   manually rename each `articles_<ric>_reuters_full.parquet` to `<ticker>.parquet`.
2. **`02_finbert_sentiment.py`** — FinBERT score per article + daily aggregates.
3. **`03_enrich_news.py`** — add market-session counts → `enriched/`.
4. **`04_clean_taq.py`** — filter WRDS TAQ trades → `clean data/`.
5. **`05_descriptives.py`** — the four thesis figures → `results/figures/`.
6. **`06_main_models.py`** — all result tables (RK + RV) → `results/`.


## Outputs
- **`results/`** — in-sample grid, coefficient tables (Driscoll–Kraay and
  Newey–West), residual cross-correlation test, out-of-sample Diebold–Mariano,
  and the Model Confidence Set; each as `.csv` and `.tex`, for RK and RV.
- **`results/figures/`** — ACF, RV vs log-RV distribution, returns per stock,
  log-RV per stock (RK and RV).
