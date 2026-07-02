#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
checks/vif_collinearity.py
=================================================================
Ad-hoc robustness check (not part of the main results run): collinearity of the
news regressors in the panel, in particular P_pos vs P_neg in model M6.

Reports, on the panel from 06_main_models.py:
  - raw and news-days-only correlation of P_pos and P_neg
  - partial correlation given the HAR lags
  - VIFs of the M6 regressor block (HAR + P_pos + P_neg) and of the tone block

Run:  python checks/vif_collinearity.py
"""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# load the canonical backbone (one level up) without running its analysis
_BACKBONE = Path(__file__).resolve().parent.parent / "06_main_models.py"
_spec = importlib.util.spec_from_file_location("thesis_model", _BACKBONE)
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

panel = T.assemble_panel()            # rebuilds the panel (takes a while)

HAR = ["rk_d", "rk_w", "rk_m"]        # RK-version HAR lags
mask = (panel["P_pos"] != 0) | (panel["P_neg"] != 0)   # real news days only

print("corr(P_pos, P_neg) — all stock-days (incl. zero-news):")
print(panel[["P_pos", "P_neg"]].corr())
print("\ncorr(P_pos, P_neg) — news days only:")
print(panel.loc[mask, ["P_pos", "P_neg"]].corr())

# ---- partial correlation given the HAR lags --------------------------------
try:
    import pingouin as pg
    print("\nPartial corr P_pos~P_neg | HAR (all stock-days):")
    print(pg.partial_corr(panel, x="P_pos", y="P_neg", covar=HAR).round(4))
    print("\nPartial corr P_pos~P_neg | HAR (news days only):")
    print(pg.partial_corr(panel[mask], x="P_pos", y="P_neg", covar=HAR).round(4))
except ImportError:
    print("\n(pingouin not installed; skipping partial correlation. "
          "`pip install pingouin` to enable.)")


def vif_table(df, cols):
    X = add_constant(df[cols].astype(float))
    out = {c: variance_inflation_factor(X.values, i)
           for i, c in enumerate(X.columns) if c != "const"}
    return pd.Series(out, name="VIF").round(3)


for label, cols in [("M6 block: HAR + P_pos + P_neg", HAR + ["P_pos", "P_neg"]),
                    ("tone block: HAR + x_int + a_sent + d_sent + G",
                     HAR + ["x_int", "a_sent", "d_sent", "G"])]:
    print(f"\nVIF — {label}")
    print("  all stock-days:")
    print(vif_table(panel, cols).to_string())
    print("  news days only:")
    print(vif_table(panel[mask], cols).to_string())
