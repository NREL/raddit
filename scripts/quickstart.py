#!/usr/bin/env python3
"""
quickstart_minimal.py
-------------------------------------------------
A CPU-only, ≤10-minute reproduction of the core prediction
pipeline on a 1/300 sampled job-trace.

Outputs:
  • baseline_power_vs_actual.png
  • baseline_runtime_vs_actual.png
  • semantic_power_vs_actual.png
  • semantic_runtime_vs_actual.png
All plots are written to  ../data/quickstart_output/
"""

import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
SAMPLE_PARQUET = Path("../data/historic_job_trace_sample.parquet")
OUTPUT_DIR     = Path("../data/quickstart_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_STATE   = 123

def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
#  Load / verify sample data
# ------------------------------------------------------------------
def load_sample_df() -> pd.DataFrame:
    if SAMPLE_PARQUET.exists():
        df = pd.read_parquet(SAMPLE_PARQUET)
    else:
        raise FileNotFoundError(
            f"Sample parquet not found at {SAMPLE_PARQUET}. "
            "Please run sampling script or place file accordingly."
        )
    # Minimal sanity check
    expected = {"avg_power_per_node", "wallclock_used_sec", "script"}
    if not expected.issubset(df.columns):
        raise ValueError(f"Sample parquet missing columns: {expected-df.columns.keys()}")
    return df.reset_index(drop=True)

# ------------------------------------------------------------------
#  Baseline models
# ------------------------------------------------------------------
def train_baseline(df: pd.DataFrame):
    feats  = ["nodes_req", "wallclock_req_sec"]          # simple numeric subset
    target_power   = "avg_power_per_node"
    target_runtime = "wallclock_used_sec"

    # Fill NAs just in case
    df[feats] = df[feats].fillna(0)

    # 80/20 split
    train_df = df.sample(frac=0.8, random_state=RANDOM_STATE)
    test_df  = df.drop(train_df.index)

    X_tr, X_te = train_df[feats], test_df[feats]

    y_tr_pwr, y_te_pwr = train_df[target_power],   test_df[target_power]
    y_tr_rt,  y_te_rt  = train_df[target_runtime], test_df[target_runtime]

    mod_pwr = XGBRegressor(n_estimators=200, max_depth=6, random_state=RANDOM_STATE)
    mod_rt  = XGBRegressor(n_estimators=200, max_depth=6, random_state=RANDOM_STATE)

    mod_pwr.fit(X_tr, y_tr_pwr)
    mod_rt.fit (X_tr, y_tr_rt)

    pred_pwr = mod_pwr.predict(X_te)
    pred_rt  = mod_rt.predict(X_te)

    print(f"[Baseline] MAE power   : {mean_absolute_error(y_te_pwr, pred_pwr):8.2f} W")
    print(f"[Baseline] MAE runtime : {mean_absolute_error(y_te_rt,  pred_rt ):8.2f} s")

    scatter_plot(
        y_te_pwr, pred_pwr, "Actual Power (W)",
        "Predicted Power (W)",
        OUTPUT_DIR / "baseline_power_vs_actual.png"
    )

    scatter_plot(
        y_te_rt, pred_rt,  "Actual Runtime (s)",
        "Predicted Runtime (s)",
        OUTPUT_DIR / "baseline_runtime_vs_actual.png"
    )

# ------------------------------------------------------------------
#  Lightweight “embeddings” + semantic search
# ------------------------------------------------------------------
def semantic_search_quick(df: pd.DataFrame):
    """
    Stand-in for the heavy LLM workflow:
      • TF-IDF on job script text  → 10 000 dims
      • Truncated SVD → 100 dims   (≈“embedding”)
      • 5-NN cosine search within that space
    """
    vec = TfidfVectorizer(max_features=10_000)
    X_tfidf = vec.fit_transform(df["script"].fillna(""))

    svd = TruncatedSVD(n_components=100, random_state=RANDOM_STATE)
    emb = svd.fit_transform(X_tfidf)

    # Hold-out 20 % as queries
    query_mask = df.sample(frac=0.2, random_state=RANDOM_STATE).index
    train_idx  = df.index.difference(query_mask)

    train_emb  = emb[train_idx]
    query_emb  = emb[query_mask]

    nbrs = NearestNeighbors(n_neighbors=5, metric="cosine").fit(train_emb)
    dists, neigh_idx = nbrs.kneighbors(query_emb)

    # Weighted predictions
    true_pwr = df.loc[query_mask, "avg_power_per_node"].values
    true_rt  = df.loc[query_mask, "wallclock_used_sec"].values

    pred_pwr = []
    pred_rt  = []
    for row_nbrs, row_dist in zip(neigh_idx, dists):
        w = 1 / (row_dist + 1e-6)
        w /= w.sum()
        pred_pwr.append((df.loc[train_idx[row_nbrs], "avg_power_per_node"].values * w).sum())
        pred_rt .append((df.loc[train_idx[row_nbrs], "wallclock_used_sec"].values   * w).sum())

    print(f"[Semantic] MAE power   : {mean_absolute_error(true_pwr, pred_pwr):8.2f} W")
    print(f"[Semantic] MAE runtime : {mean_absolute_error(true_rt,  pred_rt ):8.2f} s")

    scatter_plot(
        true_pwr, pred_pwr, "Actual Power (W)",
        "Predicted Power (W)",
        OUTPUT_DIR / "semantic_power_vs_actual.png"
    )
    scatter_plot(
        true_rt,  pred_rt,  "Actual Runtime (s)",
        "Predicted Runtime (s)",
        OUTPUT_DIR / "semantic_runtime_vs_actual.png"
    )

# ------------------------------------------------------------------
#  Utility: scatter + y=x line
# ------------------------------------------------------------------
def scatter_plot(y_true, y_pred, xlab, ylab, out_path):
    plt.figure(figsize=(4,4), dpi=150)
    plt.scatter(y_true, y_pred, s=8, alpha=0.6)
    lims = [min(y_true)*0.9, max(y_true)*1.1]
    plt.plot(lims, lims, "--k", linewidth=0.8)
    plt.xlabel(xlab); plt.ylabel(ylab)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ------------------------------------------------------------------
#  Main driver
# ------------------------------------------------------------------
def main():
    ensure_output_dir()
    df = load_sample_df()

    print(f"Loaded sample dataset: {len(df):,} jobs")

    train_baseline(df)
    semantic_search_quick(df)

    print(f"\nQuick-start complete ✔  — plots saved to  {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
