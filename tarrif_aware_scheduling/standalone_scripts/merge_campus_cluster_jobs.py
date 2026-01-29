import os
import pandas as pd
import numpy as np

# -----------------------------
# Config (adjust paths as needed)
# -----------------------------
CAMPUS_CLUSTER_CSV = "../data/campus_cluster_demand.csv"
JOBS_CSV           = "../data/jobs_demand_kw.csv"
OUT_CSV            = "../data/merged_campus_cluster_jobs.csv"

# If minor meter/job noise yields small negatives, clip background to >= 0:
CLIP_BACKGROUND_NONNEG = True

# -----------------------------
# Helpers
# -----------------------------
def _pick_jobs_power_col(cols):
    for c in ("avg_power_kw", "cluster_jobs_kw", "jobs_kw", "avg_power_consumed", "power_kw"):
        if c in cols:
            return c
    raise KeyError(
        "Could not find a jobs power column. "
        "Expected one of: avg_power_kw, cluster_jobs_kw, jobs_kw, avg_power_consumed, power_kw"
    )

def _read_campus_cluster(path):
    df = pd.read_csv(path)
    # Parse and index by timestamp (tz-aware if offset present)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    # Ensure numeric
    for c in ("campus_kw", "cluster_kw"):
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}' in {path}")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["campus_kw", "cluster_kw"])
    return df

def _read_jobs(path):
    j = pd.read_csv(path)
    if "timestamp" not in j.columns:
        raise KeyError(f"Missing 'timestamp' in {path}")
    power_col = _pick_jobs_power_col(set(j.columns))
    j["timestamp"] = pd.to_datetime(j["timestamp"], errors="coerce")
    j = j.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    j["jobs_kw"] = pd.to_numeric(j[power_col], errors="coerce")
    j = j.dropna(subset=["jobs_kw"])
    return j[["jobs_kw"]]

# -----------------------------
# Main merge
# -----------------------------
def main():
    # Read sources
    df_cc = _read_campus_cluster(CAMPUS_CLUSTER_CSV)  # campus_kw, cluster_kw
    df_jobs = _read_jobs(JOBS_CSV)                    # jobs_kw

    # Align to campus/cluster timestamps so we keep the utility’s 15-min grid
    merged = df_cc.join(df_jobs, how="left")

    # Derived signals
    # Baseline = campus without cluster
    merged["baseline_kw"] = merged["campus_kw"] - merged["cluster_kw"]

    # Fill missing jobs with 0 for arithmetic (no jobs in that interval)
    jobs_kw_filled = merged["jobs_kw"].fillna(0.0)

    # Background (GPUs/shared/other) = cluster - jobs
    merged["background_kw"] = merged["cluster_kw"] - jobs_kw_filled

    # Reorder columns
    col_order = [
        "campus_kw", "cluster_kw", "jobs_kw", "baseline_kw",
        "background_kw"
    ]
    cols_present = [c for c in col_order if c in merged.columns]
    merged = merged[cols_present]
    merged.dropna(inplace=True, subset='jobs_kw')

    # Write out
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    merged.to_csv(OUT_CSV, index=True, date_format=None)
    print(f"[OK] Wrote merged data: {OUT_CSV} ({len(merged):,} rows)")
    # Quick peek
    print(merged.head(3))

if __name__ == "__main__":
    main()
