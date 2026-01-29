# ============================
# PART A — Config & Inputs
# ============================

import os
import sys
import math
import yaml
import pandas as pd
import numpy as np

CONFIG_YAML_PATH = "config.yaml"

# ============================
# Helpers
# ============================

def load_yaml(path_or_obj):
    """Load config from YAML file path or a dict-like object."""
    if isinstance(path_or_obj, (dict,)):
        return path_or_obj
    if not os.path.exists(path_or_obj):
        raise FileNotFoundError(f"Config YAML not found: {path_or_obj}")
    with open(path_or_obj, "r") as f:
        return yaml.safe_load(f)

def parse_ts_local(ts_str_or_obj, tz_name):
    """Parse a timestamp string (with or without offset) and convert to the target tz."""
    ts = pd.to_datetime(ts_str_or_obj, errors="raise", utc=True)
    return ts.tz_convert(tz_name)

def make_trailing_grid(start_local, end_local, minutes=15):
    """
    Build a 15-min trailing label grid
    where each label T represents [T-delta, T).
    """
    delta = pd.Timedelta(minutes=minutes)
    first_label = start_local + delta
    # Inclusive end label at 'end_local' so the last window is [end-delta, end)
    if end_local < first_label:
        return pd.DatetimeIndex([], tz=start_local.tz)
    return pd.date_range(start=first_label, end=end_local, freq=f"{minutes}min", tz=start_local.tz)

def load_and_merge_blackouts(csv_path):
    """
    Read maintenance CSV with columns 'start','end' (tz-aware or strings with offsets).
    Returns a list of (start,end) as tz-aware Timestamps merged for overlaps.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Maintenance CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if not {"start", "end"}.issubset(df.columns):
        raise ValueError("Maintenance CSV must have columns: start,end")
    df["start"] = pd.to_datetime(df["start"], errors="raise")
    df["end"]   = pd.to_datetime(df["end"],   errors="raise")
    bad = (df["end"] <= df["start"])
    if bad.any():
        raise ValueError(f"Found {bad.sum()} maintenance rows with end <= start.")
    df = df.sort_values("start").reset_index(drop=True)
    merged = []
    cur_s, cur_e = None, None
    for s, e in zip(df["start"], df["end"]):
        if cur_s is None:
            cur_s, cur_e = s, e
            continue
        if s <= cur_e:  # overlap or touching
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    if cur_s is not None:
        merged.append((cur_s, cur_e))
    return merged

def require_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name}: missing required column(s): {missing}")

def half_up_whole_kw(x_kw, eps=1e-6):
    """
    Half-up to whole kW:
      >= .5 rounds up; < .5 rounds down.
      eps handles floating fuzz.
    """
    frac = x_kw - np.floor(x_kw)
    up = (frac + eps) >= 0.5
    return np.floor(x_kw) + up.astype(int)

# ============================
# Load CONFIG
# ============================
cfg = load_yaml(CONFIG_YAML_PATH)

# Pull sections (with minimal validation)
calendar = cfg.get("calendar", {})
rates    = cfg.get("rates", {})
billing  = cfg.get("billing", {})
policies = cfg.get("policies", {})
solve    = cfg.get("solve", {})
outputs  = cfg.get("outputs", {})
paths    = cfg.get("paths", {})

# Required calendar fields
cal_tz   = calendar.get("timezone", "America/Denver")
cal_start_local = parse_ts_local(calendar["start"], cal_tz)
cal_end_local   = parse_ts_local(calendar["end"],   cal_tz)
holidays_list   = calendar.get("holidays", [])

# Basic billing checks
interval_minutes = billing.get("interval_minutes", 15)
record_labeling  = billing.get("record_labeling", "trailing")
rounding_mode    = billing.get("demand_billing_rounding", "half_up_whole_kw")
if interval_minutes != 15:
    raise ValueError("This pipeline assumes 15-minute billing intervals.")
if record_labeling.lower() != "trailing":
    raise ValueError("This pipeline assumes 'trailing' record labeling (T means [T-15m, T)).")
if rounding_mode.lower() != "half_up_whole_kw":
    raise ValueError("This pipeline assumes 'half_up_whole_kw' demand rounding.")

# Policies flags
multi_node_same_type          = bool(policies.get("multi_node_same_type", True))
allow_any_type_if_unset       = bool(policies.get("allow_any_type_if_unset", True))
forbid_maintenance_overlap    = bool(policies.get("forbid_maintenance_overlap", True))
background_nonnegative_clip   = bool(policies.get("background_nonnegative_clip", True))
holidays_as_weekends          = bool(policies.get("holidays_treated_like_weekends", True))

# Solve config
half_up_eps = float(solve.get("half_up_epsilon_kw", 1e-6))
stageA_solver = solve.get("stageA_solver", "highs")
stageA_minutes = int(solve.get("stageA_minutes", 5))
stageB_minutes_per_window = int(solve.get("stageB_minutes_per_window", 60))
window_weeks = int(solve.get("window_weeks", 6))
overlap_weeks = int(solve.get("overlap_weeks", 2))

# Paths
maintenance_csv = paths.get("maintenance_windows_csv")
merged_csv      = paths.get("merged_csv") or "../data/merged_campus_cluster_jobs.csv"  # fallback
jobs_csv        = paths.get("jobs_csv")
nodes_csv       = paths.get("nodes_csv")

# ============================
# Load & merge maintenance
# ============================
blackouts = []
if forbid_maintenance_overlap:
    if not maintenance_csv:
        raise FileNotFoundError("policies.forbid_maintenance_overlap is true but paths.maintenance_windows_csv is not set.")
    blackouts = load_and_merge_blackouts(maintenance_csv)

# ============================
# Load merged timeseries
# ============================
if not merged_csv or not os.path.exists(merged_csv):
    raise FileNotFoundError(f"paths.merged_csv not found: {merged_csv}")
merged = pd.read_csv(merged_csv)
require_columns(
    merged,
    ["timestamp", "baseline_kw", "background_kw", "campus_kw", "cluster_kw", "jobs_kw"],
    "merged_csv"
)
merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="coerce")
merged = merged.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

# Optional clipping of background
merged["background_kw_raw"] = merged["background_kw"].astype(float)
if background_nonnegative_clip:
    merged["background_kw"] = merged["background_kw_raw"].clip(lower=0.0)

# ============================
# Build 15-min trailing grid and validate coverage
# ============================
slot_ends = make_trailing_grid(cal_start_local, cal_end_local, minutes=interval_minutes)

# Reindex merged to the expected grid (we'll keep only expected labels)
merged_on_grid = merged.reindex(slot_ends)

missing = merged_on_grid["baseline_kw"].isna().sum() + merged_on_grid["background_kw"].isna().sum()
# Also check the other columns for coverage (QA)
missing_any = merged_on_grid[["campus_kw","cluster_kw","jobs_kw"]].isna().sum().sum()

# ============================
# Quick QA identities
# ============================
# cluster ≈ jobs + background
res_cluster = (merged_on_grid["cluster_kw"] - (merged_on_grid["jobs_kw"].fillna(0) + merged_on_grid["background_kw"])).abs()
# campus ≈ baseline + cluster
res_campus  = (merged_on_grid["campus_kw"] - (merged_on_grid["baseline_kw"] + merged_on_grid["cluster_kw"])).abs()

# ============================
# GO / NO-GO PRINTS
# ============================
print("=== PART A: GO/NO-GO CHECKS ===")
print(f"YAML loaded from: {CONFIG_YAML_PATH}")
print(f"Calendar: tz={cal_tz}, start_local={cal_start_local}, end_local={cal_end_local}")
print(f"Billing: interval={interval_minutes} min, labeling={record_labeling}, rounding={rounding_mode}")
print(f"Policies: multi_node_same_type={multi_node_same_type}, allow_any_type_if_unset={allow_any_type_if_unset}, "
      f"forbid_maintenance_overlap={forbid_maintenance_overlap}, background_nonnegative_clip={background_nonnegative_clip}, "
      f"holidays_treated_like_weekends={holidays_as_weekends}")
print(f"Solve: StageA={stageA_solver}({stageA_minutes}m), StageB_window={window_weeks}w+{overlap_weeks}w overlap "
      f"({stageB_minutes_per_window}m/window), half_up_eps={half_up_eps}")
print(f"Paths: maintenance_csv={maintenance_csv}, merged_csv={merged_csv}, jobs_csv={jobs_csv}, nodes_csv={nodes_csv}")

if forbid_maintenance_overlap:
    print(f"[GO] Maintenance blackouts loaded & merged: {len(blackouts)} window(s).")
    for i, (s,e) in enumerate(blackouts[:3]):
        print(f"   - blackout[{i}]: {s} → {e}")

print(f"[GO] Merged CSV rows: {len(merged):,}")
print(f"[GO] Merged time range: {merged.index.min()} → {merged.index.max()}")
print(f"[GO] Required columns present: baseline_kw, background_kw, campus_kw, cluster_kw, jobs_kw")

print(f"[GO] Built trailing grid: {len(slot_ends):,} labels from {slot_ends.min()} → {slot_ends.max()}")

if missing == 0 and missing_any == 0:
    print("[GO] Coverage: merged series fully cover the horizon grid (no gaps).")
else:
    # Show a small sample of missing stamps for debugging
    is_missing_any = merged_on_grid[["baseline_kw","background_kw","campus_kw","cluster_kw","jobs_kw"]].isna().any(axis=1)
    missing_stamps = list(merged_on_grid.index[is_missing_any][:5])
    print("[NO-GO] Coverage gaps detected on the grid.")
    print(f"       Missing count across required columns: baseline/background={missing}, any_col={missing_any}")
    print(f"       Sample missing timestamps (first 5): {missing_stamps}")

# Residual QA (not hard-fail; informative)
if len(merged_on_grid) > 0:
    r1 = float(np.nanmax(res_cluster.values)) if not res_cluster.empty else float("nan")
    r2 = float(np.nanmax(res_campus.values))  if not res_campus.empty  else float("nan")
    print(f"[QA] Max |cluster - (jobs+background)| = {r1:.6f} kW")
    print(f"[QA] Max |campus - (baseline+cluster)| = {r2:.6f} kW")

# Final explicit verdict
all_good = (missing == 0 and missing_any == 0)
print(f"VERDICT: {'GO' if all_good else 'NO-GO'}")
