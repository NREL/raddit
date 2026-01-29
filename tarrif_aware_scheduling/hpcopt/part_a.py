# hpcopt/part_a.py
from __future__ import annotations
import os
import yaml
import pandas as pd
import numpy as np
from dataclasses import dataclass

# ------------ helpers ------------
def _load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config YAML not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _parse_local(ts, tz):
    return pd.to_datetime(ts, errors="raise", utc=True).tz_convert(tz)

def _make_trailing_grid(start_local: pd.Timestamp, end_local: pd.Timestamp, minutes: int) -> pd.DatetimeIndex:
    delta = pd.Timedelta(minutes=minutes)
    first_label = start_local + delta
    if end_local < first_label:
        return pd.DatetimeIndex([], tz=start_local.tz)
    return pd.date_range(start=first_label, end=end_local, freq=f"{minutes}min", tz=start_local.tz)

def _merge_blackouts(csv_path: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    df = pd.read_csv(csv_path)
    if not {"start","end"}.issubset(df.columns):
        raise ValueError("Maintenance CSV must have columns: start,end")
    df["start"] = pd.to_datetime(df["start"], errors="raise")
    df["end"]   = pd.to_datetime(df["end"],   errors="raise")
    bad = (df["end"] <= df["start"])
    if bad.any():
        raise ValueError(f"Found {bad.sum()} maintenance rows with end <= start.")
    df = df.sort_values("start").reset_index(drop=True)
    out, cur = [], None
    for s, e in zip(df["start"], df["end"]):
        if cur is None:
            cur = [s, e]
        elif s <= cur[1]:
            cur[1] = max(cur[1], e)
        else:
            out.append(tuple(cur)); cur = [s, e]
    if cur: out.append(tuple(cur))
    return out

def _require_cols(df: pd.DataFrame, cols: list[str], name: str):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"{name}: missing columns: {miss}")

# ------------ results container ------------
@dataclass
class PartAResult:
    cfg: dict
    slot_ends: pd.DatetimeIndex
    merged_on_grid: pd.DataFrame
    blackouts: list[tuple[pd.Timestamp, pd.Timestamp]]
    go: bool

# ------------ main API ------------
def run_part_a(config_path: str) -> PartAResult:
    cfg = _load_yaml(config_path)

    calendar = cfg.get("calendar", {})
    billing  = cfg.get("billing", {})
    policies = cfg.get("policies", {})
    solve    = cfg.get("solve", {})
    paths    = cfg.get("paths", {})

    # calendar
    tz = calendar.get("timezone", "America/Denver")
    start_local = _parse_local(calendar["start"], tz)
    end_local   = _parse_local(calendar["end"],   tz)

    # billing checks
    interval_minutes = int(billing.get("interval_minutes", 15))
    if interval_minutes != 15:
        raise ValueError("This pipeline assumes 15-minute billing intervals.")
    if billing.get("record_labeling","trailing").lower() != "trailing":
        raise ValueError("This pipeline assumes 'trailing' record labeling.")
    if billing.get("demand_billing_rounding","half_up_whole_kw").lower() != "half_up_whole_kw":
        raise ValueError("This pipeline assumes 'half_up_whole_kw' rounding.")

    # policies
    forbid_maint = bool(policies.get("forbid_maintenance_overlap", True))
    bg_clip      = bool(policies.get("background_nonnegative_clip", True))
    holidays_as_weekends = bool(policies.get("holidays_treated_like_weekends", True))

    # solve
    half_up_eps = float(solve.get("half_up_epsilon_kw", 1e-6))

    # paths
    maintenance_csv = paths.get("maintenance_windows_csv")
    merged_csv      = paths.get("merged_csv") or "../data/merged_campus_cluster_jobs.csv"

    # maintenance
    blackouts = []
    if forbid_maint:
        if not maintenance_csv or not os.path.exists(maintenance_csv):
            raise FileNotFoundError("forbid_maintenance_overlap is true but maintenance_windows_csv is missing.")
        blackouts = _merge_blackouts(maintenance_csv)

    # merged
    if not os.path.exists(merged_csv):
        raise FileNotFoundError(f"paths.merged_csv not found: {merged_csv}")
    merged = pd.read_csv(merged_csv)
    _require_cols(merged, ["timestamp","baseline_kw","background_kw","campus_kw","cluster_kw","jobs_kw"], "merged_csv")
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="coerce")
    merged = merged.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    merged["background_kw_raw"] = merged["background_kw"].astype(float)
    if bg_clip:
        merged["background_kw"] = merged["background_kw_raw"].clip(lower=0.0)

    # grid + coverage
    slot_ends = _make_trailing_grid(start_local, end_local, minutes=interval_minutes)
    merged_on_grid = merged.reindex(slot_ends)

    missing_required = (
        merged_on_grid["baseline_kw"].isna().sum()
        + merged_on_grid["background_kw"].isna().sum()
    )
    missing_any = merged_on_grid[["campus_kw","cluster_kw","jobs_kw"]].isna().sum().sum()

    # identities
    res_cluster = (merged_on_grid["cluster_kw"] - (merged_on_grid["jobs_kw"].fillna(0) + merged_on_grid["background_kw"])).abs()
    res_campus  = (merged_on_grid["campus_kw"]  - (merged_on_grid["baseline_kw"] + merged_on_grid["cluster_kw"])).abs()

    # prints
    print("=== PART A: GO/NO-GO CHECKS ===")
    print(f"YAML: {config_path}")
    print(f"Calendar: tz={tz}, start={start_local}, end={end_local}")
    print(f"Billing: 15-min, trailing, rounding=half_up_whole_kw, eps={half_up_eps}")
    print(f"Policies: forbid_maintenance_overlap={forbid_maint}, background_nonnegative_clip={bg_clip}, holidays_treated_like_weekends={holidays_as_weekends}")
    print(f"Paths: maintenance_csv={maintenance_csv}, merged_csv={merged_csv}")
    if forbid_maint:
        print(f"[GO] Maintenance windows: {len(blackouts)}")
        for i,(s,e) in enumerate(blackouts[:3]):
            print(f"  - blackout[{i}]: {s} → {e}")

    print(f"[GO] Merged rows: {len(merged):,}")
    print(f"[GO] Merged range: {merged.index.min()} → {merged.index.max()}")
    print(f"[GO] Grid: {len(slot_ends):,} labels; {slot_ends.min()} → {slot_ends.max()}")

    if missing_required == 0 and missing_any == 0:
        print("[GO] Coverage: merged fully covers the grid.")
        go = True
    else:
        is_missing = merged_on_grid[["baseline_kw","background_kw","campus_kw","cluster_kw","jobs_kw"]].isna().any(axis=1)
        stamps = list(merged_on_grid.index[is_missing][:5])
        print("[NO-GO] Coverage gaps on the grid.")
        print(f"       Missing required={missing_required}, any_col={missing_any}")
        print(f"       Sample timestamps: {stamps}")
        go = False

    if len(merged_on_grid):
        r1 = float(np.nanmax(res_cluster.values)) if not res_cluster.empty else float("nan")
        r2 = float(np.nanmax(res_campus.values))  if not res_campus.empty  else float("nan")
        print(f"[QA] Max |cluster - (jobs+background)| = {r1:.6f} kW")
        print(f"[QA] Max |campus - (baseline+cluster)| = {r2:.6f} kW")

    print(f"VERDICT: {'GO' if go else 'NO-GO'}")
    return PartAResult(cfg=cfg, slot_ends=slot_ends, merged_on_grid=merged_on_grid, blackouts=blackouts, go=go)

# ------------ CLI ------------
def main(argv: list[str] | None = None):
    import argparse
    ap = argparse.ArgumentParser(description="HPCOpt Part A — Config & Inputs")
    ap.add_argument("--config", "-c", required=True, help="Path to YAML config file")
    args = ap.parse_args(argv)
    _ = run_part_a(args.config)

if __name__ == "__main__":
    main()

