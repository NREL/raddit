# hpcopt/part_c.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

from .part_a import PartAResult

# ----------------- helpers -----------------

def _require_cols(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name}: missing required columns: {missing}")

def _to_local_ts(s: pd.Series, tz: str) -> pd.Series:
    """
    Parse timestamps; if naive, localize to tz; if offset-aware, keep instant and convert to tz.
    Empty strings/NaN -> NaT.
    """
    out = pd.to_datetime(s.replace({"": np.nan}), errors="coerce")
    if out.dt.tz is None:
        # localize naive; do not convert
        out = out.dt.tz_localize(tz)
    else:
        # convert to target tz to keep absolute instants comparable
        out = out.dt.tz_convert(tz)
    return out

def _load_nodes(nodes_csv: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not nodes_csv or not os.path.exists(nodes_csv):
        raise FileNotFoundError(f"nodes CSV not found: {nodes_csv}")
    raw = pd.read_csv(nodes_csv)
    _require_cols(raw, ["type_id", "class", "count"], "nodes_csv")

    # Keep cpu_exclusive only
    inv = raw.loc[raw["class"].astype(str).str.lower() == "cpu_exclusive"].copy()
    inv.rename(columns={"type_id": "node_type"}, inplace=True)
    inv["node_type"] = inv["node_type"].astype(str)
    inv["count"] = pd.to_numeric(inv["count"], errors="coerce").fillna(0).astype(int)
    inv = inv.loc[inv["count"] > 0].reset_index(drop=True)

    info = {
        "types": inv["node_type"].nunique(),
        "total_nodes": int(inv["count"].sum()),
        "by_type": inv.groupby("node_type")["count"].sum().to_dict()
    }
    if inv.empty:
        raise ValueError("nodes_csv after filtering for class=='cpu_exclusive' is empty.")
    return inv, info

def _load_jobs_specific(jobs_csv: str, interval_minutes: int, tz: str,
                        default_node_type: Optional[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Schema expected:
      job_id,duration_slots,avg_power_kw,nodes,node_type,release_ts,deadline_ts
    - duration_slots is a real number of *billing slots*; convert using interval_minutes.
    - avg_power_kw is for the WHOLE JOB (energy / time), not per-node.
    """
    if not jobs_csv or not os.path.exists(jobs_csv):
        raise FileNotFoundError(f"jobs CSV not found: {jobs_csv}")
    df = pd.read_csv(jobs_csv)
    _require_cols(df, ["job_id","duration_slots","avg_power_kw","nodes","node_type","release_ts","deadline_ts"], "jobs_csv")

    # Duration (seconds) from slots
    dur_slots = pd.to_numeric(df["duration_slots"], errors="coerce")
    dur_sec = dur_slots * (interval_minutes * 60.0)

    # Power (kW) is whole-job average (already energy/time)
    power_kw = pd.to_numeric(df["avg_power_kw"], errors="coerce")

    # Nodes
    nodes = pd.to_numeric(df["nodes"], errors="coerce").astype("Int64")

    # Node type (fill with default if missing/NA and only one type exists)
    node_type = df["node_type"].astype("string")
    if default_node_type is not None:
        node_type = node_type.fillna(default_node_type).replace({"": default_node_type})

    # Optional releases/deadlines
    release_ts = _to_local_ts(df["release_ts"], tz)
    deadline_ts = _to_local_ts(df["deadline_ts"], tz)

    # Build clean table
    jobs = pd.DataFrame({
        "job_id": df["job_id"].astype(str),
        "duration_seconds": dur_sec.astype(float),
        "avg_power_kw": power_kw.astype(float),
        "nodes": nodes.astype("int64"),
        "node_type": node_type.astype(str),
        "release_ts": release_ts,
        "deadline_ts": deadline_ts
    })

    # Derived energy (kWh)
    jobs["energy_kwh"] = jobs["avg_power_kw"] * (jobs["duration_seconds"] / 3600.0)

    # Validity (strict)
    valid = (
        jobs["duration_seconds"].notna() & (jobs["duration_seconds"] > 0) &
        jobs["avg_power_kw"].notna() & (jobs["avg_power_kw"] >= 0) &
        jobs["nodes"].notna() & (jobs["nodes"] >= 1) &
        jobs["node_type"].astype(str).str.len().gt(0)
    )
    jobs["valid"] = valid

    info = {
        "raw_rows": len(df),
        "sched_rows": int(valid.sum()),
        "total_energy_kwh_sched": float(jobs.loc[valid, "energy_kwh"].sum()),
        "min_duration_s": float(jobs.loc[valid, "duration_seconds"].min()) if valid.any() else np.nan,
        "max_duration_s": float(jobs.loc[valid, "duration_seconds"].max()) if valid.any() else np.nan,
        "min_power_kw": float(jobs.loc[valid, "avg_power_kw"].min()) if valid.any() else np.nan,
        "max_power_kw": float(jobs.loc[valid, "avg_power_kw"].max()) if valid.any() else np.nan,
    }
    # Keep only valid rows for scheduling
    jobs_sched = jobs.loc[valid].reset_index(drop=True)
    return jobs_sched, info

# ----------------- result container -----------------

@dataclass
class PartCResult:
    jobs_sched: pd.DataFrame    # job_id,duration_seconds,avg_power_kw,nodes,node_type,release_ts,deadline_ts,energy_kwh
    nodes_inv: pd.DataFrame     # node_type,count
    go: bool

# ----------------- main API -----------------

def run_part_c(part_a: PartAResult) -> PartCResult:
    cfg = part_a.cfg
    billing  = cfg.get("billing", {})
    policies = cfg.get("policies", {})
    paths    = cfg.get("paths", {})

    tz = cfg.get("calendar", {}).get("timezone", "America/Denver")
    interval_minutes = int(billing.get("interval_minutes", 15))
    allow_any_type_if_unset = bool(policies.get("allow_any_type_if_unset", True))

    jobs_csv  = paths.get("jobs_csv")
    nodes_csv = paths.get("nodes_csv")

    # Load nodes (cpu_exclusive only)
    nodes_inv, nodes_info = _load_nodes(nodes_csv)

    # If there's exactly one node type, use it as default for missing job types
    default_type = nodes_inv["node_type"].iloc[0] if nodes_inv["node_type"].nunique() == 1 else (None if allow_any_type_if_unset else None)

    # Load jobs per the provided schema
    jobs_sched, jobs_info = _load_jobs_specific(jobs_csv, interval_minutes, tz, default_type)

    # Validate that all job node_types exist in inventory
    inv_types = set(nodes_inv["node_type"].unique())
    job_types = set(jobs_sched["node_type"].unique())
    missing_types = sorted(list(job_types - inv_types))

    # ------------- GO/NO-GO prints -------------
    print("=== PART C: GO/NO-GO CHECKS ===")
    print(f"[GO] Raw jobs rows              : {jobs_info['raw_rows']:,}")
    print(f"[GO] Schedulable CPU-only rows  : {jobs_info['sched_rows']:,}")
    print(f"[GO] Energy (sched jobs)        : {jobs_info['total_energy_kwh_sched']:.6f} kWh")
    print(f"[GO] Duration range (s)         : {jobs_info['min_duration_s']} → {jobs_info['max_duration_s']}")
    print(f"[GO] Power range (kW)           : {jobs_info['min_power_kw']} → {jobs_info['max_power_kw']}")
    print(f"[GO] Nodes inventory types      : {nodes_info['types']}  total nodes: {nodes_info['total_nodes']}")
    print(f"[GO] Nodes by type              : {nodes_info['by_type']}")

    if missing_types:
        print(f"[NO-GO] Job node_type(s) not in inventory: {missing_types}")

    # Samples
    print("[SAMPLE] Jobs (first 3):")
    print(jobs_sched.head(3).to_string(index=False))
    print("[SAMPLE] Nodes inventory:")
    print(nodes_inv.to_string(index=False))

    go = (len(jobs_sched) > 0) and (len(missing_types) == 0)
    print(f"VERDICT: {'GO' if go else 'NO-GO'}")

    # Keep only the columns we need downstream (neatly ordered)
    jobs_sched = jobs_sched[[
        "job_id","duration_seconds","avg_power_kw","nodes","node_type","release_ts","deadline_ts","energy_kwh"
    ]]
    nodes_inv = nodes_inv[["node_type","count"]].reset_index(drop=True)

    return PartCResult(jobs_sched=jobs_sched, nodes_inv=nodes_inv, go=go)

# Optional CLI for Part C
def main(argv=None):
    import argparse
    from .part_a import run_part_a
    ap = argparse.ArgumentParser(description="HPCOpt Part C — Jobs & Nodes ingest (specific schema)")
    ap.add_argument("--config", "-c", required=True, help="Path to YAML config")
    args = ap.parse_args(argv)
    res_a = run_part_a(args.config)
    _ = run_part_c(res_a)

if __name__ == "__main__":
    main()
