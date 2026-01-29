#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
from zoneinfo import ZoneInfo

# ---------- Config ----------
DEMAND_CSV  = Path("../data/STM_net_elec_interval.csv")
CLUSTER_CSV = Path("../../influx/data/cluster_kw.csv")
OUTPUT_CSV  = Path("../data/campus_cluster_demand.csv")
LOCAL_TZ    = ZoneInfo("America/Denver")
# ----------------------------

def read_demand(path: Path) -> pd.DataFrame:
    """
    Parse 'Interval Start Time' as America/Denver local (naive -> aware),
    handle DST fall-back robustly, then create a UTC-minute merge key.
    Returns: ['ts_utc_min','display_ts','round_demand_kw']
    """
    df = pd.read_csv(path)
    if "Interval Start Time" not in df.columns or "Round Demand" not in df.columns:
        raise ValueError("Expected columns: 'Interval Start Time' and 'Round Demand'")

    # Parse explicit M/D/YY H:MM
    ts_naive = pd.to_datetime(df["Interval Start Time"], format="%m/%d/%y %H:%M", errors="coerce")
    if ts_naive.isna().any():
        bad = df[ts_naive.isna()].index[:5].tolist()
        raise ValueError(f"Failed to parse some 'Interval Start Time' values; example bad rows: {bad}")

    # Sort to help inference across DST boundaries
    df = df.assign(_ts_naive=ts_naive).sort_values("_ts_naive").reset_index(drop=True)

    # Try to localize; fall back to explicit disambiguation for repeated "fall back" minutes
    try:
        ts_local = df["_ts_naive"].dt.tz_localize(LOCAL_TZ, ambiguous="infer", nonexistent="shift_forward")
    except Exception:
        key = df["_ts_naive"].dt.strftime("%Y-%m-%d %H:%M")
        counts = key.map(key.value_counts())
        pos = key.groupby(key).cumcount()
        ambiguous_flags = (counts > 1) & (pos == 0)  # first duplicate minute = DST, second = standard
        ts_local = df["_ts_naive"].dt.tz_localize(LOCAL_TZ, ambiguous=ambiguous_flags.values, nonexistent="shift_forward")

    # Stable merge key: round in UTC to avoid DST ambiguity
    ts_utc_min = ts_local.dt.tz_convert("UTC").dt.floor("min")
    display_ts = ts_utc_min.dt.tz_convert(LOCAL_TZ)

    df = df.rename(columns={"Round Demand": "round_demand_kw"})
    df["round_demand_kw"] = pd.to_numeric(df["round_demand_kw"], errors="coerce")

    return pd.DataFrame({
        "ts_utc_min": ts_utc_min,
        "display_ts": display_ts,
        "round_demand_kw": df["round_demand_kw"]
    })

def read_cluster(path: Path) -> pd.DataFrame:
    """
    Read cluster CSV with 'timestamp' (offset) and 'cluster_kw'.
    Returns: ['ts_utc_min','cluster_kw']
    """
    df = pd.read_csv(path)
    if "timestamp" not in df.columns or "cluster_kw" not in df.columns:
        raise ValueError("Expected columns 'timestamp' and 'cluster_kw' in cluster CSV")

    # Parse mixed offsets to UTC
    ts_raw = df["timestamp"].astype(str).str.strip()
    ts = pd.to_datetime(ts_raw, errors="coerce", utc=True)
    if ts.isna().any():
        bad = df[ts.isna()].index[:5].tolist()
        raise ValueError(f"Failed to parse some 'timestamp' values in cluster CSV; example bad rows: {bad}")

    df = df.assign(ts=ts).sort_values("ts").reset_index(drop=True)
    df["cluster_kw"] = pd.to_numeric(df["cluster_kw"], errors="coerce")

    ts_utc_min = df["ts"].dt.floor("min")  # already UTC

    return pd.DataFrame({
        "ts_utc_min": ts_utc_min,
        "cluster_kw": df["cluster_kw"]
    })

def main():
    demand = read_demand(DEMAND_CSV)
    cluster = read_cluster(CLUSTER_CSV)

    # Left join on UTC-minute key
    merged = demand.merge(cluster, on="ts_utc_min", how="left")

    # Local display timestamp with offset (minute precision)
    merged["timestamp"] = merged["display_ts"].map(lambda x: x.isoformat(sep=" ", timespec="minutes"))

    out = merged[["timestamp", "campus_kw", "cluster_kw"]].sort_values("timestamp")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(out):,} rows to {OUTPUT_CSV.resolve()}")

if __name__ == "__main__":
    main()

