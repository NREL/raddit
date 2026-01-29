import os
import math
import numpy as np
import pandas as pd

# -------------------------------------------------
# Core helpers
# -------------------------------------------------

def _half_up_int(z: float, eps: float) -> int:
    """half-up billed kW = ceil(z - 0.5 + eps)."""
    return int(math.ceil(z - 0.5 + eps))

def _half_step_margin(z_raw: float, eps: float) -> float:
    """
    How far (kW) until the next 0.5kW billing step under half-up.
    Useful for peak-risk reporting.
    """
    k = math.ceil(z_raw - 0.5 + eps)
    thr = k + 0.5 - eps
    return max(0.0, thr - z_raw)

def _month_labels(idx: pd.DatetimeIndex) -> pd.Index:
    """Return month labels 'YYYY-MM' using local timestamps."""
    return pd.Index(idx.strftime("%Y-%m"), name="month")

def _ensure_series_from_cpu15_csv(cpu15_csv: str) -> pd.Series:
    """
    Load the Stage B CPU kW 15-min series from CSV written by:
        res_e_pd.cpu_kw_15.to_frame("scheduled_cpu_kw").to_csv(...)
    This tries common timestamp column names.
    """
    df = pd.read_csv(cpu15_csv)
    # Guess timestamp column
    ts_col = None
    for c in ("slot_end_local", "timestamp", df.columns[0]):
        if c in df.columns:
            try:
                _ = pd.to_datetime(df[c], errors="raise")
                ts_col = c
                break
            except Exception:
                continue
    if ts_col is None:
        raise ValueError(f"Could not find a datetime column in {cpu15_csv}.")

    # Guess value col
    if "scheduled_cpu_kw" in df.columns:
        val_col = "scheduled_cpu_kw"
    else:
        # last column fallback
        val_col = df.columns[-1]

    s = pd.Series(
        pd.to_numeric(df[val_col], errors="coerce").values,
        index=pd.to_datetime(df[ts_col], errors="coerce"),
        name="scheduled_cpu_kw",
    ).dropna()
    # dedupe timestamps just in case
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s

def _align_series_to_index(power_series: pd.Series, target_index: pd.DatetimeIndex) -> np.ndarray:
    """
    Align an arbitrary kW series to the Stage A/B 15-min grid.
    Robust to tz differences, DST gaps (nonexistent times), and repeated times.
    Returns ndarray ordered like target_index.
    """
    s = power_series.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("power_series must be indexed by a DatetimeIndex")

    tz_t = getattr(target_index, "tz", None)
    tz_s = getattr(s.index, "tz", None)

    # --- Coerce series index tz to target tz with DST handling ---
    if tz_t is not None:
        # Target is tz-aware -> localize or convert series to that tz
        if tz_s is None:
            # Naive -> interpret as local wall time in target tz, fixing DST:
            s.index = s.index.tz_localize(
                tz_t,
                nonexistent="shift_forward",   # e.g., 02:00 on spring-forward -> 03:00
                ambiguous="infer"              # fall-back repeated hour, let pandas infer
            )
        elif str(tz_s) != str(tz_t):
            s.index = s.index.tz_convert(tz_t)
    else:
        # Target is tz-naive -> drop tz info without shifting wall times
        if tz_s is not None:
            # Remove tz, preserving wall-clock labels
            s.index = s.index.tz_localize(None)

    # --- 1) exact reindex ---
    aligned = s.reindex(target_index)

    # --- 2) nearest within 8 minutes if sparse ---
    if aligned.isna().mean() > 0.01:
        left = (
            pd.DataFrame({"ts": target_index})
            .sort_values("ts")
            .reset_index(drop=True)
        )
        right = (
            s.rename("kw")
             .rename_axis("ts")
             .reset_index()
             .sort_values("ts")
        )
        merged = pd.merge_asof(
            left, right, on="ts", direction="nearest",
            tolerance=pd.Timedelta("8min")
        )
        aligned = merged.set_index("ts")["kw"]

    return aligned.fillna(0.0).to_numpy(float)

# -------------------------------------------------
# Billing math on a campus net load series
# -------------------------------------------------

def _compute_bill_from_net(
    idx_15: pd.DatetimeIndex,
    net_kw_15: np.ndarray,
    attrs15: pd.DataFrame,
    cfg_solve: dict,
    cfg_billing: dict
):
    """
    Given:
      - idx_15: 15-min slot_ends (tz-aware or naive, same tz as attrs15)
      - net_kw_15: campus net load (kW) array aligned to idx_15
      - attrs15: res_b.attrs aligned to idx_15. Must include:
          tou_price_per_kwh,
          gt_eligible (bool),
          dist_rate_per_kw_month,
          gt_rate_per_kw_month
      - cfg_solve: res_a.cfg["solve"]
      - cfg_billing: res_a.cfg["billing"]

    Returns:
      month_table (DataFrame) and totals ($ energy, $ demand, $ total)
    """
    price15   = attrs15["tou_price_per_kwh"].to_numpy(float)
    gt_mask15 = attrs15["gt_eligible"].to_numpy(bool)
    months15  = _month_labels(idx_15)

    eps  = float(cfg_solve.get("half_up_epsilon_kw", 1e-6))
    dt_h = float(cfg_billing.get("interval_minutes", 15)) / 60.0

    rows = []
    for m in sorted(months15.unique().tolist()):
        sel = (months15 == m)
        if not np.any(sel):
            continue

        # Dist (anytime)
        z_dist = float(np.max(net_kw_15[sel]))
        k_dist = _half_up_int(z_dist, eps)

        # G&T (only GT-eligible slots)
        sel_gt = sel & gt_mask15
        if np.any(sel_gt):
            z_gt = float(np.max(net_kw_15[sel_gt]))
            k_gt = _half_up_int(z_gt, eps)
        else:
            z_gt, k_gt = 0.0, 0

        # pull this month's $/kW rates from first row
        first_row = attrs15.loc[sel].iloc[0]
        dist_rate = float(first_row["dist_rate_per_kw_month"])
        gt_rate   = float(first_row["gt_rate_per_kw_month"])

        # Energy charge: sum(price * kW * hours)
        energy_usd_m = float(np.sum(price15[sel] * net_kw_15[sel] * dt_h))
        demand_usd_m = dist_rate * k_dist + gt_rate * k_gt

        rows.append({
            "month": m,
            "Z_dist_raw_kw": z_dist,
            "K_dist_kw": k_dist,
            "Z_gt_raw_kw": z_gt,
            "K_gt_kw": k_gt,
            "dist_rate_per_kw_month": dist_rate,
            "gt_rate_per_kw_month": gt_rate,
            "energy_cost_month_usd": energy_usd_m,
            "demand_cost_month_usd": demand_usd_m,
            "total_cost_month_usd": energy_usd_m + demand_usd_m,
        })

    month_table = pd.DataFrame(rows).set_index("month").sort_index()
    energy_total = float(month_table["energy_cost_month_usd"].sum())
    demand_total = float(month_table["demand_cost_month_usd"].sum())
    total_cost   = energy_total + demand_total

    return month_table, energy_total, demand_total, total_cost

# -------------------------------------------------
# Stage B costing (same as before)
# -------------------------------------------------

def stageB_costing(
    res_a,
    res_b,
    res_e_pd=None,
    cpu15_csv=None,
    out_dir="../results/hpc_sched_results",
    write_timeseries=True,
    write_peak_risk=True,
    max_job_ids_per_slot=25,   # reserved hook for later (peak-risk active jobs)
):
    """
    Compute Stage B's monthly bill.

    Inputs:
      res_a: PartAResult   (baseline/background + cfg)
      res_b: PartBResult   (attrs with rates & TOU flags)
      res_e_pd OR cpu15_csv: gives scheduled CPU kW / 15min.
    """
    os.makedirs(out_dir, exist_ok=True)

    idx_15   = res_a.slot_ends
    attrs15  = res_b.attrs.loc[idx_15]
    merged15 = res_a.merged_on_grid.loc[idx_15][["baseline_kw", "background_kw"]]

    cfg_solve    = res_a.cfg.get("solve", {})
    cfg_billing  = res_a.cfg.get("billing", {})
    eps          = float(cfg_solve.get("half_up_epsilon_kw", 1e-6))
    dt_h         = float(cfg_billing.get("interval_minutes", 15)) / 60.0
    margin_kw    = float(cfg_solve.get("demand_threshold_margin_kw", 25.0))

    # Get Stage B CPU -> campus net
    if res_e_pd is not None:
        cpu_kw_15 = res_e_pd.cpu_kw_15.reindex(idx_15).fillna(0.0)
    elif cpu15_csv is not None:
        s = _ensure_series_from_cpu15_csv(cpu15_csv)
        cpu_kw_15 = s.reindex(idx_15).fillna(0.0)
    else:
        raise ValueError("Provide res_e_pd OR cpu15_csv.")

    baseline_kw   = merged15["baseline_kw"].to_numpy(float)
    background_kw = merged15["background_kw"].to_numpy(float)
    cpu_arr       = cpu_kw_15.to_numpy(float)

    net_kw_15 = baseline_kw + background_kw + cpu_arr

    # Bill calc
    month_table, e_total, d_total, t_total = _compute_bill_from_net(
        idx_15,
        net_kw_15,
        attrs15,
        cfg_solve,
        cfg_billing,
    )

    # Write Stage B bill
    bill_path = os.path.join(out_dir, "stageB_monthly_bill.csv")
    month_table.to_csv(bill_path)
    print(f"[WRITE] stageB monthly bill -> {bill_path}")

    # Optional per-slot timeseries export
    if write_timeseries:
        price15   = attrs15["tou_price_per_kwh"].to_numpy(float)
        gt_mask15 = attrs15["gt_eligible"].to_numpy(bool)
        ts = pd.DataFrame({
            "timestamp": idx_15,
            "baseline_kw": baseline_kw,
            "background_kw": background_kw,
            "scheduled_cpu_kw": cpu_arr,
            "campus_net_kw": net_kw_15,
            "energy_price_per_kwh": price15,
            "gt_eligible": gt_mask15.astype(int),
        })
        ts_path = os.path.join(out_dir, "stageB_timeseries_with_cost.csv")
        ts.to_csv(ts_path, index=False)
        print(f"[WRITE] stageB timeseries -> {ts_path}")
    else:
        ts_path = None

    # Optional peak-risk report (based on margin to next 0.5 kW step)
    if write_peak_risk:
        price15   = attrs15["tou_price_per_kwh"].to_numpy(float)
        gt_mask15 = attrs15["gt_eligible"].to_numpy(bool)

        margin_dist = np.array([_half_step_margin(v, eps) for v in net_kw_15])
        margin_gt   = np.array([
            _half_step_margin(v, eps) if g else np.nan
            for (v,g) in zip(net_kw_15, gt_mask15)
        ])
        risk_mask = (margin_dist <= margin_kw) | (
            np.isfinite(margin_gt) & (margin_gt <= margin_kw)
        )
        risk_df = pd.DataFrame({
            "timestamp": idx_15[risk_mask],
            "month":     idx_15[risk_mask].strftime("%Y-%m"),
            "campus_net_kw": net_kw_15[risk_mask],
            "margin_to_next_0p5_dist_kw": margin_dist[risk_mask],
            "margin_to_next_0p5_gt_kw":   margin_gt[risk_mask],
            "energy_price_per_kwh": price15[risk_mask],
        })
        peak_risk_path = os.path.join(out_dir, "stageB_peak_risk.csv")
        risk_df.to_csv(peak_risk_path, index=False)
        print(f"[WRITE] stageB peak-risk -> {peak_risk_path}  (rows={len(risk_df):,})")
    else:
        peak_risk_path = None

    print("=== STAGE B COST SUMMARY ===")
    print(f"Energy $ : {e_total:,.2f}")
    print(f"Demand $ : {d_total:,.2f}")
    print(f"TOTAL  $ : {t_total:,.2f}")

    return {
        "month_table": month_table,
        "energy_cost_usd": e_total,
        "demand_cost_usd": d_total,
        "objective_usd": t_total,
        "paths": {
            "monthly_bill_csv": bill_path,
            "timeseries_csv": ts_path,
            "peak_risk_csv": peak_risk_path,
        },
        "idx_15": idx_15,
        "net_kw_15": net_kw_15,
        "attrs15": attrs15,
        "cfg_solve": cfg_solve,
        "cfg_billing": cfg_billing,
    }

# -------------------------------------------------
# Stage A costing (NEW)
# -------------------------------------------------

def stageA_costing(
    res_a,
    res_b,
    res_d,  # StageAResult with y_envelope
    out_dir="../results/hpc_sched_results",
    write_timeseries=True
):
    """
    Compute a full-campus monthly bill using the Stage A envelope (baseline + background + y_envelope).
    """
    os.makedirs(out_dir, exist_ok=True)

    idx_15   = res_a.slot_ends
    attrs15  = res_b.attrs.loc[idx_15]
    merged15 = res_a.merged_on_grid.loc[idx_15][["baseline_kw", "background_kw"]]

    cfg_solve    = res_a.cfg.get("solve", {})
    cfg_billing  = res_a.cfg.get("billing", {})

    baseline_kw   = merged15["baseline_kw"].to_numpy(float)
    background_kw = merged15["background_kw"].to_numpy(float)

    # Stage A envelope (kW at 15-min ends)
    y_env = res_d.y_envelope.reindex(idx_15).fillna(0.0).to_numpy(float)

    net_kw_15 = baseline_kw + background_kw + y_env

    # Bill calc
    month_table, e_total, d_total, t_total = _compute_bill_from_net(
        idx_15,
        net_kw_15,
        attrs15,
        cfg_solve,
        cfg_billing,
    )

    # Write Stage A bill
    bill_path = os.path.join(out_dir, "stageA_monthly_bill.csv")
    month_table.to_csv(bill_path)
    print(f"[WRITE] stageA monthly bill -> {bill_path}")

    # Optional per-slot timeseries export
    if write_timeseries:
        price15   = attrs15["tou_price_per_kwh"].to_numpy(float)
        gt_mask15 = attrs15["gt_eligible"].to_numpy(bool)
        ts = pd.DataFrame({
            "timestamp": idx_15,
            "baseline_kw": baseline_kw,
            "background_kw": background_kw,
            "stageA_y_kw": y_env,
            "campus_net_kw": net_kw_15,
            "energy_price_per_kwh": price15,
            "gt_eligible": gt_mask15.astype(int),
        })
        ts_path = os.path.join(out_dir, "stageA_timeseries_with_cost.csv")
        ts.to_csv(ts_path, index=False)
        print(f"[WRITE] stageA timeseries -> {ts_path}")
    else:
        ts_path = None

    print("=== STAGE A COST SUMMARY ===")
    print(f"Energy $ : {e_total:,.2f}")
    print(f"Demand $ : {d_total:,.2f}")
    print(f"TOTAL  $ : {t_total:,.2f}")

    return {
        "month_table": month_table,
        "energy_cost_usd": e_total,
        "demand_cost_usd": d_total,
        "objective_usd": t_total,
        "paths": {
            "monthly_bill_csv": bill_path,
            "timeseries_csv": ts_path,
        },
        "idx_15": idx_15,
        "net_kw_15": net_kw_15,
        "attrs15": attrs15,
        "cfg_solve": cfg_solve,
        "cfg_billing": cfg_billing,
    }

# -------------------------------------------------
# Actual metered load costing
# -------------------------------------------------

def load_actual_demand_csv(actual_csv_path: str) -> pd.Series:
    """
    Reads your real metered campus data CSV with columns:
        "Interval Start Time", "Round Demand"
    Assumptions:
      - "Interval Start Time" is the START of the interval (e.g. 09/30/19 0:00)
      - Intervals are 15 minutes long.
    We convert that to a trailing-end timestamp index that matches Stage A/B convention:
      end_ts = start_ts + 15min
    We return a Series (kW) indexed by those end timestamps.
    """
    raw = pd.read_csv(actual_csv_path)

    # Parse timestamps
    ts_start = pd.to_datetime(raw["Interval Start Time"], errors="coerce")
    kw = pd.to_numeric(raw["Round Demand"], errors="coerce")

    good = (~ts_start.isna()) & (~kw.isna())
    ts_start = ts_start[good]
    kw       = kw[good]

    # shift start->end of 15-min interval
    end_ts = ts_start + pd.Timedelta("15min")

    s = pd.Series(kw.values, index=end_ts, name="actual_kw").sort_index()
    # Drop dupes if any
    s = s[~s.index.duplicated(keep="last")]
    return s

def actual_costing_from_series(
    actual_kw_series: pd.Series,
    idx_15: pd.DatetimeIndex,
    attrs15: pd.DataFrame,
    cfg_solve: dict,
    cfg_billing: dict,
):
    """
    Line up the real metered kW ('actual_kw_series') with Stage A/B 15-min grid idx_15,
    then compute a bill for that aligned load.
    """
    # Align to model grid (nearest within 8 minutes fallback)
    actual_arr = _align_series_to_index(actual_kw_series, idx_15)

    # Bill calc on actual campus net
    month_table, e_total, d_total, t_total = _compute_bill_from_net(
        idx_15,
        actual_arr,
        attrs15,
        cfg_solve,
        cfg_billing,
    )

    return {
        "month_table": month_table,
        "energy_cost_usd": e_total,
        "demand_cost_usd": d_total,
        "objective_usd": t_total,
        "aligned_kw": actual_arr,
    }

# -------------------------------------------------
# Wrapper: compare Stage A + Stage B vs actual
# -------------------------------------------------

def compare_stageA_B_actual(
    res_a,
    res_b,
    res_d=None,          # StageAResult; if None, Stage A section is skipped
    res_e_pd=None,
    cpu15_csv=None,
    actual_csv_path=None,
    out_dir="../results/hpc_sched_results"
):
    """
    1) Compute Stage B bill (scheduled baseline+background+CPU).
    2) Optionally compute Stage A bill (baseline+background+y_envelope).
    3) Optionally compute Actual bill from metered CSV.
    4) Print totals and write a combined per-month CSV.
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- Stage B
    sb = stageB_costing(
        res_a,
        res_b,
        res_e_pd=res_e_pd,
        cpu15_csv=cpu15_csv,
        out_dir=out_dir,
        write_timeseries=True,
        write_peak_risk=True,
    )

    # --- Stage A (optional)
    sa = None
    if res_d is not None:
        sa = stageA_costing(
            res_a,
            res_b,
            res_d,
            out_dir=out_dir,
            write_timeseries=True
        )

    # --- Actual (optional)
    actual = None
    if actual_csv_path is not None:
        actual_series = load_actual_demand_csv(actual_csv_path)
        actual = actual_costing_from_series(
            actual_series,
            idx_15=sb["idx_15"],
            attrs15=sb["attrs15"],
            cfg_solve=sb["cfg_solve"],
            cfg_billing=sb["cfg_billing"],
        )

    # --- Print comparison
    print("\n=== COST TOTALS ===")
    print(f"Stage B Energy $ : {sb['energy_cost_usd']:,.2f}")
    print(f"Stage B Demand $ : {sb['demand_cost_usd']:,.2f}")
    print(f"Stage B TOTAL  $ : {sb['objective_usd']:,.2f}")

    if sa is not None:
        print(f"\nStage A Energy $ : {sa['energy_cost_usd']:,.2f}")
        print(f"Stage A Demand $ : {sa['demand_cost_usd']:,.2f}")
        print(f"Stage A TOTAL  $ : {sa['objective_usd']:,.2f}")
        print(f"\nΔ (B - A) Energy $ : {sb['energy_cost_usd'] - sa['energy_cost_usd']:,.2f}")
        print(f"Δ (B - A) Demand $ : {sb['demand_cost_usd'] - sa['demand_cost_usd']:,.2f}")
        print(f"Δ (B - A) TOTAL  $ : {sb['objective_usd'] - sa['objective_usd']:,.2f}")

    if actual is not None:
        print(f"\nActual  Energy $ : {actual['energy_cost_usd']:,.2f}")
        print(f"Actual  Demand $ : {actual['demand_cost_usd']:,.2f}")
        print(f"Actual  TOTAL  $ : {actual['objective_usd']:,.2f}")
        print(f"\nΔ (B - Actual) Energy $ : {sb['energy_cost_usd'] - actual['energy_cost_usd']:,.2f}")
        print(f"Δ (B - Actual) Demand $ : {sb['demand_cost_usd'] - actual['demand_cost_usd']:,.2f}")
        print(f"Δ (B - Actual) TOTAL  $ : {sb['objective_usd'] - actual['objective_usd']:,.2f}")
        if sa is not None:
            print(f"\nΔ (A - Actual) Energy $ : {sa['energy_cost_usd'] - actual['energy_cost_usd']:,.2f}")
            print(f"Δ (A - Actual) Demand $ : {sa['demand_cost_usd'] - actual['demand_cost_usd']:,.2f}")
            print(f"Δ (A - Actual) TOTAL  $ : {sa['objective_usd'] - actual['objective_usd']:,.2f}")

    # --- Combined per-month CSV
    mt = sb["month_table"].add_prefix("stageB_")
    if sa is not None:
        mt = mt.join(sa["month_table"].add_prefix("stageA_"), how="outer")
    if actual is not None:
        mt = mt.join(actual["month_table"].add_prefix("actual_"), how="outer")

    combined_path = os.path.join(out_dir, "stageA_stageB_actual_monthly.csv")
    mt.to_csv(combined_path)
    print(f"\n[WRITE] stageA_stageB_actual_monthly -> {combined_path}")
    print("\n[PER-MONTH BREAKDOWN]")
    print(mt.to_string(float_format=lambda v: f"{v:,.2f}"))

    return sa, sb, actual

# -------------------------------------------------
# Backwards-compatible wrapper name (optional)
# -------------------------------------------------
def compare_stageB_to_actual(
    res_a,
    res_b,
    res_e_pd=None,
    cpu15_csv=None,
    actual_csv_path=None,
    out_dir="../results/hpc_sched_results",
    res_d=None,   # allow passing StageAResult here too
):
    """
    Kept for backwards compatibility; now also accepts res_d to compute Stage A.
    """
    return compare_stageA_B_actual(
        res_a=res_a,
        res_b=res_b,
        res_d=res_d,
        res_e_pd=res_e_pd,
        cpu15_csv=cpu15_csv,
        actual_csv_path=actual_csv_path,
        out_dir=out_dir
    )

# -------------------------------------------------
# How to run in the notebook (examples)
# -------------------------------------------------
# Example 1: live objects res_a, res_b, res_d, res_e_pd, and an "actuals" CSV.
#
# sa_res, sb_res, actual_res = compare_stageA_B_actual(
#     res_a,
#     res_b,
#     res_d=res_d,
#     res_e_pd=res_e_pd,
#     actual_csv_path="../data/actual_metered_demand.csv",
#     out_dir="../results/hpc_sched_results"
# )
#
# Example 2: you don't have res_e_pd in memory, but you do have its CSV.
#
# sa_res, sb_res, actual_res = compare_stageA_B_actual(
#     res_a,
#     res_b,
#     res_d=res_d,
#     cpu15_csv="../results/hpc_sched_results/stageB_pd_cpu15.csv",
#     actual_csv_path="../data/actual_metered_demand.csv",
#     out_dir="../results/hpc_sched_results"
# )
