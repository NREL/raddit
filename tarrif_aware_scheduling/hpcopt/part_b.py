# hpcopt/part_b.py
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Literal, Dict, Any, Tuple, List

from .part_a import PartAResult  # uses the earlier result type

# ----------------- helpers -----------------

def _parse_local_time_mm(str_hhmm: str) -> int:
    """'HH:MM' -> minutes since midnight."""
    hh, mm = str_hhmm.strip().split(":")
    return int(hh) * 60 + int(mm)

def _in_window(mins_local: np.ndarray, start_min: int, end_min: int) -> np.ndarray:
    """
    Return boolean mask for time-of-day membership in [start, end) (no wrap).
    Assumes windows don't cross midnight (true for our tariffs).
    """
    return (mins_local >= start_min) & (mins_local < end_min)

def _build_holiday_set(cfg_calendar: Dict[str, Any]) -> set:
    hol = cfg_calendar.get("holidays", []) or []
    # Normalize to date() objects (naive local date strings are fine)
    out = set()
    for d in hol:
        # Accept forms 'YYYY-MM-DD' or Timestamp-like; keep only the date
        dt = pd.to_datetime(str(d), errors="coerce")
        if pd.isna(dt):
            continue
        out.add(dt.date())
    return out

# ----------------- main API -----------------

@dataclass
class PartBResult:
    attrs: pd.DataFrame
    go: bool

def run_part_b(part_a: PartAResult) -> PartBResult:
    """
    Build per-slot attributes for the 15-min trailing grid:
      - weekday/weekend/holiday flags (based on slot START local time)
      - TOU price and label
      - G&T eligibility and $/kW by month (seasonal)
      - Distribution $/kW-month
    """
    cfg = part_a.cfg
    slot_ends = part_a.slot_ends  # local tz
    if len(slot_ends) == 0:
        raise ValueError("PartB: empty slot grid from Part A.")

    calendar = cfg.get("calendar", {})
    rates    = cfg.get("rates", {})
    billing  = cfg.get("billing", {})
    policies = cfg.get("policies", {})

    tz = calendar.get("timezone", "America/Denver")
    holidays_as_weekends = bool(policies.get("holidays_treated_like_weekends", True))
    holiday_set = _build_holiday_set(calendar)

    # Effective start (local)
    eff = rates.get("effective_start_local")
    eff_local = pd.to_datetime(eff, errors="coerce")
    if eff_local.tzinfo is None and not pd.isna(eff_local):
        # Assume the calendar tz if naive
        eff_local = eff_local.tz_localize(tz)

    # --- Build slot starts in local tz (trailing windows)
    slot_starts = slot_ends - pd.Timedelta(minutes=int(billing.get("interval_minutes", 15)))

    # Local calendar tags from START instant
    start_local = pd.Series(slot_starts)
    # Compute weekday/weekend/holiday based on local calendar at slot start
    dow = start_local.dt.weekday  # Mon=0
    is_weekend = (dow >= 5).to_numpy()
    # Holiday by local date
    is_holiday = start_local.dt.date.map(lambda d: d in holiday_set).to_numpy()

    # Weekday/non-holiday indicator for tariff windows
    is_weekday = ~is_weekend
    if holidays_as_weekends:
        is_weekday_non_holiday = is_weekday & (~is_holiday)
    else:
        # If not treating holidays like weekends, still exclude holidays when the window
        # explicitly says "mon-fri_non_holiday".
        is_weekday_non_holiday = is_weekday & (~is_holiday)

    # Minutes since midnight (local) at slot start
    mins_local = (start_local.dt.hour * 60 + start_local.dt.minute).to_numpy()

    # --- TOU pricing
    tou = rates.get("tou_energy", {})
    onpeak = tou.get("onpeak_weekday_day", {})
    # TODO: Example times. Should pull these times from config
    on_start = _parse_local_time_mm(onpeak.get("start_local", "10:00"))
    on_end   = _parse_local_time_mm(onpeak.get("end_local",   "20:00"))
    on_price = float(onpeak.get("price_per_kwh", np.nan))

    # off-peak (all other times)
    offpeak = tou.get("offpeak_other", {})
    off_price = float(offpeak.get("price_per_kwh", np.nan))

    is_onpeak = is_weekday_non_holiday & _in_window(mins_local, on_start, on_end)
    tou_price = np.where(is_onpeak, on_price, off_price)
    tou_label = np.where(is_onpeak, "onpeak_weekday_day", "offpeak_other")

    # --- G&T eligibility
    dem = rates.get("demand", {})
    gt_win = (dem.get("windows", {}) or {}).get("gt", {})
    # TODO: Example times. Should pull these times from config
    gt_start = _parse_local_time_mm(gt_win.get("start_local", "12:00"))
    gt_end   = _parse_local_time_mm(gt_win.get("end_local",   "15:00"))
    gt_eligible = is_weekday_non_holiday & _in_window(mins_local, gt_start, gt_end)

    # --- $/kW rates by month (seasonal)
    seasons = dem.get("seasons", {"winter_months":[10,11,12,1,2,3,4,5], "summer_months":[6,7,8,9]})
    gt_rates = dem.get("gt_per_kw_month", {})
    gt_winter = float(gt_rates.get("winter", np.nan))
    gt_summer = float(gt_rates.get("summer", np.nan))
    dist_rate = float(dem.get("distribution_per_kw_month", np.nan))

    months = start_local.dt.month.to_numpy()
    season_labels = np.array(["summer"] * len(months), dtype=object)
    season_labels[np.isin(months, seasons.get("winter_months", []))] = "winter"
    season_labels[np.isin(months, seasons.get("summer_months", []))] = "summer"

    gt_rate = np.where(season_labels == "winter", gt_winter, gt_summer)

    # --- Effective date mask (warn if any slots precede effective date)
    pre_eff = np.zeros(len(slot_ends), dtype=bool)
    if eff_local is not None and not pd.isna(eff_local):
        # Ensure both sides are comparable in the same tz (should already be true)
        if getattr(eff_local, "tz", None) is not None and slot_ends.tz is not None:
            # (No-op in most cases; both are local tz already)
            pass
        # Comparison yields a NumPy boolean array directly
        pre_eff = np.asarray(slot_ends < eff_local, dtype=bool)


    # --- Build attributes DF
    attrs = pd.DataFrame({
        "slot_start_local": slot_starts,
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "is_weekday_non_holiday": is_weekday_non_holiday,
        "tou_label": tou_label,
        "tou_price_per_kwh": tou_price,
        "gt_eligible": gt_eligible,
        "gt_rate_per_kw_month": gt_rate,
        "dist_rate_per_kw_month": dist_rate,
        "season": season_labels,
        "pre_effective_rates": pre_eff,
    }, index=slot_ends)
    attrs.index.name = "slot_end_local"

    # ---------------- GO/NO-GO checks & prints ----------------
    total = len(attrs)
    n_on  = int(is_onpeak.sum())
    n_off = total - n_on
    n_gt  = int(gt_eligible.sum())

    # Sanity: no NaNs in prices or rates
    nan_tou = int(np.isnan(attrs["tou_price_per_kwh"]).sum())
    nan_gt  = int(np.isnan(attrs["gt_rate_per_kw_month"]).sum())
    nan_dist= int(np.isnan(attrs["dist_rate_per_kw_month"]).sum())

    # Season split
    n_w = int((attrs["season"] == "winter").sum())
    n_s = int((attrs["season"] == "summer").sum())

    print("=== PART B: GO/NO-GO CHECKS ===")
    print(f"[GO] Slots: {total:,} ({attrs.index.min()} → {attrs.index.max()}) tz={attrs.index.tz}")
    print(f"[GO] On-peak slots: {n_on:,}  | Off-peak slots: {n_off:,}")
    print(f"[GO] G&T-eligible slots: {n_gt:,}")
    print(f"[GO] Season split: winter={n_w:,}, summer={n_s:,}")
    print(f"[GO] Rate params: dist=${dist_rate:.2f}/kW-mo, GT(winter)=${gt_winter:.2f}, GT(summer)=${gt_summer:.2f}")
    if eff_local is not None and not pd.isna(eff_local):
        n_pre = int(pre_eff.sum())
        print(f"[INFO] Slots before effective_start_local={eff_local}: {n_pre}")
    # NaN checks
    if nan_tou == 0 and nan_gt == 0 and nan_dist == 0:
        print("[GO] No NaNs in TOU or demand rates.")
        go = True
    else:
        print(f"[NO-GO] NaNs detected: tou={nan_tou}, gt_rate={nan_gt}, dist_rate={nan_dist}")
        go = False

    # Sample lines for visual inspection
    print("[SAMPLE] First 3 rows:")
    print(attrs.head(3).to_string())

    return PartBResult(attrs=attrs, go=go)

# Optional CLI
def main(argv=None):
    import argparse
    from .part_a import run_part_a
    ap = argparse.ArgumentParser(description="HPCOpt Part B — Calendar & slot attributes")
    ap.add_argument("--config", "-c", required=True, help="Path to YAML config")
    args = ap.parse_args(argv)
    res_a = run_part_a(args.config)
    _ = run_part_b(res_a)

if __name__ == "__main__":
    main()
