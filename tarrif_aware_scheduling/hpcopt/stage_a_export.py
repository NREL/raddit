# hpcopt/stage_a_export.py
from __future__ import annotations
import os
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple

from .part_a import PartAResult
from .part_b import PartBResult
from .stage_a_lp import StageAResult

def _next_half_up_threshold(k_int: int, eps: float) -> float:
    """
    For half-up billing, after billing at integer k, the *next* increment happens at k+0.5-eps.
    """
    return (k_int + 0.5 - eps)

@dataclass
class ExportResult:
    features: pd.DataFrame
    month_caps: pd.DataFrame
    go: bool
    out_features_csv: str | None
    out_caps_csv: str | None

def export_stage_a_artifacts(
    part_a: PartAResult,
    part_b: PartBResult,
    stage_a: StageAResult,
    out_dir: str | None = None,
    eps: float = 1e-6,
) -> ExportResult:
    """
    Build per-slot features for the packer and month caps from Stage A outputs.
    Optionally write CSVs to out_dir.
    """
    idx = part_a.slot_ends
    attrs = part_b.attrs.loc[idx]
    merged = part_a.merged_on_grid.loc[idx]  # baseline/background on the grid

    # Envelope
    y = stage_a.y_envelope.reindex(idx).fillna(0.0)  # kW
    # Month caps table
    caps = stage_a.month_table.copy()

    # Map months for each slot (local)
    month_labels = pd.Index(idx.strftime("%Y-%m"), name="month")
    caps = caps.copy()
    # Build per-month cap thresholds for DIST and GT (next half step to trigger higher billing)
    caps["dist_cap_next_half_kw"] = caps["K_dist_kw"].astype(int).apply(lambda k: _next_half_up_threshold(k, eps))
    caps["gt_cap_next_half_kw"]   = caps["K_gt_kw"].astype(int).apply(lambda k: _next_half_up_threshold(k, eps))

    # Broadcast per-month thresholds to slots
    dist_cap_next = pd.Series(index=idx, dtype=float)
    gt_cap_next   = pd.Series(index=idx, dtype=float)
    for m in caps.index:
        mask = (month_labels == m)
        dist_cap_next.loc[mask] = caps.loc[m, "dist_cap_next_half_kw"]
        gt_cap_next.loc[mask]   = caps.loc[m, "gt_cap_next_half_kw"]

    # Compose features DF
    slot_start = idx - pd.Timedelta(minutes=int(part_a.cfg["billing"]["interval_minutes"]))
    feats = pd.DataFrame({
        "slot_end_local": idx,
        "slot_start_local": slot_start,
        "y_kw": y.values,
        "baseline_kw": merged["baseline_kw"].values,
        "background_kw": merged["background_kw"].values,
        "tou_price_per_kwh": attrs["tou_price_per_kwh"].values,
        "gt_eligible": attrs["gt_eligible"].values,
        "month": month_labels.values,
        "dist_cap_next_half_kw": dist_cap_next.values,
        "gt_cap_next_half_kw": gt_cap_next.values,
    }, index=idx)

    # Net target (for intuition; packer will compute actual with scheduled load)
    feats["target_net_kw"] = feats["baseline_kw"] + feats["background_kw"] + feats["y_kw"]

    # Risk to next 0.5-kW threshold (positive = safe margin; negative = already over)
    feats["risk_to_dist_cap_kw"] = feats["dist_cap_next_half_kw"] - feats["target_net_kw"]
    feats["risk_to_gt_cap_kw"]   = np.where(
        feats["gt_eligible"],
        feats["gt_cap_next_half_kw"] - feats["target_net_kw"],
        np.inf  # not applicable outside GT window
    )

    # Clip tiny numerical noise
    feats["risk_to_dist_cap_kw"] = feats["risk_to_dist_cap_kw"].round(6)
    feats["risk_to_gt_cap_kw"]   = feats["risk_to_gt_cap_kw"].replace(np.inf, 1e12).round(6).replace(1e12, np.inf)

    # Mrgins from month table should be in [0, 0.5]
    ok_margins = (
        (caps["margin_to_next_0p5_dist_kw"].fillna(0) >= -1e-5) &
        (caps["margin_to_next_0p5_dist_kw"].fillna(0) <= 0.5 + 1e-3) &
        (caps["margin_to_next_0p5_gt_kw"].fillna(0)   >= -1e-5) &
        (caps["margin_to_next_0p5_gt_kw"].fillna(0)   <= 0.5 + 1e-3)
    )
    go = bool(ok_margins.all())

    out_features_csv = None
    out_caps_csv = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_features_csv = os.path.join(out_dir, "envelope_features.csv")
        out_caps_csv     = os.path.join(out_dir, "month_caps.csv")
        feats.to_csv(out_features_csv, index=False)
        caps.to_csv(out_caps_csv)

    # Prints (go/no-go)
    print("=== STAGE A EXPORT: GO/NO-GO ===")
    print(f"[GO] features rows: {len(feats):,}  cols: {len(feats.columns)}")
    print(f"[GO] caps rows    : {len(caps):,}")
    print(f"[SAMPLE features]")
    print(feats.head(3).to_string(index=False))
    print(f"[SAMPLE caps]")
    print(caps.head(3).to_string())
    if not go:
        bad_m = caps.loc[~ok_margins].index.tolist()[:5]
        print(f"[WARN] Some month margins out of expected [0,0.5] range. Sample months: {bad_m}")
    print(f"VERDICT: {'GO' if go else 'CHECK'} (export complete)")
    return ExportResult(features=feats, month_caps=caps, go=go, out_features_csv=out_features_csv, out_caps_csv=out_caps_csv)
