# hpcopt/stage_a_plots.py
from __future__ import annotations
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .part_a import PartAResult
from .part_b import PartBResult
from .stage_a_lp import StageAResult

def plot_stage_a_timeseries(
    res_a: PartAResult,
    res_b: PartBResult,
    res_d: StageAResult,
    out_path: Optional[str] = "../results/stageA_y_kw.png",
    save_csv: Optional[str] = None,
    show: bool = False,
    shade_gt: bool = False,         # set True to lightly shade G&T-eligible spans
) -> pd.Series:
    """
    Plot the Stage-A envelope power y(t) [kW] over time (15-min trailing labels).
    Saves a PNG (and optional CSV). Returns the y(t) series.
    """
    # --- Pull series (15-min trailing labels, timezone-aware) ---
    y = res_d.y_envelope.copy()  # pd.Series indexed by slot_end_local
    idx = y.index

    # Basic checks
    assert isinstance(idx, pd.DatetimeIndex), "y_envelope index must be a DatetimeIndex"
    assert y.notna().all(), "y_envelope contains NaNs"

    # Energy implied by y(t) on 15-min grid
    energy_kwh = float((y * 0.25).sum())

    # --- GO/NO-GO prints ---
    print("=== STAGE A: y(t) PLOT — GO/NO-GO ===")
    print(f"[GO] y(t) points: {len(y):,}  range: {idx.min()} → {idx.max()}  tz={idx.tz}")
    print(f"[GO] Energy from y(t): {energy_kwh:,.3f} kWh")
    print(f"[GO] y(t) stats (kW): min={y.min():.3f}  p50={y.median():.3f}  max={y.max():.3f}")

    # --- Build plot ---
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(idx, y.to_numpy(float), linewidth=0.8)
    ax.set_title("Stage A envelope power y(t) [kW] (15-min trailing)")
    ax.set_xlabel("Time (slot end)")
    ax.set_ylabel("kW")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)

    # Vertical lines at month boundaries
    months = idx.to_period("M")
    boundaries = np.flatnonzero(months[1:].values != months[:-1].values)  # indices where month changes (offset by +1)
    for b in boundaries:
        ax.axvline(idx[b+1], linestyle="--", linewidth=0.6, alpha=0.6)

    # Lightly shade G&T-eligible minutes (broadcast from 15-min attrs)
    if shade_gt:
        attrs15 = res_b.attrs.loc[idx]
        gt_mask = attrs15["gt_eligible"].to_numpy(bool)
        # compact adjacent True runs to fewer spans
        if gt_mask.any():
            starts = np.flatnonzero(gt_mask & ~np.roll(gt_mask, 1))
            ends   = np.flatnonzero(gt_mask & ~np.roll(gt_mask, -1))
            # handle edge roll artifacts
            if gt_mask[0]:  starts = np.insert(starts, 0, 0)
            if gt_mask[-1]: ends   = np.append(ends, len(gt_mask)-1)
            for s, e in zip(starts, ends):
                ax.axvspan(idx[s], idx[e], alpha=0.08)

    fig.tight_layout()

    # --- Write outputs ---
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=160)
        print(f"[WRITE] PNG  -> {out_path}")
    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        y.rename("y_kw").to_csv(save_csv, index_label="slot_end_local")
        print(f"[WRITE] CSV  -> {save_csv}")

    if show:
        plt.show()
    plt.close(fig)

    return y
