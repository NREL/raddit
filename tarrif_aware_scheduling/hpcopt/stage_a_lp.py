from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import pulp
import shutil
import os

from .part_a import PartAResult
from .part_b import PartBResult
from .part_c import PartCResult

# ---------------- util helpers ----------------

def _month_labels(idx: pd.DatetimeIndex) -> pd.Index:
    """
    Return month labels as strings 'YYYY-MM' using the LOCAL tz of idx.
    This avoids tz drop warnings from .to_period('M') and preserves local month boundaries.
    """
    return pd.Index(idx.strftime("%Y-%m"), name="month")

def _solver_available(bin_name: str | None) -> bool:
    if not bin_name:
        return False
    return shutil.which(bin_name) is not None

def _choose_solver(name: str, time_limit_minutes: int, path: str | None = None):
    """
    Return a PuLP solver command instance with graceful fallbacks.
    Supports: 'highs', 'cbc', 'gurobi'. If 'path' is provided, we pass it to the solver.
    """
    name = (name or "").lower()
    tl = int(max(1, time_limit_minutes)) * 60  # seconds

    # GUROBI
    if name.startswith("gurobi"):
        try:
            return pulp.GUROBI_CMD(timeLimit=tl)
        except Exception:
            # fall through to other solvers
            pass

    # HiGHS
    if name.startswith("highs"):
        # Use explicit path if provided, else require 'highs' in PATH
        if path or _solver_available("highs"):
            try:
                return pulp.HiGHS_CMD(timeLimit=tl, path=path)
            except Exception:
                pass  # fall through

    # CBC
    if name.startswith("cbc"):
        if path or _solver_available("cbc"):
            try:
                return pulp.PULP_CBC_CMD(timeLimit=tl, path=path)
            except Exception:
                pass  # fall through

    # If requested solver is missing, try sensible fallbacks in order
    if _solver_available("highs"):
        try:
            return pulp.HiGHS_CMD(timeLimit=tl)
        except Exception:
            pass
    if _solver_available("cbc"):
        try:
            return pulp.PULP_CBC_CMD(timeLimit=tl)
        except Exception:
            pass

    # Nothing available
    raise RuntimeError(
        "No MILP solver found. Please install one of:\n"
        "  - HiGHS:  conda install -c conda-forge highs   (or brew install highs)\n"
        "  - CBC:    conda install -c conda-forge coincbc (or brew install cbc)\n"
        "Or set a solver path in YAML at solve.stageA_solver_path."
    )

def _half_step_margin(z_raw: float, eps: float) -> float:
    """
    Margin (kW) to the next 0.5 step under half-up rule:
    smallest delta >= 0 so that ceil((z+delta) - 0.5 + eps) > ceil(z - 0.5 + eps).
    """
    if np.isnan(z_raw):
        return np.nan
    k = math.ceil(z_raw - 0.5 + eps)
    thr = k + 0.5 - eps
    return max(0.0, thr - z_raw)

# --------------- result container ---------------

@dataclass
class StageAResult:
    status: str
    solver_name: str
    y_envelope: pd.Series               # kW, indexed by slot_end (local)
    month_table: pd.DataFrame           # per-month peaks/rates/K/ margins
    objective_components: Dict[str, float]
    go: bool

# --------------- main API ---------------

def run_stage_a_lp(
    part_a: PartAResult,
    part_b: PartBResult,
    part_c: PartCResult,
) -> StageAResult:
    """
    Build and solve the Stage A MILP envelope with half-up demand rounding.
    Supports optional per-slot cap y_t <= solve.envelope_max_kw (e.g., 1600).
    """
    cfg = part_a.cfg
    solve_cfg = cfg.get("solve", {})
    billing   = cfg.get("billing", {})
    interval_minutes = int(billing.get("interval_minutes", 15))
    dt_hours = interval_minutes / 60.0
    eps = float(solve_cfg.get("half_up_epsilon_kw", 1e-6))

    # Per-slot cap on y_t (kW). If None/absent => no cap.
    y_cap_cfg = solve_cfg.get("envelope_max_kw", None)
    y_cap_kw: float = float(y_cap_cfg) if y_cap_cfg is not None else float("inf")

    # Inputs
    idx = part_a.slot_ends           # local tz slot-ends
    merged = part_a.merged_on_grid   # contains baseline_kw, background_kw
    attrs  = part_b.attrs.loc[idx]   # align attrs to grid
    jobs   = part_c.jobs_sched

    # --- Maintenance blackout mask: forbid CPU envelope inside blackouts ---
    T = len(idx)
    blackout_mask = np.zeros(T, dtype=bool)
    if getattr(part_a, "blackouts", None):
        slot_delta = pd.Timedelta(minutes=interval_minutes)
        for (bs, be) in part_a.blackouts:
            # Mark any 15-min slot whose [end - slot_delta, end) overlaps (bs, be)
            blackout_mask |= (idx > bs) & (idx - slot_delta < be)


    # Sanity checks
    for col in ["baseline_kw", "background_kw"]:
        if col not in merged.columns:
            raise KeyError(f"merged_on_grid missing {col}")
    for col in ["tou_price_per_kwh", "gt_eligible", "gt_rate_per_kw_month", "dist_rate_per_kw_month"]:
        if col not in attrs.columns:
            raise KeyError(f"attrs missing {col}")

    # Constants per slot
    b_t  = merged["baseline_kw"].to_numpy(dtype=float)
    bg_t = merged["background_kw"].to_numpy(dtype=float)
    r_t  = attrs["tou_price_per_kwh"].to_numpy(dtype=float)
    gt_mask = attrs["gt_eligible"].to_numpy(dtype=bool)

    # Energy of schedulable CPU jobs (kWh)
    E_cpu_kwh = float(jobs["energy_kwh"].sum())

    # Month grouping using local month labels
    months = _month_labels(idx)                    # Index of 'YYYY-MM' strings
    month_list = sorted(months.unique().tolist())  # list of month labels

    # Per-month rates: read from first slot of each month
    gt_rate_by_m = {}
    dist_rate_by_m = {}
    for m in month_list:
        sel = (months == m)
        first_row = attrs.loc[sel].iloc[0]
        gt_rate_by_m[m]   = float(first_row["gt_rate_per_kw_month"])
        dist_rate_by_m[m] = float(first_row["dist_rate_per_kw_month"])

    # ---------- Build MILP ----------
    prob = pulp.LpProblem("StageA_Envelope", sense=pulp.LpMinimize)

    T = len(idx)

    # Variables: y_t with optional upBound = y_cap_kw
    y: Dict[int, pulp.LpVariable] = {}
    if np.isfinite(y_cap_kw):
        y = {t: pulp.LpVariable(f"y_{t}", lowBound=0.0, upBound=y_cap_kw, cat="Continuous") for t in range(T)}
    else:
        y = {t: pulp.LpVariable(f"y_{t}", lowBound=0.0, cat="Continuous") for t in range(T)}

    # Enforce y_t = 0 inside maintenance blackouts
    for t in range(T):
        if blackout_mask[t]:
            prob += (y[t] == 0.0), f"Maintenance_zero_y_{t}"


    Z_dist: Dict[str, pulp.LpVariable] = {}
    Z_gt: Dict[str, pulp.LpVariable]   = {}
    K_dist: Dict[str, pulp.LpVariable] = {}
    K_gt: Dict[str, pulp.LpVariable]   = {}
    for m in month_list:
        Z_dist[m] = pulp.LpVariable(f"Z_dist_{m}", lowBound=0.0, cat="Continuous")
        Z_gt[m]   = pulp.LpVariable(f"Z_gt_{m}",   lowBound=0.0, cat="Continuous")
        K_dist[m] = pulp.LpVariable(f"K_dist_{m}", lowBound=0,   cat="Integer")
        K_gt[m]   = pulp.LpVariable(f"K_gt_{m}",   lowBound=0,   cat="Integer")

    # Quick feasibility hint if a finite y_cap_kw is set
    if np.isfinite(y_cap_kw):
        # Only non-blackout slots can carry CPU envelope
        T_allowed = int((~blackout_mask).sum())
        horizon_hours_allowed = dt_hours * T_allowed
        cap_energy_kwh = y_cap_kw * horizon_hours_allowed
        if cap_energy_kwh + 1e-6 < E_cpu_kwh:
            print("[WARN] solve.envelope_max_kw may be too tight for energy balance "
                  "given maintenance blackouts:")
            print(f"       required={E_cpu_kwh:,.0f} kWh  "
                  f"capacity={cap_energy_kwh:,.0f} kWh "
                  f"(non-blackout hours={horizon_hours_allowed:.1f} h)")
            print("       Stage A can become infeasible unless you raise envelope_max_kw, "
                  "relax energy equality, or adjust blackouts.")


    # Energy balance
    prob += pulp.lpSum(y[t] * dt_hours for t in range(T)) == E_cpu_kwh, "EnergyBalance"

    # Peak constraints + half-up rounding per month
    for m in month_list:
        sel = np.asarray(months == m, dtype=bool)  # boolean array over T
        idx_sel = np.where(sel)[0]

        # Distribution: all slots in the month
        for t in idx_sel:
            prob += Z_dist[m] >= b_t[t] + bg_t[t] + y[t], f"Zdist_ge_{m}_{t}"

        # G&T: only eligible slots in the month
        sel_gt = sel & gt_mask
        if sel_gt.any():
            for t in np.where(sel_gt)[0]:
                prob += Z_gt[m] >= b_t[t] + bg_t[t] + y[t], f"Zgt_ge_{m}_{t}"

        # Half-up rounding: K >= Z - 0.5 + eps
        prob += K_dist[m] >= Z_dist[m] - 0.5 + eps, f"Kdist_halfup_{m}"
        prob += K_gt[m]   >= Z_gt[m]   - 0.5 + eps, f"Kgt_halfup_{m}"

    # Objective (only y(t) energy is variable; baseline/background are constants)
    energy_cost = pulp.lpSum(r_t[t] * y[t] * dt_hours for t in range(T))
    demand_cost = pulp.lpSum(
        dist_rate_by_m[m] * K_dist[m] + gt_rate_by_m[m] * K_gt[m]
        for m in month_list
    )
    prob += energy_cost + demand_cost

    # Solve (choose solver with graceful fallbacks)
    solver_name_cfg = (solve_cfg.get("stageA_solver") or "highs").lower()
    solver_path_cfg = solve_cfg.get("stageA_solver_path")  # optional, e.g., "/opt/homebrew/bin/highs" or "/usr/bin/cbc"
    time_limit      = int(solve_cfg.get("stageA_minutes", 5))

    try:
        solver = _choose_solver(solver_name_cfg, time_limit_minutes=time_limit, path=solver_path_cfg)
        status_code = prob.solve(solver)
        solver_used = type(solver).__name__
    except (pulp.PulpSolverError, RuntimeError) as e1:
        # Attempt fallback to CBC explicitly (common on conda-forge)
        try:
            solver = _choose_solver("cbc", time_limit_minutes=time_limit, path=None)
            status_code = prob.solve(solver)
            solver_used = type(solver).__name__ + " (fallback)"
        except Exception as e2:
            # Surface a clear message with guidance
            raise RuntimeError(
                f"Stage A failed: requested solver='{solver_name_cfg}' not usable "
                f"and CBC fallback failed.\n"
                f"Original error: {e1}\nFallback error: {e2}\n\n"
                "Install a solver and/or set solve.stageA_solver/solve.stageA_solver_path in your YAML."
            )

    status_str = pulp.LpStatus[status_code]
    print("=== STAGE A: SOLVE SUMMARY ===")
    print(f"Solver: {solver_used}  |  Status: {status_str}")

    # Extract results
    y_vals = np.array([pulp.value(y[t]) for t in range(T)], dtype=float)
    Z_dist_vals = {m: float(pulp.value(Z_dist[m])) for m in month_list}
    Z_gt_vals   = {m: float(pulp.value(Z_gt[m]))   for m in month_list}
    K_dist_vals = {m: int(round(pulp.value(K_dist[m]) if pulp.value(K_dist[m]) is not None else 0)) for m in month_list}
    K_gt_vals   = {m: int(round(pulp.value(K_gt[m]) if pulp.value(K_gt[m]) is not None else 0))     for m in month_list}

    # QA residuals
    energy_kwh_y = float(y_vals.sum() * dt_hours)
    energy_resid = abs(energy_kwh_y - E_cpu_kwh)

    # Month table
    rows = []
    for m in month_list:
        z_d = Z_dist_vals[m]
        z_g = Z_gt_vals[m]
        k_d = K_dist_vals[m]
        k_g = K_gt_vals[m]
        gt_rate = gt_rate_by_m[m]
        dist_rate = dist_rate_by_m[m]
        rows.append({
            "month": m,
            "Z_dist_raw_kw": z_d,
            "K_dist_kw": k_d,
            "Z_gt_raw_kw": z_g,
            "K_gt_kw": k_g,
            "gt_rate_per_kw_month": gt_rate,
            "dist_rate_per_kw_month": dist_rate,
            "margin_to_next_0p5_dist_kw": _half_step_margin(z_d, eps),
            "margin_to_next_0p5_gt_kw": _half_step_margin(z_g, eps),
            "demand_cost_month_usd": dist_rate * k_d + gt_rate * k_g
        })
    month_table = pd.DataFrame(rows).set_index("month")

    # Objective components (evaluated at solution)
    energy_cost_val  = float(sum(r_t[t] * y_vals[t] * dt_hours for t in range(T)))
    demand_cost_val  = float(month_table["demand_cost_month_usd"].sum())
    objective_val    = energy_cost_val + demand_cost_val

    # Build series for envelope
    y_series = pd.Series(y_vals, index=idx, name="y_kw")

    # GO / NO-GO
    good_status = status_str in ("Optimal", "Feasible")
    tol = 1e-5 * max(1.0, E_cpu_kwh)
    go = good_status and (energy_resid <= tol) and np.all(y_vals >= -1e-7)

    # Prints
    cap_str = "∞" if not np.isfinite(y_cap_kw) else f"{y_cap_kw:.0f}"
    print(f"[GO] envelope_max_kw            : {cap_str}")
    print(f"[GO] Energy balance sched kWh   : {E_cpu_kwh:,.3f}")
    print(f"[GO] Energy balance y(t)  kWh   : {energy_kwh_y:,.3f}   (residual={energy_resid:.6f})")
    print(f"[GO] Envelope y stats (kW)      : min={np.nanmin(y_vals):.6f}, p50={np.nanmedian(y_vals):.6f}, max={np.nanmax(y_vals):.6f}")
    if np.isfinite(y_cap_kw) and np.nanmax(y_vals) > y_cap_kw + 1e-6:
        print("[WARN] y(t) exceeds envelope_max_kw — cap not enforced correctly.")
    else:
        print("[GO] y(t) respects the per-slot cap.")
    print(f"[GO] Energy cost (CPU part) usd : {energy_cost_val:,.2f}")
    print(f"[GO] Demand cost total     usd  : {demand_cost_val:,.2f}")
    print("[SAMPLE] Month table (first 3):")
    print(month_table.head(3).to_string())

    print(f"VERDICT: {'GO' if go else 'NO-GO'}")

    return StageAResult(
        status=status_str,
        solver_name=solver_used,
        y_envelope=y_series,
        month_table=month_table,
        objective_components={
            "objective_total_usd": objective_val,
            "energy_cost_cpu_usd": energy_cost_val,
            "demand_cost_usd": demand_cost_val
        },
        go=go
    )
