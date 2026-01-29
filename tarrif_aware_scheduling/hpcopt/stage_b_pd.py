# hpcopt/stage_b_pd.py
from __future__ import annotations
import math
import heapq
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time


import numpy as np
import pandas as pd

from .part_a import PartAResult
from .part_b import PartBResult
from .part_c import PartCResult
from .stage_a_lp import StageAResult

@dataclass
class CostResult:
    energy_cost_usd: float
    demand_cost_usd: float
    objective_usd: float
    month_table: pd.DataFrame

def _half_up_int_scalar(z: float, eps: float) -> int:
    return int(math.ceil(z - 0.5 + eps))

def compute_cost_exact(
    idx_15: pd.DatetimeIndex,
    attrs_15: pd.DataFrame,
    merged_15: pd.DataFrame,
    cpu_kw_15: pd.Series,
    eps: float,
) -> CostResult:
    assert cpu_kw_15.index.equals(idx_15)
    dt_h = 0.25
    net = (merged_15["baseline_kw"].to_numpy(float)
           + merged_15["background_kw"].to_numpy(float)
           + cpu_kw_15.to_numpy(float))
    energy_usd = float(np.sum(attrs_15["tou_price_per_kwh"].to_numpy(float) * net * dt_h))

    months = pd.Index(idx_15.strftime("%Y-%m"))
    month_list = sorted(months.unique().tolist())
    rows = []
    for m in month_list:
        sel = (months == m)
        sel_gt = sel & attrs_15["gt_eligible"].to_numpy(bool)
        z_dist = float(np.max(net[sel]))
        z_gt   = float(np.max(net[sel_gt])) if np.any(sel_gt) else 0.0
        k_dist = _half_up_int_scalar(z_dist, eps)
        k_gt   = _half_up_int_scalar(z_gt, eps)
        dist_rate = float(attrs_15.loc[sel, "dist_rate_per_kw_month"].iloc[0])
        gt_rate   = float(attrs_15.loc[sel, "gt_rate_per_kw_month"].iloc[0])
        rows.append({
            "month": m,
            "Z_dist_raw_kw": z_dist,
            "K_dist_kw": k_dist,
            "Z_gt_raw_kw": z_gt,
            "K_gt_kw": k_gt,
            "dist_rate_per_kw_month": dist_rate,
            "gt_rate_per_kw_month": gt_rate,
            "demand_cost_month_usd": dist_rate * k_dist + gt_rate * k_gt,
        })
    mt = pd.DataFrame(rows).set_index("month")
    demand_usd = float(mt["demand_cost_month_usd"].sum())
    return CostResult(energy_cost_usd=energy_usd,
                      demand_cost_usd=demand_usd,
                      objective_usd=energy_usd + demand_usd,
                      month_table=mt)

# -------------------------
# Segment tree (range add + range min)
# -------------------------

class RangeAddMinSegTree:
    """
    Range add (decrement headroom) + range min query on an array (capacity headroom per minute).
    Supports:
      - add(l, r, delta): add delta to [l, r) (delta negative to consume capacity)
      - min(l, r): minimum on [l, r)
    """
    def __init__(self, n: int, init: Optional[np.ndarray] = None):
        self.n = 1
        while self.n < n:
            self.n <<= 1
        self.inf = 10**15
        self.minv = np.zeros(2*self.n, dtype=float)
        self.lz = np.zeros(2*self.n, dtype=float)
        if init is None:
            pass
        else:
            self.minv[self.n:self.n+n] = init
            for i in range(self.n-1, 0, -1):
                self.minv[i] = min(self.minv[2*i], self.minv[2*i+1])

    def _push(self, i: int):
        if self.lz[i] != 0.0:
            v = self.lz[i]
            for ch in (2*i, 2*i+1):
                self.minv[ch] += v
                self.lz[ch] += v
            self.lz[i] = 0.0

    def _add(self, i: int, l: int, r: int, ql: int, qr: int, v: float):
        if qr <= l or r <= ql:
            return
        if ql <= l and r <= qr:
            self.minv[i] += v
            self.lz[i] += v
            return
        self._push(i)
        m = (l+r)//2
        self._add(2*i, l, m, ql, qr, v)
        self._add(2*i+1, m, r, ql, qr, v)
        self.minv[i] = min(self.minv[2*i], self.minv[2*i+1])

    def add(self, l: int, r: int, v: float):
        self._add(1, 0, self.n, l, r, v)

    def _min(self, i: int, l: int, r: int, ql: int, qr: int) -> float:
        if qr <= l or r <= ql:
            return self.inf
        if ql <= l and r <= qr:
            return self.minv[i]
        self._push(i)
        m = (l+r)//2
        return min(self._min(2*i, l, m, ql, qr),
                   self._min(2*i+1, m, r, ql, qr))

    def min(self, l: int, r: int) -> float:
        return self._min(1, 0, self.n, l, r)

# -------------------------
# Params / Result
# -------------------------

@dataclass
class PDParams:
    minute_freq: str = "1min"
    iterations: int = 3
    alpha_gt: float = 8.0
    demand_margin_kw: float = 50.0
    step_spike_strength: float = 500.0
    mu_gamma: float = 0.0005
    month_quota_slack: float = 0.05
    forbid_step_increase: bool = False
    step_penalty_mult: float = 1.0
    per_group_sample_factor: int = 4
    max_candidates_per_group: int = 20000
    seed_order: str = "energy_desc"
    verbose: bool = True

    # Adaptive candidate budget
    cand_base_min: int = 1024
    cand_sqrt_coeff: float = 25.0
    cand_log_coeff: float = 250.0
    cand_frac_of_grid: float = 0.02   # 2% of the grid at most

    # Batch placement
    batch_cap: int = 1024
    # How many candidates to consider early vs late iterations
    cand_ramp_mode: str = "linear"   # "none" | "linear" | "exp"
    cand_ramp_start_frac: float = 0.5   # use 50% of adaptive nCand on iter 1
    cand_ramp_end_frac: float   = 1.0    # and 100% by the last iter


    # Pre-screen candidates
    use_feasibility_prescreen: bool = True      # enable fast reject of impossible windows
    prescreen_margin_kw: float = 0.0            # require this much residual step headroom (kW) in each touched slot
    keep_feasible_candidate_budget: bool = True

    # What to do with leftover jobs after primal–dual iterations
    # "none"         -> keep current behavior (may leave jobs unplaced)
    # "capacity_only"-> final pass that tries to place all remaining jobs using capacity only
    #                   (ignores demand guard; may increase peaks slightly)
    # "guarded"      -> final pass that still respects demand guard but scans the full grid
    final_fill_mode: str = "guarded_then_capacity"



@dataclass
class StageBPDResult:
    scheduled: pd.DataFrame            # job_id, start_ts, end_ts, nodes, node_type
    cpu_kw_15: pd.Series               # per 15-min slot kW from scheduled CPU
    cost: CostResult
    stats: Dict[str, float]
    go: bool

# -------------------------
# Minute grid helpers
# -------------------------

def _build_minute_grid(res_a: PartAResult, res_b: PartBResult, params: PDParams):
    tz = res_a.slot_ends.tz
    slot_ends15 = res_a.slot_ends  # 15-min trailing labels
    first15 = slot_ends15.min()
    last15  = slot_ends15.max()

    minute_delta = pd.Timedelta(params.minute_freq)
    assert minute_delta in [pd.Timedelta("1min"), pd.Timedelta("5min")], "Use 1min or 5min"
    k = int(pd.Timedelta("15min") / minute_delta)  # 3 if 5min, 15 if 1min
    start_minute_end = first15 - (k-1) * minute_delta
    minute_ends = pd.date_range(start=start_minute_end, end=last15, freq=minute_delta, tz=tz)

    # Map 15-min end j -> indices of its minute ends
    idx_map_15_to_min = []
    pos = 0
    # Build lookup: position in minute_ends for each 15-min end
    min_pos = {t: i for i, t in enumerate(minute_ends)}
    for t15 in slot_ends15:
        start = t15 - (k-1) * minute_delta
        idxs = [min_pos[start + i*minute_delta] for i in range(k)]
        idx_map_15_to_min.append(idxs)
    idx_map_15_to_min = np.array(idx_map_15_to_min, dtype=int)  # shape (T15, k)

    # Broadcast attrs (TOU price, GT flags) to minute grid by copying to k minutes inside each 15-min slot
    attrs15 = res_b.attrs.loc[slot_ends15]
    price15 = attrs15["tou_price_per_kwh"].to_numpy(float)
    gt15    = attrs15["gt_eligible"].to_numpy(bool)

    price_min = np.zeros(len(minute_ends), dtype=float)
    gt_min    = np.zeros(len(minute_ends), dtype=bool)
    for j in range(len(slot_ends15)):
        idxs = idx_map_15_to_min[j]
        price_min[idxs] = price15[j]
        gt_min[idxs]    = gt15[j]

    # Capacity (nodes) per minute: default sum of CPU-exclusive nodes; 0 during maintenance overlaps
    total_nodes = int(res_c_nodes_total(res_a, res_b))  # helper below
    cap_min = np.full(len(minute_ends), total_nodes, dtype=float)
    if res_a.cfg.get("policies", {}).get("forbid_maintenance_overlap", True):
        for (bs, be) in res_a.blackouts:
            mask = (minute_ends > bs) & (minute_ends - minute_delta < be)
            cap_min[mask] = 0.0

    # Month labels for both grids
    months15 = pd.Index(slot_ends15.strftime("%Y-%m"))
    months_min = pd.Index(minute_ends.strftime("%Y-%m"))

    return {
        "minute_ends": minute_ends,
        "minute_delta": minute_delta,
        "k_per_15": k,
        "slot_ends15": slot_ends15,
        "idx_map_15_to_min": idx_map_15_to_min,
        "price_min": price_min,
        "gt_min": gt_min,
        "cap_min": cap_min,
        "months15": months15,
        "months_min": months_min,
    }

def res_c_nodes_total(res_a: PartAResult, res_b: PartBResult) -> int:
    # We don't have res_c here; capacity depends only on nodes inventory (part C).
    # In the driver below we'll pass capacity from res_c; this fallback is not used.
    return 0

# -------------------------
# Grouping jobs
# -------------------------

@dataclass
class JobGroup:
    dur_min: int         # integer minutes (grid-aligned)
    nodes: int
    job_ids: np.ndarray
    job_p_kw: np.ndarray # per-job average power (kW)
    count: int

def build_groups(res_c: PartCResult, params: PDParams) -> List[JobGroup]:
    gmins = int(pd.Timedelta(params.minute_freq).total_seconds() // 60)
    df = res_c.jobs_sched.copy()

    # Round duration up to grid (at least 1 grid)
    dur_min = np.maximum(1, np.ceil(df["duration_seconds"].to_numpy(float)/60.0 / gmins).astype(int)) * gmins
    nodes = df["nodes"].astype(int).to_numpy()
    pkw   = df["avg_power_kw"].to_numpy(float)
    job_ids = df["job_id"].astype(str).to_numpy()

    # Group by (dur_min, nodes)
    key = np.stack([dur_min, nodes], axis=1)
    # stable mapping to groups
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    groups: List[JobGroup] = []
    for gid, (D, N) in enumerate(uniq):
        mask = (inv == gid)
        ids  = job_ids[mask]
        p    = pkw[mask]
        # sort within group by power desc so we place heavier jobs into cheapest windows first
        order = np.argsort(-p)
        groups.append(JobGroup(
            dur_min=int(D),
            nodes=int(N),
            job_ids=ids[order],
            job_p_kw=p[order],
            count=mask.sum()
        ))
    return groups

# -------------------------
# Price field & step spikes
# -------------------------

def _build_base_price_min(grid, res_a: PartAResult, alpha_gt: float) -> np.ndarray:
    return grid["price_min"] + alpha_gt * grid["gt_min"].astype(float)

def _compute_month_targets_kwh(res_d: StageAResult) -> Dict[str, float]:
    # From Stage A envelope (y_kw) per 15min: E_m = sum(y_kw * 0.25) per month
    idx = res_d.y_envelope.index
    months = pd.Index(idx.strftime("%Y-%m"))
    Y = res_d.y_envelope.to_numpy(float)
    E = {}
    for m in sorted(months.unique().tolist()):
        sel = (months == m)
        E[m] = float(np.sum(Y[sel] * 0.25))
    return E

def _broadcast_month_penalty_to_minutes(mu_by_month: Dict[str, float], months_min: pd.Index) -> np.ndarray:
    mu_min = np.zeros(len(months_min), dtype=float)
    for i, m in enumerate(months_min):
        mu_min[i] = mu_by_month.get(m, 0.0)
    return mu_min

def _step_spike_from_headroom(
    headroom_kw_15: np.ndarray,
    margin_kw: float,
    strength: float
) -> np.ndarray:
    """
    headroom_kw_15: array >= -inf
    For margin > margin_kw: 0
    For margin <= margin_kw: monotone increasing spike, +inf as margin->0-
    """
    eps = 1e-6
    spike = np.zeros_like(headroom_kw_15, dtype=float)
    # Negative margins: very large spike
    neg = (headroom_kw_15 < 0)
    spike[neg] = strength * 1e6
    # Near threshold
    near = (headroom_kw_15 >= 0) & (headroom_kw_15 < margin_kw)
    # Smooth barrier: strength * (1/(margin+eps) - 1/(margin_kw+eps))
    spike[near] = strength * (1.0/(headroom_kw_15[near]+eps) - 1.0/(margin_kw+eps))
    return spike


# -------------------------
# Batch placement helpers
# -------------------------

# --- Adaptive top-k budget for candidate start times -------------------------
def _adaptive_nCand(N_positions: int, 
                    g_count: int, g_job_p_kw: float, g_dur_min: float, 
                    params: PDParams) -> int:
    # sublinear in group size, + hard/global caps, + fraction of grid
    sublinear = int(params.cand_sqrt_coeff * np.sqrt(max(1, g_count)) +
                    params.cand_log_coeff  * np.log1p(max(1, g_count)))
    frac_cap  = int(params.cand_frac_of_grid * N_positions)
    nCand = min(
        N_positions,
        max(params.cand_base_min, sublinear),
        frac_cap if frac_cap > 0 else N_positions,
        int(params.max_candidates_per_group),
    )
    if hasattr(params, "per_group_sample_factor"):
        nCandFinal = min(nCand, int(np.sqrt(g_count) * params.per_group_sample_factor * max(100,int(np.mean(g_job_p_kw) * g_dur_min))))
        print(f'{nCand:06d}\t{g_count:06d}\t{np.mean(g_job_p_kw)*g_dur_min:.8g}\t{nCandFinal:06d}', end='\r')
    return max(1, nCandFinal)


def _cand_iter_multiplier(it: int, iters: int, params: PDParams) -> float:
    """
    Returns a factor in (0, 1+] used to scale the adaptive nCand as
    outer iteration 'it' increases. it in [1..iters].
    """
    if params.cand_ramp_mode == "none" or iters <= 1:
        return 1.0

    a = float(params.cand_ramp_start_frac)
    b = float(params.cand_ramp_end_frac)
    t = 0.0 if iters <= 1 else (it - 1) / max(1, 10 - 1)

    if params.cand_ramp_mode == "linear":
        if it >= 10:
            return 1
        return max(0.0, a + (b - a) * t)

    if params.cand_ramp_mode == "exp":
        # geometric interpolation from a -> b across iterations
        base = max(a, 1e-9)
        return max(0.0, base * ((b / base) ** t))

    return 1.0


# --- Fast top-k indices (cheapest windows) -----------------------------------
def _topk_start_indices(roll: np.ndarray, k: int) -> np.ndarray:
    if k >= len(roll):
        return np.argsort(roll)
    part = np.argpartition(roll, k)[:k]         # unsorted top-k
    return part[np.argsort(roll[part])]         # sort just the top-k block

# --- Fractional coverage of 15-min slots for a [t0, t0+dur) minute window ----
def _edge_fractional_weights(k_per_15: int, dur_bins: int, start_mod: int):
    # minutes per slot = k_per_15; start_mod in [0, k_per_15-1]
    left_frac_bins = min(dur_bins, k_per_15 - start_mod) if start_mod > 0 else 0
    rem = dur_bins - left_frac_bins
    full_slots = max(0, rem // k_per_15)
    right_rem = rem - full_slots * k_per_15
    weights = []
    if left_frac_bins > 0:
        weights.append(("left", left_frac_bins / k_per_15))
    for _ in range(full_slots):
        weights.append(("full", 1.0))
    if right_rem > 0:
        weights.append(("right", right_rem / k_per_15))
    return weights  # e.g. [('left', 2/3), ('full', 1.0), ('right', 1/3)]



# --- Max batch size at start t0 without violating capacity or billed steps ---
def _safe_batch_size_at_start(t0: int,
                              dur_bins: int,
                              need_nodes: int,
                              p_one: float,
                              k_per_15: int,
                              headroom_dist_15: np.ndarray,
                              headroom_gt_15: np.ndarray,
                              gt_mask15: np.ndarray,
                              seg: RangeAddMinSegTree,
                              batch_cap: int) -> int:
    # Capacity-limited number of jobs
    nodes_headroom = seg.min(t0, t0 + dur_bins)
    cap_limited = int(nodes_headroom // max(1, need_nodes))
    if cap_limited <= 0:
        return 0

    # Step-limited number of jobs (fractional per-slot headroom bound)
    start_mod = t0 % k_per_15
    weights = _edge_fractional_weights(k_per_15, dur_bins, start_mod)
    j0 = t0 // k_per_15  # first 15-min slot id touched
    # map weights to touched 15-min slot indices
    touched = []
    off = 0
    if weights and weights[0][0] == "left":
        touched.append(j0); off += 1
    full_count = sum(1 for w in weights if w[0] == "full")
    for _ in range(full_count):
        touched.append(j0 + off); off += 1
    if weights and weights[-1][0] == "right":
        touched.append(j0 + off)

    slot_weights = []
    for lab, frac in weights:
        slot_weights.append(frac if lab in ("left", "right") else 1.0)

    def _slot_limit(j, w):
        hr_d = headroom_dist_15[j]
        hr_g = headroom_gt_15[j] if gt_mask15[j] else np.inf
        hr = min(hr_d, hr_g)
        if not np.isfinite(hr) or w <= 0:
            return np.inf
        return max(0, int(np.floor(hr / (p_one * w + 1e-12))))

    step_limits = [ _slot_limit(j, w) for j, w in zip(touched, slot_weights) ]
    step_limited = int(min(step_limits)) if len(step_limits) else int(np.inf)

    return max(0, min(cap_limited, step_limited, int(batch_cap)))


# -------------------------
# Main solver
# -------------------------

def run_stage_b_primal_dual(
    res_a: PartAResult,
    res_b: PartBResult,
    res_c: PartCResult,
    res_d: StageAResult,
    params: Optional[PDParams] = None
) -> StageBPDResult:
    if params is None:
        params = PDParams()

    cfg = res_a.cfg
    tz = res_a.slot_ends.tz
    eps = float(cfg.get("solve", {}).get("half_up_epsilon_kw", 1e-6))
    margin_cfg = float(cfg.get("solve", {}).get("demand_threshold_margin_kw", params.demand_margin_kw))

    # --- Minute grid and mappings (built from Part A/B info) ---
    grid = _build_minute_grid(res_a, res_b, params)
    minute_ends: pd.DatetimeIndex = grid["minute_ends"]
    minute_delta: pd.Timedelta    = grid["minute_delta"]
    k_per_15: int                 = grid["k_per_15"]
    idx_map_15_to_min: List[np.ndarray] = grid["idx_map_15_to_min"]
    slot_ends15: pd.DatetimeIndex = grid["slot_ends15"]
    months15: pd.Index            = grid["months15"]      # Index of "YYYY-MM" strings, aligned to slot_ends15
    months_min: pd.Index          = grid["months_min"]    # Index of "YYYY-MM" strings, aligned to minute_ends

    # --- Capacity over minutes: total nodes with maintenance blackouts set to 0 ---
    total_nodes = int(res_c.nodes_inv["count"].sum())
    cap_min = np.full(len(minute_ends), total_nodes, dtype=float)
    if cfg.get("policies", {}).get("forbid_maintenance_overlap", True):
        for (bs, be) in res_a.blackouts:
            # minute bin "active" if its [start,end) intersects blackout
            mask = (minute_ends > bs) & (minute_ends - minute_delta < be)
            cap_min[mask] = 0.0

    # --- Base price (minutes): TOU + GT steering (broadcast from 15-min attrs) ---
    price_base_min = _build_base_price_min(grid, res_a, params.alpha_gt)

    # --- Job groups (by (duration_minutes, nodes)), with job ids and per-job power kw ---
    groups = build_groups(res_c, params)

    # --- Duals for monthly quota and the targets derived from Stage A envelope ---
    mu_by_month: Dict[str, float] = {m: 0.0 for m in sorted(months15.unique().tolist())}
    month_targets_kwh = _compute_month_targets_kwh(res_d)   # dict: "YYYY-MM" -> kWh
    month_slack = params.month_quota_slack

    # --- State arrays we build up while placing jobs ---
    cpu_kw_min = np.zeros(len(minute_ends), dtype=float)    # scheduled CPU kW at minute resolution
    seg = RangeAddMinSegTree(len(minute_ends), init=cap_min)

    # Helper: aggregate minute CPU -> 15-min averages using the precomputed mapping
    def cpu_kw_min_to_15(cpu_min: np.ndarray) -> np.ndarray:
        vals = np.zeros(len(slot_ends15), dtype=float)
        for j in range(len(slot_ends15)):
            idxs = idx_map_15_to_min[j]
            # minute bins are constant within each bin; average them to match trailing 15-min label
            vals[j] = float(np.mean(cpu_min[idxs])) if len(idxs) else 0.0
        return vals

    # Convenience (15-min): tariff attrs and baseline/background on the grid
    attrs15 = res_b.attrs.loc[slot_ends15]
    merged15 = res_a.merged_on_grid.loc[slot_ends15][["baseline_kw", "background_kw"]]

    # Month masks over 15-min grid (robust to dtype: yields bool ndarray)
    month_list = sorted(months15.unique().tolist())
    month_mask_15 = {m: np.asarray(months15 == m, dtype=bool) for m in month_list}
    gt_mask15 = attrs15["gt_eligible"].to_numpy(bool)

    # --- GO/NO-GO header ---
    print("=== PART E′: PRIMAL–DUAL PACK (INIT) ===")
    print(f"[GO] Minute grid: {params.minute_freq}  bins={len(minute_ends):,}  (k_per_15={k_per_15})")
    print(f"[GO] Total nodes: {total_nodes}  | Maintenance mins with cap=0: {(cap_min == 0).sum():,}")
    print(f"[GO] Groups: {len(groups)} unique (duration,nodes) buckets")

    # --- Seed ordering of groups (where to start placing first) ---
    if params.seed_order in ("energy_desc", "energy_asc"):
        key_energy = []
        for g in groups:
            # sum of job energies in kWh for the group (sum p_kw) * (duration hours)
            e_group_kwh = float(np.mean(g.job_p_kw) * (g.dur_min / 60.0))
            key_energy.append(e_group_kwh)
        key = np.array(key_energy)
        order = np.argsort(-key) if params.seed_order == "energy_desc" else np.argsort(key)
    elif params.seed_order == "power_desc":
        keyp = np.array([float(np.max(g.job_p_kw)) if g.count > 0 else 0.0 for g in groups])
        order = np.argsort(-keyp)
    else:  # random
        rng = np.random.default_rng(42)
        order = np.arange(len(groups))
        rng.shuffle(order)

    placements: List[Tuple[str, pd.Timestamp, pd.Timestamp, int, str]] = []  # (job_id, start_ts, end_ts, nodes, node_type)

    # --- Main primal–dual iterations ---
    print('Iterations:',params.iterations)
    for it in range(1, params.iterations + 1):
        print('Iteration number:', it)
        t_iter_start = time.time()

        # Compute current 15-min net
        cpu15 = cpu_kw_min_to_15(cpu_kw_min)
        net15 = (
            merged15["baseline_kw"].to_numpy(float)
            + merged15["background_kw"].to_numpy(float)
            + cpu15
        )

        # Per-month billed thresholds from Stage A (billed whole kW + 0.5 - eps)
        Kd = res_d.month_table["K_dist_kw"].astype(int).to_dict()
        Kg = res_d.month_table["K_gt_kw"].astype(int).to_dict()

        thr_dist = np.zeros(len(slot_ends15), dtype=float)
        thr_gt   = np.zeros(len(slot_ends15), dtype=float)
        for m in month_list:
            sel = month_mask_15[m]
            kd = Kd.get(m, 0)
            kg = Kg.get(m, 0)
            thr_dist[sel] = kd + (0.5 - eps)
            thr_gt[sel]   = kg + (0.5 - eps)

        # Headroom to demand steps (G&T headroom only where eligible)
        headroom_dist = thr_dist - net15
        headroom_gt   = thr_gt   - net15
        headroom_gt[~gt_mask15] = np.inf

        # Broadcast month-penalty duals to minutes
        mu_min = _broadcast_month_penalty_to_minutes(mu_by_month, months_min)

        placed_this_iter = 0
        cand_evals = 0
        cap_skips = 0
        guard_skips = 0
        prescreen_skips = 0

        # Allocate per group
        order_it = 0
        for gi in order:
            order_it += 1
            g = groups[gi]
            if g.count == 0:
                continue

            # --- Per-group refresh of headroom and price field ---
            # Use current minute placements (cpu_kw_min) to rebuild 15-min net load
            cpu15_now = cpu_kw_min_to_15(cpu_kw_min)
            net15_now = (
                merged15["baseline_kw"].to_numpy(float)
                + merged15["background_kw"].to_numpy(float)
                + cpu15_now
            )
        
            # Headroom to the *raw* half-up thresholds set by Stage A
            headroom_dist = thr_dist - net15_now
            headroom_gt   = thr_gt   - net15_now
            headroom_gt[~gt_mask15] = np.inf
        
            # Build step spikes (15-min) using group-fresh headrooms
            spike_dist_15 = _step_spike_from_headroom(
                headroom_dist, margin_cfg, params.step_spike_strength
            )
            spike_gt_15 = _step_spike_from_headroom(
                headroom_gt, margin_cfg, params.step_spike_strength
            )
        
            # Broadcast spikes to the minute grid for this group
            spike_min = np.zeros(len(minute_ends), dtype=float)
            for j in range(len(slot_ends15)):
                idxs = idx_map_15_to_min[j]
                if len(idxs):
                    val = spike_dist_15[j] + spike_gt_15[j]
                    spike_min[idxs] += val
        
            # Group-local price field: base TOU + GT steer + (fresh) step spikes + month duals
            price_k = price_base_min + spike_min + mu_min


            # duration in minute bins, node need
            step_mins = int(minute_delta.total_seconds() // 60)
            dur_bins = int(g.dur_min / max(1, step_mins))
            if dur_bins <= 0:
                # guard: zero-length jobs shouldn't exist, but skip if bad input appears
                continue
            need_nodes = g.nodes

            # Candidate batching
            # Rolling sum of prices to find cheap windows of length dur_bins
            csum = np.concatenate(([0.0], np.cumsum(price_k)))
            end_idx = np.arange(0, len(minute_ends) - dur_bins + 1) + dur_bins
            start_idx = np.arange(0, len(minute_ends) - dur_bins + 1)
            roll = csum[end_idx] - csum[start_idx]
            
            # Adaptive candidate count + iteration ramp
            N_positions = len(start_idx)
            nCand_base  = _adaptive_nCand(N_positions, g.count, g.job_p_kw, g.dur_min, params)
            mult        = _cand_iter_multiplier(it, params.iterations, params)
            nCand       = max(1, int(nCand_base * mult))
            
            # respect global grid cap
            nCand = min(nCand, N_positions)
            
            order_idx = _topk_start_indices(roll, nCand)
            chunk     = int(max(32, min(8192, nCand // 4)))

            job_cursor = 0
        
            for base in range(0, nCand, chunk):
                cand = order_idx[base:base + chunk]
                cand_evals += len(cand)
   
                for t0 in cand:
                    if job_cursor >= g.count:
                        break
                    # Quick capacity pre-check
                    if seg.min(t0, t0 + dur_bins) < need_nodes:
                        cap_skips += 1
                        continue
            
                    # Max batch we can safely place at this t0 without increasing billed steps
                    p_one = float(g.job_p_kw[job_cursor]) 
                    batch = _safe_batch_size_at_start(
                        t0, dur_bins, need_nodes, p_one, k_per_15,
                        headroom_dist, headroom_gt, gt_mask15,
                        seg, params.batch_cap
                    )
                    # Also cap by remaining jobs in the group
                    batch = min(batch, g.count - job_cursor)
                    
                    if batch <= 0:
                        guard_skips += 1
                        continue

                    seg.add(t0, t0 + dur_bins, -need_nodes * batch)
                    cpu_kw_min[t0:t0 + dur_bins] += p_one * batch
                    
                    # Safety assert: no negative headroom in the affected interval
                    assert seg.min(t0, t0 + dur_bins) >= -1e-9, "Capacity violation after batch commit"
                    
                    # Record placements
                    start_ts = minute_ends[t0] - minute_delta
                    end_ts   = start_ts + pd.Timedelta(minutes=g.dur_min)
                    jobs_slice = g.job_ids[job_cursor: job_cursor + batch]
                    placements.extend(
                        [(jid, start_ts, end_ts, g.nodes, "cpu_exclusive") for jid in jobs_slice]
                    )
                    
                    job_cursor       += batch
                    placed_this_iter += batch

            
                if job_cursor >= g.count:
                    break

            
            
            # Shrink group remainder
            if job_cursor > 0:
                groups[gi] = JobGroup(
                    dur_min=g.dur_min, nodes=g.nodes,
                    job_ids=g.job_ids[job_cursor:], job_p_kw=g.job_p_kw[job_cursor:],
                    count=g.count - job_cursor
                )
                
        if params.verbose:
            print(f"[iter {it}] group {order_it}/{len(order)}  nCand_base={nCand_base:,}  mult={mult:.2f}  nCand={nCand:,}")

        # --- Dual update for monthly quotas (nudges energy away from overfull months) ---
        cpu15 = cpu_kw_min_to_15(cpu_kw_min)
        dt_h = 0.25
        for m in month_list:
            sel = month_mask_15[m]
            if not np.any(sel):
                continue
            placed_kwh_m = float(np.sum(cpu15[sel] * dt_h))
            target_kwh_m = float(month_targets_kwh.get(m, 0.0))
            if target_kwh_m > 0:
                over = placed_kwh_m - (1.0 + month_slack) * target_kwh_m
                if over > 0:
                    mu_by_month[m] += params.mu_gamma * over

        if params.verbose:
            elapsed = time.time() - t_iter_start
            print(
                f"[ITER {it}] placed={placed_this_iter:,} "
                f"| cand_evals={cand_evals:,} | prescreen_skips={prescreen_skips:,} "
                f"| cap_skips={cap_skips:,} | guard_skips={guard_skips:,} "
                f"| time={elapsed:.1f}s"
            )
        if placed_this_iter == 0:
            break


    # --- Final fill: place remaining jobs (large-first, cost-greedy) -------------
    remaining_jobs = sum(g.count for g in groups)
    if remaining_jobs > 0 and params.final_fill_mode != "none":
        print(f"[FINAL FILL] Remaining jobs after primal–dual passes: {remaining_jobs:,}")

        # ---------- Common price field + headroom for the first pass ----------
        # Rebuild CPU 15-min and net load from current minute schedule
        cpu15_now = cpu_kw_min_to_15(cpu_kw_min)
        net15_now = (
            merged15["baseline_kw"].to_numpy(float)
            + merged15["background_kw"].to_numpy(float)
            + cpu15_now
        )

        # Per-month Stage A billed thresholds
        Kd = res_d.month_table["K_dist_kw"].astype(int).to_dict()
        Kg = res_d.month_table["K_gt_kw"].astype(int).to_dict()

        thr_dist = np.zeros(len(slot_ends15), dtype=float)
        thr_gt   = np.zeros(len(slot_ends15), dtype=float)
        for m in month_list:
            sel = month_mask_15[m]
            kd = Kd.get(m, 0)
            kg = Kg.get(m, 0)
            thr_dist[sel] = kd + (0.5 - eps)
            thr_gt[sel]   = kg + (0.5 - eps)

        # Headroom vs Stage A thresholds
        headroom_dist = thr_dist - net15_now
        headroom_gt   = thr_gt   - net15_now
        headroom_gt[~gt_mask15] = np.inf  # G&T only where eligible

        # Build spikes for pricing (not for feasibility in capacity-only mode)
        spike_dist_15 = _step_spike_from_headroom(
            headroom_dist, margin_cfg, params.step_spike_strength
        )
        spike_gt_15 = _step_spike_from_headroom(
            headroom_gt, margin_cfg, params.step_spike_strength
        )

        # Broadcast spikes to minutes
        spike_min_final = np.zeros(len(minute_ends), dtype=float)
        for j in range(len(slot_ends15)):
            idxs = idx_map_15_to_min[j]
            if len(idxs):
                val = spike_dist_15[j] + spike_gt_15[j]
                spike_min_final[idxs] += val

        # Month duals on minutes
        mu_min_final = _broadcast_month_penalty_to_minutes(mu_by_month, months_min)

        # Final price field: TOU + GT steer + spikes + month duals
        price_fill_min = price_base_min + spike_min_final + mu_min_final

        # Sort leftover groups by energy-per-job (kWh) descending: large jobs first
        leftover_indices = [i for i, g in enumerate(groups) if g.count > 0]
        def _group_energy_per_job(gi: int) -> float:
            g = groups[gi]
            if g.count == 0:
                return 0.0
            return float(np.mean(g.job_p_kw) * (g.dur_min / 60.0))

        leftover_indices.sort(key=_group_energy_per_job, reverse=True)

        # Helper: which 15-min slots a [t0, t0+dur) window touches, with fractional weights
        def _touched_slots_and_weights(t0: int, dur_bins: int):
            start_mod = t0 % k_per_15
            weights = _edge_fractional_weights(k_per_15, dur_bins, start_mod)
            j0 = t0 // k_per_15
            touched = []
            slot_weights = []
            off = 0
            if weights and weights[0][0] == "left":
                touched.append(j0)
                slot_weights.append(weights[0][1])
                off += 1
            full_count = sum(1 for lab, _ in weights if lab == "full")
            for _ in range(full_count):
                touched.append(j0 + off)
                slot_weights.append(1.0)
                off += 1
            if weights and weights[-1][0] == "right":
                touched.append(j0 + off)
                slot_weights.append(weights[-1][1])
            return touched, slot_weights

        # ---------- PASS 1: guarded or capacity-only, depending on mode ----------
        # For "guarded_then_capacity", this first pass runs in guarded mode.
        fill_placed_guarded = 0
        mode_first = (
            "guarded"
            if params.final_fill_mode in ("guarded", "guarded_then_capacity")
            else "capacity_only"
        )

        for gi in leftover_indices:
            g = groups[gi]
            if g.count == 0:
                continue

            step_mins = int(minute_delta.total_seconds() // 60)
            dur_bins = int(g.dur_min / max(1, step_mins))
            if dur_bins <= 0:
                continue
            need_nodes = g.nodes

            N_positions = len(minute_ends) - dur_bins + 1
            if N_positions <= 0:
                continue

            # Rolling sum of price_fill_min over windows of length dur_bins
            csum = np.concatenate(([0.0], np.cumsum(price_fill_min)))
            roll = csum[dur_bins:] - csum[:-dur_bins]
            order_idx_full = np.argsort(roll)

            job_cursor = 0
            for t0 in order_idx_full:
                if job_cursor >= g.count:
                    break

                # Capacity check (always)
                if seg.min(t0, t0 + dur_bins) < need_nodes:
                    continue

                p_one = float(g.job_p_kw[job_cursor])

                if mode_first == "capacity_only":
                    ok = True
                else:
                    # Guarded: enforce Stage A demand thresholds
                    touched, slot_weights = _touched_slots_and_weights(t0, dur_bins)
                    ok = True
                    for j_slot, frac in zip(touched, slot_weights):
                        delta = p_one * frac
                        # Distribution headroom
                        if headroom_dist[j_slot] - delta < 0:
                            ok = False
                            break
                        # G&T headroom where applicable
                        if gt_mask15[j_slot] and np.isfinite(headroom_gt[j_slot]):
                            if headroom_gt[j_slot] - delta < 0:
                                ok = False
                                break

                if not ok:
                    continue

                # Commit job
                seg.add(t0, t0 + dur_bins, -need_nodes)
                cpu_kw_min[t0:t0 + dur_bins] += p_one

                if mode_first == "guarded":
                    # Update headroom so subsequent jobs see reduced margin
                    touched, slot_weights = _touched_slots_and_weights(t0, dur_bins)
                    for j_slot, frac in zip(touched, slot_weights):
                        delta = p_one * frac
                        headroom_dist[j_slot] -= delta
                        if gt_mask15[j_slot] and np.isfinite(headroom_gt[j_slot]):
                            headroom_gt[j_slot] -= delta

                start_ts = minute_ends[t0] - minute_delta
                end_ts   = start_ts + pd.Timedelta(minutes=g.dur_min)
                jid = g.job_ids[job_cursor]
                placements.append((jid, start_ts, end_ts, g.nodes, "cpu_exclusive"))

                job_cursor += 1
                fill_placed_guarded += 1

            if job_cursor > 0:
                groups[gi] = JobGroup(
                    dur_min=g.dur_min,
                    nodes=g.nodes,
                    job_ids=g.job_ids[job_cursor:],
                    job_p_kw=g.job_p_kw[job_cursor:],
                    count=g.count - job_cursor,
                )

        remaining_jobs_after = sum(g.count for g in groups)
        print(f"[FINAL FILL] Placed in final fill (pass 1, {mode_first}): {fill_placed_guarded:,}")
        print(f"[FINAL FILL] Unscheduled after pass 1: {remaining_jobs_after:,}")

        # ---------- PASS 2: optional capacity-only on leftovers ----------
        if params.final_fill_mode == "guarded_then_capacity" and remaining_jobs_after > 0:
            print(f"[FINAL FILL CAPACITY] Starting capacity-only pass for {remaining_jobs_after:,} leftover jobs")

            # We can ignore demand spikes for this pass (capacity-only feasibility)
            mu_min_cap = _broadcast_month_penalty_to_minutes(mu_by_month, months_min)
            price_fill_min_cap = price_base_min + mu_min_cap

            leftover_indices_cap = [i for i, g in enumerate(groups) if g.count > 0]
            leftover_indices_cap.sort(key=_group_energy_per_job, reverse=True)

            fill_placed_cap = 0

            for gi in leftover_indices_cap:
                g = groups[gi]
                if g.count == 0:
                    continue

                step_mins = int(minute_delta.total_seconds() // 60)
                dur_bins = int(g.dur_min / max(1, step_mins))
                if dur_bins <= 0:
                    continue
                need_nodes = g.nodes

                N_positions = len(minute_ends) - dur_bins + 1
                if N_positions <= 0:
                    continue

                # Rolling sum of price_fill_min_cap over windows of length dur_bins
                csum_cap = np.concatenate(([0.0], np.cumsum(price_fill_min_cap)))
                roll_cap = csum_cap[dur_bins:] - csum_cap[:-dur_bins]
                order_idx_full_cap = np.argsort(roll_cap)

                job_cursor = 0
                for t0 in order_idx_full_cap:
                    if job_cursor >= g.count:
                        break

                    # Capacity-only check
                    if seg.min(t0, t0 + dur_bins) < need_nodes:
                        continue

                    p_one = float(g.job_p_kw[job_cursor])

                    # Commit job (no demand guard here)
                    seg.add(t0, t0 + dur_bins, -need_nodes)
                    cpu_kw_min[t0:t0 + dur_bins] += p_one

                    start_ts = minute_ends[t0] - minute_delta
                    end_ts   = start_ts + pd.Timedelta(minutes=g.dur_min)
                    jid = g.job_ids[job_cursor]
                    placements.append((jid, start_ts, end_ts, g.nodes, "cpu_exclusive"))

                    job_cursor += 1
                    fill_placed_cap += 1

                if job_cursor > 0:
                    groups[gi] = JobGroup(
                        dur_min=g.dur_min,
                        nodes=g.nodes,
                        job_ids=g.job_ids[job_cursor:],
                        job_p_kw=g.job_p_kw[job_cursor:],
                        count=g.count - job_cursor,
                    )

            remaining_jobs_after_cap = sum(g.count for g in groups)
            print(f"[FINAL FILL CAPACITY] Placed in capacity-only pass: {fill_placed_cap:,}")
            print(f"[FINAL FILL CAPACITY] Unscheduled after final fill: {remaining_jobs_after_cap:,}")


    # --- Build outputs (15-min series) and compute exact campus cost with billing ---
    cpu15 = cpu_kw_min_to_15(cpu_kw_min)
    cpu15_series = pd.Series(cpu15, index=slot_ends15, name="scheduled_cpu_kw")

    cost = compute_cost_exact(
        slot_ends15,
        attrs15,
        res_a.merged_on_grid.loc[slot_ends15],
        cpu15_series,
        eps
    )

    sched_df = pd.DataFrame(
        placements,
        columns=["job_id", "start_ts", "end_ts", "nodes", "node_type"]
    )

    # --- GO/NO-GO summary ---
    total_jobs = int(res_c.jobs_sched.shape[0])
    scheduled_jobs = int(sched_df.shape[0])
    coverage = 100.0 * (scheduled_jobs / max(1, total_jobs))

    stageA_K = res_d.month_table[["K_dist_kw", "K_gt_kw"]].copy()
    stageB_K = cost.month_table[["K_dist_kw", "K_gt_kw"]].copy()
    inc_months = (stageB_K > stageA_K).any(axis=1)
    inc_month_list = stageB_K.index[inc_months].tolist()

    print("=== PART E′: PRIMAL–DUAL PACK (GO/NO-GO) ===")
    print(f"[GO] Scheduled jobs: {scheduled_jobs:,} / {total_jobs:,}  ({coverage:.2f}%)")
    print(f"[GO] CPU 15-min peak kW: {cpu15_series.max():.3f}")
    print(f"[GO] Objective (campus) usd: {cost.objective_usd:,.2f}")
    if inc_month_list:
        print(f"[WARN] Months with higher billed peaks vs Stage A: {inc_month_list}")
    else:
        print("[GO] No billed-peak increases vs Stage A")

    # Marginal dollar impact vs baseline+background only
    baseline_only = compute_cost_exact(
        slot_ends15,
        attrs15,
        res_a.merged_on_grid.loc[slot_ends15],
        pd.Series(0.0, index=slot_ends15),
        eps
    )
    print(f"[INFO] Marginal cost of scheduled CPU subset usd: {cost.objective_usd - baseline_only.objective_usd:,.2f}")

    # Month quota diagnostics
    targets_kwh = pd.Series(month_targets_kwh, name="target_cpu_kwh")
    placed_kwh = (cpu15_series * 0.25).groupby(cpu15_series.index.strftime("%Y-%m")).sum()
    quota_df = pd.concat([targets_kwh, placed_kwh.rename("placed_cpu_kwh")], axis=1).fillna(0.0)
    quota_df["ratio"] = np.where(
        quota_df["target_cpu_kwh"] > 0,
        quota_df["placed_cpu_kwh"] / quota_df["target_cpu_kwh"],
        np.nan,
    )
    print("[QUOTAS] (first 6 rows)")
    print(quota_df.head(6).to_string())

    go = True  # capacity & guards enforced; raise to False only on hard violations
    return StageBPDResult(
        scheduled=sched_df,
        cpu_kw_15=cpu15_series,
        cost=cost,
        stats={
            "scheduled_jobs": scheduled_jobs,
            "coverage_pct": coverage,
            "cpu_peak_kw": float(cpu15_series.max()),
            "objective_usd": float(cost.objective_usd),
        },
        go=go
    )