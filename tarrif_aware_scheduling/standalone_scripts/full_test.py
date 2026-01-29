import os
import time
import shutil
import numpy as np
import pandas as pd

from hpcopt import run_part_a
from hpcopt.part_b import run_part_b
from hpcopt.part_c import run_part_c
from hpcopt.stage_a_lp import run_stage_a_lp
from hpcopt.stage_b_pd import run_stage_b_primal_dual, PDParams

# ---------- Helpers ----------
def _month_targets_from_stage_a(res_d: "StageAResult") -> pd.Series:
    # Envelope y_kw on 15-min grid -> monthly kWh targets
    y = res_d.y_envelope  # pd.Series indexed by 15-min slot ends
    kwh = (y * 0.25).groupby(y.index.strftime("%Y-%m")).sum()
    kwh.name = "target_cpu_kwh"
    return kwh

# ---------- Run Parts A–D ----------
t0 = time.time()
res_a = run_part_a("config.yaml")
res_b = run_part_b(res_a)
res_c = run_part_c(res_a)
res_d = run_stage_a_lp(res_a, res_b, res_c)

# ---------- Checkpoint/output directories ----------
out_dir = res_a.cfg.get("outputs", {}).get("directory", "../results")
os.makedirs(out_dir, exist_ok=True)
ckpt_dir = os.path.join(out_dir, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)


params = PDParams(
    # Resolution / outer loops
    minute_freq="1min",
    iterations=10,

    # Price field shaping
    alpha_gt=4.0,
    demand_margin_kw=10.0,
    step_spike_strength=100.0,

    # Month quota duals
    mu_gamma=0.0002,
    month_quota_slack=0.03,

    # Billed-peak policy
    forbid_step_increase=False,
    step_penalty_mult=0.0,

    # Seeding strategy
    seed_order="energy_desc",

    verbose=True,

    # Candidate window budget (broad but not wasteful)
    per_group_sample_factor=10,
    max_candidates_per_group=2_000_000,
    cand_base_min=217_000,
    cand_sqrt_coeff=60.0,
    cand_log_coeff=600.0,
    cand_frac_of_grid=1,

    # Batch placement
    batch_cap=1,
    cand_ramp_mode = "linear",
    cand_ramp_start_frac = 0.5,
    cand_ramp_end_frac   = 1.0,
)

res_e_pd = run_stage_b_primal_dual(res_a, res_b, res_c, res_d, params)
t_all = time.time() - t0

# ---------- Checks & Reporting ----------
print("\n=== PART E′: CHECKS ===")
print("[CHECK] scheduled rows:", len(res_e_pd.scheduled))
total_jobs = int(res_c.jobs_sched.shape[0])
cov = 100.0 * len(res_e_pd.scheduled) / max(1, total_jobs)
print(f"[CHECK] coverage: {len(res_e_pd.scheduled):,} / {total_jobs:,} ({cov:.2f}%)")

print("[CHECK] cpu series stats:\n", res_e_pd.cpu_kw_15.describe())
print("[CHECK] objective usd:", f"{res_e_pd.cost.objective_usd:,.2f}")

print("\n[Stage A K_dist/K_gt]:")
print(res_d.month_table[["K_dist_kw","K_gt_kw"]].to_string())

print("\n[Stage B′ K_dist/K_gt]:")
print(res_e_pd.cost.month_table[["K_dist_kw","K_gt_kw"]].to_string())

# Warn if Stage B′ increased any billed peaks vs Stage A
stageA_K = res_d.month_table[["K_dist_kw","K_gt_kw"]]
stageB_K = res_e_pd.cost.month_table[["K_dist_kw","K_gt_kw"]]
inc_months = (stageB_K > stageA_K).any(axis=1)
inc_list = stageB_K.index[inc_months].tolist()
if inc_list:
    print(f"\n[WARN] Months with higher billed peaks vs Stage A: {inc_list}")
else:
    print("\n[GO] No billed-peak increases vs Stage A")

# Quota diagnostics: placed vs target (from Stage A envelope)
targets = _month_targets_from_stage_a(res_d)
placed = (res_e_pd.cpu_kw_15 * 0.25).groupby(res_e_pd.cpu_kw_15.index.strftime("%Y-%m")).sum()
quota_df = pd.concat([targets, placed.rename("placed_cpu_kwh")], axis=1).fillna(0.0)
quota_df["ratio"] = np.where(quota_df["target_cpu_kwh"] > 0,
                             quota_df["placed_cpu_kwh"] / quota_df["target_cpu_kwh"], np.nan)
print("\n[QUOTAS] target vs placed (first 6 rows):")
print(quota_df.head(6).to_string())

print(f"\n[INFO] Total wall time: {t_all:.1f}s")

# Totals
total_target = float((res_d.y_envelope * 0.25).sum())
total_placed = float((res_e_pd.cpu_kw_15 * 0.25).sum())
print("TOTAL target kWh (Stage A):", total_target)
print("TOTAL placed kWh (Stage B):", total_placed)
print("Gap kWh (unscheduled or guarded):", total_target - total_placed)

# Cross-check: total job energy from Part C (should ~== total_target)
job_energy_kwh = float(
    (res_c.jobs_sched["avg_power_kw"].to_numpy() *
     res_c.jobs_sched["duration_seconds"].to_numpy() / 3600.0).sum()
)
print("TOTAL job energy kWh (Part C):", job_energy_kwh)

# How much energy is in the unscheduled jobs?
scheduled_ids = set(res_e_pd.scheduled["job_id"].astype(str))
unsched = res_c.jobs_sched[~res_c.jobs_sched["job_id"].astype(str).isin(scheduled_ids)]
unsched_energy_kwh = float((unsched["avg_power_kw"] * unsched["duration_seconds"] / 3600.0).sum())
print("Unscheduled energy kWh:", unsched_energy_kwh)
print("Share unscheduled (%):", 100.0 * unsched_energy_kwh / max(1.0, total_target))


# ---------- Write outputs ----------
sched_path = os.path.join(out_dir, "stageB_pd_schedule.csv")
cpu15_path = os.path.join(out_dir, "stageB_pd_cpu15.csv")
res_e_pd.scheduled.to_csv(sched_path, index=False)
res_e_pd.cpu_kw_15.to_frame("scheduled_cpu_kw").to_csv(cpu15_path)
print(f"[WRITE] schedule -> {sched_path}")
print(f"[WRITE] cpu 15-min -> {cpu15_path}")

# ---------- (Optional) Show last few progress lines after completion ----------
log_path = os.path.join(ckpt_dir, "stageB_progress_log.csv")
if os.path.exists(log_path):
    try:
        tail_n = 5
        df_log = pd.read_csv(log_path)
        print(f"\n[PROGRESS LOG tail {tail_n}]")
        print(df_log.tail(tail_n).to_string(index=False))
    except Exception as e:
        print(f"[WARN] Could not read progress log: {e}")
