from hpcopt.part_f_costing import compare_stageA_B_actual

from hpcopt import run_part_a
from hpcopt.part_b import run_part_b
from hpcopt.part_c import run_part_c
from hpcopt.stage_a_lp import run_stage_a_lp

import numpy as np

res_a = run_part_a("config.yaml")
res_b = run_part_b(res_a)
res_c = run_part_c(res_a)
res_d = run_stage_a_lp(res_a, res_b, res_c)

sa_res, sb_res, actual_res = compare_stageA_B_actual(
    res_a,
    res_b,
    res_d=res_d,
    cpu15_csv="../results/hpc_sched_results/stageB_pd_cpu15.csv",
    actual_csv_path="../data/actual_demand.csv",
    out_dir="../results/hpc_sched_results"
)

dt_h = float(sb_res["cfg_billing"].get("interval_minutes", 15)) / 60.0

price = sb_res["attrs15"]["tou_price_per_kwh"].to_numpy(float)
label = sb_res["attrs15"]["tou_label"].to_numpy(str)
is_on = (label == "onpeak_weekday_day")

dP = sb_res["net_kw_15"] - actual_res["aligned_kw"]          # kW
dE = dP * dt_h                                               # kWh-equivalent per interval
dCost = dE * price                                           # $

print("ΔkWh on-peak:  ", dE[is_on].sum())
print("ΔkWh off-peak: ", dE[~is_on].sum())
print("Δ$  on-peak:   ", dCost[is_on].sum())
print("Δ$  off-peak:  ", dCost[~is_on].sum())
print("Δ$  total:     ", dCost.sum())


actual_aligned = actual_res["aligned_kw"]
print("fraction aligned actual == 0:", np.mean(actual_aligned == 0.0))
print("min aligned actual:", actual_aligned.min(), "median:", np.median(actual_aligned))


attrs = sb_res["attrs15"]
dt_h = float(sb_res["cfg_billing"].get("interval_minutes", 15)) / 60.0

dE = (sb_res["net_kw_15"] - actual_res["aligned_kw"]) * dt_h

is_on   = attrs["tou_label"].eq("onpeak_weekday_day").to_numpy()
is_wknd = attrs["is_weekend"].to_numpy()

print("ΔkWh weekends:         ", dE[is_wknd].sum())
print("ΔkWh weekday off-peak: ", dE[(~is_wknd) & (~is_on)].sum())
print("ΔkWh weekday on-peak:  ", dE[is_on].sum())
print("ΔkWh total:            ", dE.sum())


dt_h = float(sb_res["cfg_billing"].get("interval_minutes", 15)) / 60.0
idx = sb_res["idx_15"]

# Stage B components
merged15 = res_a.merged_on_grid.loc[idx][["baseline_kw","background_kw"]]
base_bg = (merged15["baseline_kw"] + merged15["background_kw"]).to_numpy(float)

cpu_sb = (sb_res["net_kw_15"] - base_bg)          # scheduled_cpu_kw (what Stage B adds)
cpu_sb_kwh = cpu_sb.sum() * dt_h

# "Implied jobs+residual" in the meter, using the same base+bg model
actual = actual_res["aligned_kw"]
resid_actual = (actual - base_bg)                 # what the meter has above base+bg
resid_actual_kwh = resid_actual.sum() * dt_h

print("StageB scheduled CPU kWh:", cpu_sb_kwh)
print("Meter residual kWh (Actual - base-bg):", resid_actual_kwh)
print("ΔkWh (CPU - residual):", cpu_sb_kwh - resid_actual_kwh)

dt_h = float(sb_res["cfg_billing"].get("interval_minutes", 15)) / 60.0
idx = sb_res["idx_15"]
attrs = sb_res["attrs15"]

merged15 = res_a.merged_on_grid.loc[idx][["baseline_kw","background_kw"]]
basebg = (merged15["baseline_kw"] + merged15["background_kw"]).to_numpy(float)

actual = actual_res["aligned_kw"]
resid = actual - basebg  # "meter residual"

print("resid min:", resid.min())
print("fraction resid<0:", np.mean(resid < 0))
print("kWh of negative resid (magnitude):", np.clip(-resid, 0, None).sum() * dt_h)

is_wknd = attrs["is_weekend"].to_numpy()
print("mean(basebg-actual) weekends kW:", np.mean((basebg - actual)[is_wknd]))
print("kWh(basebg-actual) weekends:", ((basebg - actual) * dt_h)[is_wknd].sum())
