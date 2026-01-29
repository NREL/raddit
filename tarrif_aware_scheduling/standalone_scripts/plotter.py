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

from hpcopt.stage_a_plots import plot_stage_a_timeseries


# ---------- Run Parts A–D ----------
t0 = time.time()
res_a = run_part_a("config.yaml")
res_b = run_part_b(res_a)
res_c = run_part_c(res_a)
res_d = run_stage_a_lp(res_a, res_b, res_c)

_ = plot_stage_a_timeseries(
    res_a, res_b, res_d,
    out_path="../results/hpc_sched_results/stageA_y_kw.png",
    save_csv="../results/hpc_sched_results/stageA_y_kw.csv",
    show=False,
    shade_gt=False
)

print("\n[CHECK] Stage A y(t) plot written and series exported.")
