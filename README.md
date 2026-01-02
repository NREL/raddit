# Resource-Aware Data center Digital Twin (RADDiT)  

RADDiT is a modular, real-time digital-twin platform designed to optimize datacenter energy use by linking job-level predictions, validated scheduling simulation, and grid-aware control.

<img width="700" height="442" alt="raddit" src="https://github.com/user-attachments/assets/dcfa0564-62cb-4c1a-bb8d-d836ca1305fc" />

_**RADDiT system loop.** Operational telemetry and job metadata feed a validated scheduling simulator and predictive models, while external grid/energy signals inform an optimization engine that selects control actions (e.g., scheduling priorities and power caps) applied back to the data center._

---

RADDiT aims to evolve datacenters from passive, high-variance loads into **active, intelligent participants** in grid operations.

- **Grid support and resilience:** Aligns job scheduling and control with external energy signals (e.g., behind-the-meter PV, grid prices/demand, utility programs such as FlexConnect).  
- **Operational cost savings:** Reduces peak power draw and demand charges.  
- **Sector-wide impact:** Techniques are intended to scale to DOE HPC facilities, other federal labs, utilities, hyperscalers, and campus microgrids.  
- **Primary audiences include:** DOE HPC and facility managers, grid operators and aggregators, hyperscale DC/infrastructure engineers, campus-scale energy systems and microgrid integrators.

---

## Brief Project Description & Scope

**Core innovations:**
- **Per-Job Power Prediction:** LLM embeddings of enriched job scripts and metadata; similarity-based inference from historical job neighbors (submission-time predictions).  
- **Digital-Twin–Based Control:** Real-time coordination of datacenter power with grid signals using validated scheduling simulations and predictive models.  
- **Intelligent Job Scheduling & Power Capping:**  
  - Reprioritizes jobs in SLURM using predicted power and external energy signals (SiteFactor approach—no Slurm core changes).  
---

## Findings to Date

- Achieved **~17 W median per-job power-prediction error** (~5% relative) on CPU-exclusive jobs.  
- **Validated high-fidelity scheduling simulator** on Kestrel job traces (alignment on throughput, wait-time distributions, and power over time).  
- **Demonstrated energy-aware scheduling** that aligns workload with on-campus PV and a utility program (e.g., PG&E FlexConnect) via multi-objective optimization (power alignment and wait time).

---

## What This Repository Contains

This repo provides the **reproducible research artifacts** for the paper and project components listed above. The **FastSim-based Slurm emulator** used in the study is currently in NREL licensing review; corresponding **inputs/outputs are provided** here to ensure end-to-end reproducibility of figures and results.

| Path / file                        | Contents                                                                          |
|------------------------------------|-----------------------------------------------------------------------------------|
| `data/`                            | Static datasets (job traces, simulator I/O, encrypted embeddings where required). |
| `scripts/`                         | Executables for embedding, semantic search, prediction, and energy-aware priority.|
| `scripts/quickstart.py`            | **Quick Start #1** — sampled CPU-only pipeline on ~10k jobs.                      |
| `scripts/quickstart_embedding.py`  | **Quick Start #2** — end-to-end embedding → vector DB → semantic retrieval demo.  |
| `notebooks/`                       | Jupyter notebooks to regenerate paper figures.                                    |
| `requirements.txt`                 | Python dependencies (versions pinned as needed for reproducibility).              |

> **Note on privacy:** Job scripts and metadata may contain sensitive details. This repository distributes sanitized/aggregated artifacts. All LLM embeddings were computed on-premises.

---

## Environment Setup

    git clone https://github.com/nrel/raddit.git
    cd raddit

    # Create and activate a fresh Python ≥3.9 environment
    python -m venv raddit_env
    source raddit_env/bin/activate        # Windows: raddit_env\Scripts\activate.bat
    pip install --upgrade pip
    pip install -r requirements.txt

---


## License & Software Record

Released under the **BSD 3-Clause License** (see `LICENSE`).  
Covered by **NREL Software Record SWR-23-34**.  
This work builds on prior related work available at [https://github.com/NREL/eagle-jobs/](https://github.com/NREL/eagle-jobs/), which is also covered under the same Software Record.
