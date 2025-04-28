# Energy-Aware HPC Job Scheduling — **Artifact Repository**

This repo accompanies the SC’25 submission **“Energy-Aware HPC Job Scheduling: Bridging the Gap from Theory to Practice.”**  
It contains **all computational artifacts** needed to reproduce the paper’s figures (aside from the *FastSim* simulator, which is still under license review).

---

## Paper Contributions
| ID | Contribution |
|----|--------------|
| **C1** | Per-job power & runtime prediction using LLM embeddings of enriched job scripts. |
| **C2** | Extensions to *FastSim*: high-fidelity, high-throughput Slurm simulator (not yet open-source). |
| **C3** | Lightweight energy-aware scheduling strategy integrated via Slurm site factor. |

---

## Repository Layout (top-level)

| Path | What it holds |
|------|---------------|
| `data/` | All static datasets (job trace, simulation outputs, embeddings, etc.). |
| `scripts/` | All executable code (model training, semantic search, etc.). |
| `notebooks/` | Jupyter notebooks for analysis & figure generation. |
| `requirements.txt` | List of Python dependencies. |
| `scripts/quickstart.py` | **One-click** simplified, limited reproduction on a sample dataset. |

---

## Quick Start (5 minutes)

> **Goal:** Reproduce a *miniature* version of the power & runtime prediction process using down-sampled data.

```bash
# 1) Clone and enter the repo
git clone https://github.com/nrel/raddit.git
cd raddit

# 2) Create Python 3.9+ environment and install deps
python -m venv raddit_env
source raddit_env/bin/activate            # on Windows: raddit_env\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# 3) Run the sample pipeline
cd scripts
python quickstart_sample.py
