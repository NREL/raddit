# Energy-Aware HPC Job Scheduling — **Artifact Repository**

This repository accompanies the SC’25 paper  
**“Energy-Aware HPC Job Scheduling: Bridging the Gap from Theory to Practice.”**

It contains **all computational artifacts** needed to reproduce the paper’s
results—except the *FastSim* simulator, which is still in license review
(the corresponding input/output datasets are provided).

---

## Paper Contributions

| ID  | Contribution                                                                                                         |
|-----|-----------------------------------------------------------------------------------------------------------------------|
| **C1** | Per-job power & runtime prediction with LLM embeddings of enriched job scripts.                                    |
| **C2** | *FastSim* extension: high-fidelity, high-throughput Slurm simulator (not yet open-source).                         |
| **C3** | Lightweight energy-aware scheduling strategy implemented via Slurm site-factor priority.                           |

---

## Repository Layout (top-level)

| Path / file                        | What it holds                                                                  |
|-----------------------------------|--------------------------------------------------------------------------------|
| `data/`                            | All static datasets (job trace, simulation outputs, encrypted embeddings, …).  |
| `scripts/`                         | Executable code (model training, semantic search, energy-aware priority, …).   |
| `scripts/quickstart.py`    | **Quick-Start #1** – tiny CPU-only pipeline on a sampled trace of 10k jobs.           |
| `scripts/quickstart_embedding.py` | **Quick-Start #2** – end-to-end embedding ➜ vector DB ➜ semantic search demo. |
| `notebooks/`                       | Jupyter notebooks that generate all paper figures.                              |
| `requirements.txt`                 | Python dependencies (no pinned versions—latest stable is fine).                 |

---

## Environment Setup

```bash
git clone https://github.com/nrel/raddit.git
cd raddit

# Create and activate a fresh Python ≥3.9 env
python -m venv raddit_env
source raddit_env/bin/activate          # Windows: raddit_env\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt