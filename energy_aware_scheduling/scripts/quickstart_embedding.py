#!/usr/bin/env python3
"""
quickstart_embedding_semantic.py
--------------------------------
End-to-end demo of the *real* embedding ➜ vector-DB ➜ semantic-search
workflow, on **10 randomly-selected jobs**.

✔  Uses CUDA (or Apple MPS) automatically if available, otherwise CPU.  
✔  Creates an *ephemeral* Milvus Lite DB (`quickstart_vectors.db`) so it
   will not overwrite any project data.  
✔  Prints per-job power & runtime predictions for the 5 query jobs.

Run with:

    python quickstart_embedding_semantic.py
"""

import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from pymilvus import MilvusClient

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from optimum.quanto import QuantizedModelForCausalLM, qint8

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
HIST_TRACE = Path("../data/historic_job_trace.parquet")
COLLECTION   = "quickstart_power_runtime"
DB_PATH      = "../data/quickstart_output/quickstart_vectors.db" 
MODEL_NAME   = "Linq-AI-Research/Linq-Embed-Mistral"
N_EMBED_DIM  = 4096


# ------------------------------------------------------------------
# DEVICE SELECTION 
# ------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"[INFO] Using device: {DEVICE}")

# ------------------------------------------------------------------
# LOAD 10 RANDOM JOBS 
# ------------------------------------------------------------------
sample_df = pd.read_parquet(HIST_TRACE).sample(n=10, random_state=42).reset_index(drop=True)

# helper to stringify job
def render_job(row) -> str:
    parts = [
        f"User: {row.user}",
        f"Account: {row.account}",
        f"Partition: {row.partition}",
        f"Job Type: {row.job_type}",
        f"Job Name: {row['name']}",
        f"QOS: {row.qos}",
        f"Submit Line: {row.submit_line}",
        "Script:",
        str(row.script),
    ]
    return "\n\n".join(parts)

job_strings = [render_job(r) for _, r in sample_df.iterrows()]

# ------------------------------------------------------------------
# EMBEDDINGS 
# ------------------------------------------------------------------
print("Loading tokenizer & model…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
print("Quantizing model…")
qmodel = QuantizedModelForCausalLM.quantize(model, weights=qint8, exclude='lm_head')
del model


def embed(texts):
    tok = tokenizer(
        texts,
        max_length=2048,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        out = qmodel(**tok).last_hidden_state
        # last token pooling
        seq_len = tok["attention_mask"].sum(dim=1) - 1
        pooled  = out[torch.arange(out.size(0), device=DEVICE), seq_len]
        pooled  = F.normalize(pooled, p=2, dim=1)
    return pooled.cpu().numpy()

print("Embedding 10 jobs…")
embeds = embed(job_strings)                    # (10, 4096)

# ------------------------------------------------------------------
# BUILD MINI VECTOR DB 
# ------------------------------------------------------------------
os.makedirs("../data/quickstart_output", exist_ok=True)

# clean old DB if exists
if Path(DB_PATH).exists():
    shutil.rmtree(DB_PATH)

client = MilvusClient(DB_PATH)
print("Milvus Lite DB created.")

if client.has_collection(COLLECTION):
    client.drop_collection(COLLECTION)
client.create_collection(
    collection_name=COLLECTION,
    dimension=N_EMBED_DIM,
    index_type="FLAT",
    metric_type="COSINE"
)

# split 5 → DB, 5 → query
db_rows   = list(range(5))
query_rows = list(range(5, 10))

# Insert reference vectors
to_insert = []
for rid in db_rows:
    to_insert.append({
        "id": int(rid),
        "vector": embeds[rid].tolist(),
        "avg_power_per_node": float(sample_df.loc[rid, "avg_power_per_node"]),
        "wallclock_used_sec": float(sample_df.loc[rid, "wallclock_used_sec"]),
    })
client.insert(collection_name=COLLECTION, data=to_insert)
client.flush(collection_name=COLLECTION)
print(f"[INFO] Inserted {len(db_rows)} vectors into collection.")

# ------------------------------------------------------------------
# QUERY + PREDICT 
# ------------------------------------------------------------------
def predict_for_row(idx: int):
    vec = embeds[idx].tolist()
    search_res = client.search(
        collection_name=COLLECTION,
        data=[vec],
        limit=5,
        output_fields=["avg_power_per_node", "wallclock_used_sec"]
    )[0]

    weights, powers, runtimes = [], [], []
    for hit in search_res:
        d = hit.distance
        w = max(1.0, (d - 0.9) * 100)          # Weight neighbors by similarity
        weights.append(w)
        powers.append(hit.entity["avg_power_per_node"])
        runtimes.append(hit.entity["wallclock_used_sec"])

    wsum = sum(weights)
    pred_power   = sum(w * p  for w, p  in zip(weights, powers))   / wsum
    pred_runtime = sum(w * rt for w, rt in zip(weights, runtimes)) / wsum

    return pred_power, pred_runtime

print("\n=== Predictions for 5 query jobs ===")
for idx in query_rows:
    true_pwr = sample_df.loc[idx, "avg_power_per_node"]
    true_rt  = sample_df.loc[idx, "wallclock_used_sec"]
    pred_pwr, pred_rt = predict_for_row(idx)

    print(f"- Job {idx:02d}: "
          f"True P={true_pwr:7.1f} W  Pred P={pred_pwr:7.1f} W   |   "
          f"True RT={true_rt/3600:5.2f} h  Pred RT={pred_rt/3600:5.2f} h")

print("\n[✓] Quick-start embedding + semantic search complete.\n"
      f"    Vector DB stored at: {DB_PATH}")
