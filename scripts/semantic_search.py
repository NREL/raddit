#!/usr/bin/env python
import os
import polars as pl
import pandas as pd
from datetime import datetime, timedelta
from pymilvus import MilvusClient
from tqdm import tqdm
import math
from pathlib import Path

def load_and_prepare_data(chunk_dir: str) -> pl.DataFrame:
    paths = sorted(Path(chunk_dir).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet files in {chunk_dir}")

    dfs = []
    current_id = 0
    for p in tqdm(paths, desc="Load chunks"):
        t = pl.read_parquet(p)
        t = t.with_columns([
            pl.col("enc_embedding_int8").alias("vector"),
            pl.arange(current_id, current_id + t.height).alias("row_id")
        ])
        current_id += t.height

        dfs.append(
            t.select([
                "row_id",
                "avg_power_per_node", "wallclock_used_sec",
                "submit_time", "end_time",
                "vector"
            ])
        )
    df = pl.concat(dfs, how="vertical")
    print(f"Loaded {len(df):,} rows from {len(paths)} chunks")
    
    # Convert time columns to epoch seconds.
    df = df.with_columns([
        (pl.col("end_time").cast(pl.Datetime).dt.timestamp("us") / 1_000_000).alias("end_time_epoch"),
        (pl.col("submit_time").cast(pl.Datetime).dt.timestamp("us") / 1_000_000).alias("submit_time_epoch")
    ])
    
    return df

def run_hourly_iteration_incremental(df, client, start_date_str, end_date_str, batch_size=10000, populate=True):
    """
    Iterates hourly between start_date and end_date.
    
    If 'populate' is True:
      - Initially inserts all jobs that ended before start_date.
      - Incrementally inserts new jobs (filtered by end_time_epoch) in batches.
    
    For each hour, the function:
      - Defines a test set as jobs submitted in that hour.
      - Performs a vector search on the "vector" field using a FLAT index,
        retrieving the nearest 5 neighbors.
      - Computes:
            predicted_power = weighted average of the neighbors’ avg_power_per_node,
            power_min = min(avg_power_per_node of neighbors),
            power_max = max(avg_power_per_node of neighbors),
            predicted_runtime = weighted average of the neighbors’ wallclock_used_sec,
            runtime_min = min(wallclock_used_sec of neighbors),
            runtime_max = max(wallclock_used_sec of neighbors),
      - Saves the predictions for that split time to a CSV file (filename includes the split time).
    
    Returns a list of result dictionaries.
    """
    start_date = datetime.fromisoformat(start_date_str)
    end_date   = datetime.fromisoformat(end_date_str)
    current_time = start_date
    results = []
    start_date_epoch = int(start_date.timestamp())
    
    def insert_batches(df_to_insert):
        total_rows = df_to_insert.height
        for start in tqdm(range(0, total_rows, batch_size), total=math.ceil(total_rows/batch_size)):
            batch_df = df_to_insert.slice(start, batch_size)
            data = [
                {
                    "id": row["row_id"],  
                    "vector": list(row["vector"]),
                    "avg_power_per_node": row["avg_power_per_node"],
                    "wallclock_used_sec": row["wallclock_used_sec"],
                    "end_time_epoch": row["end_time_epoch"]
                }
                for row in batch_df.to_dicts()
            ]
            client.insert(collection_name="power_and_runtime_collection", data=data)
            client.flush(collection_name="power_and_runtime_collection")
            print(f"Inserted batch rows {start} to {start + min(batch_size, total_rows - start)}")
    
    if populate:
        print('Initializing vector database')
        initial_train_df = df.filter(pl.col("end_time_epoch") < start_date_epoch)
        if initial_train_df.height > 0:
            insert_batches(initial_train_df)
            print(f"Inserted initial training records: {initial_train_df.height}")
        else:
            print("No initial training records to insert")
    else:
        print("Skipping initial population; using existing data.")
    
    last_split_time_epoch = start_date_epoch
    
    while current_time < end_date:
        split_time = current_time
        split_time_epoch = int(split_time.timestamp())
        next_hour = split_time + timedelta(hours=1)
        next_hour_epoch = int(next_hour.timestamp())
        
        print(f"\nProcessing split time: {split_time.isoformat()}")
        if populate:
            # Insert incremental new training records.
            new_train_df = df.filter(
                (pl.col("end_time_epoch") >= last_split_time_epoch) &
                (pl.col("end_time_epoch") < split_time_epoch)
            )
            if new_train_df.height > 0:
                insert_batches(new_train_df)
                print(f"Inserted {new_train_df.height} new training records")
            else:
                print("No new training records to insert")
            
            last_split_time_epoch = split_time_epoch
        
        # Define the test set: jobs submitted between split_time and next_hour.
        test_df = df.filter(
            (pl.col("submit_time_epoch") >= split_time_epoch) &
            (pl.col("submit_time_epoch") < next_hour_epoch)
        )
        print(f"Test set size: {test_df.height}")
        if test_df.height == 0:
            current_time = next_hour
            continue
        
        test_rows = test_df.to_dicts()
        # Build a batch of query vectors for the entire test set.
        query_vectors = [
            row["vector"].tolist() if hasattr(row["vector"], "tolist") else list(row["vector"])
            for row in test_rows
        ]
        # Filter expression: only retrieve entities with end_time_epoch less than current split time.
        expr = f"end_time_epoch < {split_time_epoch}"
        
        # Perform a batch search retrieving 5 nearest neighbors.
        search_results = client.search(
            collection_name="power_and_runtime_collection",
            data=query_vectors,
            filter=expr,
            limit=5,
            output_fields=["avg_power_per_node", "wallclock_used_sec", "end_time_epoch"]
        )
        
        predictions = []
        for row, res in zip(test_rows, search_results):
            if res and len(res) > 0:
                neighbor_details = []
                weights = []
                power_values = []
                runtime_values = []
                for hit in res:
                    d = hit["distance"]
                    w = max(1, (d - 0.9) * 100)
                    weights.append(w)
                    p = hit['entity'].get("avg_power_per_node", 0)
                    rt = hit['entity'].get("wallclock_used_sec", 0)
                    power_values.append(p)
                    runtime_values.append(rt)
                    neighbor_details.append({
                        "distance": d,
                        "weight": w,
                        "avg_power_per_node": p,
                        "wallclock_used_sec": rt
                    })
                total_weight = sum(weights)
                predicted_power = (sum(w * p for w, p in zip(weights, power_values)) / total_weight
                                   if total_weight > 0 else None)
                predicted_runtime = (sum(w * rt for w, rt in zip(weights, runtime_values)) / total_weight
                                   if total_weight > 0 else None)
                power_min = min(power_values)
                power_max = max(power_values)
                runtime_min = min(runtime_values)
                runtime_max = max(runtime_values)
                
                predictions.append({
                    "avg_power_per_node": row["avg_power_per_node"],
                    "predicted_power": predicted_power,
                    "wallclock_used_sec": row["wallclock_used_sec"],
                    "predicted_runtime": predicted_runtime,
                    "power_min": power_min,
                    "power_max": power_max,
                    "runtime_min": runtime_min,
                    "runtime_max": runtime_max,
                    "neighbors": neighbor_details,
                    "submit_time_epoch": row["submit_time_epoch"]
                })
            else:
                predictions.append({
                    "avg_power_per_node": row["avg_power_per_node"],
                    "predicted_power": None,
                    "wallclock_used_sec": row["wallclock_used_sec"],
                    "predicted_runtime": None,
                    "power_min": None,
                    "power_max": None,
                    "runtime_min": None,
                    "runtime_max": None,
                    "neighbors": None,
                    "submit_time_epoch": row["submit_time_epoch"]
                })
        
        # Save predictions for this split time to a CSV file.
        os.makedirs("../data/semantic_search", exist_ok=True)
        csv_filename = f"../data/semantic_search/results_{split_time.strftime('%Y-%m-%dT%H-%M-%S')}.csv"
        os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
        df_results = pd.DataFrame(predictions)
        df_results.to_csv(csv_filename, index=False)
        print(f"Saved results for split time {split_time.isoformat()} to {csv_filename}")
        
        results.append({
            "split_time": split_time,
            "num_new_train": new_train_df.height if populate else 0,
            "num_test": test_df.height,
            "predictions": predictions
        })
        
        print(f"Split Time: {split_time.isoformat()}, Test Count: {test_df.height}")
        current_time = next_hour
        
    return results


def main():
    client = MilvusClient("milvus_power_and_runtime.db")
    
    if client.has_collection(collection_name="power_and_runtime_collection"):
        print("Using existing collection 'power_and_runtime_collection'.")
        populate=False
    else:
        print("Creating new collection 'power_and_runtime_collection' and populating data.")
        client.create_collection(collection_name="power_and_runtime_collection", dimension=4096)
        client.create_index(
            collection_name="power_and_runtime_collection",
            index_params=[{
                "field_name": "vector",
                "index_type": "FLAT",
                "metric_type": "COSINE",
                "params": {}
            }]
        )
        populate = True
    
    chunk_dir = "../data/encrypted_embeddings"
    df = load_and_prepare_data(chunk_dir)
    
    # Run hourly iterations. If the collection is already populated, we skip insertion.
    results = run_hourly_iteration_incremental(df, client, "2024-08-01", "2025-02-01", populate=populate)
    
    print("\nSummary of results:")
    for res in results:
        print(f"Split Time: {res['split_time'].isoformat()}, New Train: {res['num_new_train']}, Test: {res['num_test']}")
    
if __name__ == "__main__":
    main()