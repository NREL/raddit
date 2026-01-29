#!/usr/bin/env python3
import os
import warnings
import concurrent.futures
from typing import Optional

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=UserWarning)

kestrel_df_global: Optional[pd.DataFrame] = None


def init_worker() -> None:
    """Load the pre-preprocessed parquet once per forked worker."""
    global kestrel_df_global
    kestrel_df_global = pd.read_parquet("./kestrel_baseline_data.parquet")


def get_one_hot_encoded_dfs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    column: str,
    is_set: bool = False,
    min_freq: int = 1000):
    if is_set:
        mlb = MultiLabelBinarizer()
        counts = pd.Series(mlb.fit_transform(train_df[column]).sum(axis=0))
        keep_ix = counts[counts >= min_freq].index
        mlb = MultiLabelBinarizer(classes=keep_ix)
        _train = mlb.fit_transform(train_df[column])
        _test = mlb.transform(test_df[column])
        feats = mlb.classes_.astype(str)
    else:
        enc = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_freq)
        _train = enc.fit_transform(train_df[[column]]).toarray()
        _test = enc.transform(test_df[[column]]).toarray()
        feats = enc.get_feature_names_out()
    return (pd.DataFrame(_train, columns=feats, index=train_df.index),
            pd.DataFrame(_test, columns=feats, index=test_df.index))


def split_train_test(df: pd.DataFrame, split_ts, train_days: int, test_hours: int):
    train_df = df.loc[
        (df["end_time"] >= split_ts - pd.Timedelta(days=train_days))
        & (df["end_time"] < split_ts)
    ].reset_index(drop=True)
    test_df = df.loc[
        (df["submit_time"] >= split_ts)
        & (df["submit_time"] < split_ts + pd.Timedelta(hours=test_hours))
    ].reset_index(drop=True)
    return train_df, test_df


def process_split_time(split_ts: pd.Timestamp, train_days: int, test_hours: int):
    """
    Return (split_time, power_file, MAE_power, rt_file, MAE_runtime)
    or None if no data for that window.
    """
    global kestrel_df_global
    train_df, test_df = split_train_test(kestrel_df_global, split_ts, train_days, test_hours)
    if train_df.empty or test_df.empty:
        return None

    oh_pairs = [
        get_one_hot_encoded_dfs(train_df, test_df, "job_type"),
        get_one_hot_encoded_dfs(train_df, test_df, "user"),
        get_one_hot_encoded_dfs(train_df, test_df, "account"),
        get_one_hot_encoded_dfs(train_df, test_df, "partition"),
        get_one_hot_encoded_dfs(train_df, test_df, "modules_set", is_set=True),
        get_one_hot_encoded_dfs(train_df, test_df, "conda_envs_set", is_set=True),
    ]
    oh_train = pd.concat([p[0] for p in oh_pairs], axis=1)
    oh_test = pd.concat([p[1] for p in oh_pairs], axis=1)

    # PCA
    n_comp = 100
    pca = PCA(n_components=n_comp)
    pcs_train = pca.fit_transform(oh_train)
    pcs_test = pca.transform(oh_test)
    pc_cols = [f"PC{i+1}" for i in range(n_comp)]
    train_df = pd.concat([train_df, pd.DataFrame(pcs_train, columns=pc_cols)], axis=1)
    test_df = pd.concat([test_df, pd.DataFrame(pcs_test, columns=pc_cols)], axis=1)

    feature_cols = ["nodes_req", "wallclock_req_seconds", "processors_req", "memory_req_raw"] + pc_cols
    X_tr, X_te = train_df[feature_cols], test_df[feature_cols]

    # Power Model
    y_tr_pwr = train_df["avg_power_per_node"]
    y_te_pwr = test_df["avg_power_per_node"]

    mod_pwr = XGBRegressor()
    mod_pwr.fit(X_tr, y_tr_pwr)
    pred_pwr = mod_pwr.predict(X_te)
    mae_pwr = mean_absolute_error(y_te_pwr, pred_pwr)

    # Runtime Model
    y_tr_rt = train_df["wallclock_used_sec"]
    y_te_rt = test_df["wallclock_used_sec"]

    mod_rt = XGBRegressor()
    mod_rt.fit(X_tr, y_tr_rt)
    pred_rt = mod_rt.predict(X_te)
    mae_rt = mean_absolute_error(y_te_rt, pred_rt)

    # Save results
    stamp = split_ts.strftime("%Y-%m-%d_%H-%M")
    os.makedirs("../data/baseline/power", exist_ok=True)
    os.makedirs("../data/baseline/runtime", exist_ok=True)

    power_file = f"../data/baseline/power/{stamp}.csv"
    runtime_file = f"../data/baseline/runtime/{stamp}.csv"

    pd.DataFrame(
        {
            "job_id": test_df["job_id"],
            "array_pos": test_df["array_pos"],
            "submit_time": test_df["submit_time"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "avg_power_per_node": y_te_pwr,
            "predicted_power": pred_pwr,
        }
    ).to_csv(power_file, index=False)

    pd.DataFrame(
        {
            "job_id": test_df["job_id"],
            "array_pos": test_df["array_pos"],
            "submit_time": test_df["submit_time"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "wallclock_used_sec": y_te_rt,
            "predicted_runtime": pred_rt,
        }
    ).to_csv(runtime_file, index=False)

    return split_ts, power_file, mae_pwr, runtime_file, mae_rt


def main():
    start_date, end_date = "2024-08-01", "2025-02-01"
    timestep_h = 1
    train_days, test_hours = 100, timestep_h

    split_times = pd.date_range(start=start_date, end=end_date, freq=f"{timestep_h}h")

    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=8, initializer=init_worker
    ) as ex:
        fut_map = {
            ex.submit(process_split_time, st, train_days, test_hours): st
            for st in split_times
        }
        for fut in concurrent.futures.as_completed(fut_map):
            st = fut_map[fut]
            try:
                res = fut.result()
                if res:
                    split_ts, f_pwr, mae_pwr, f_rt, mae_rt = res
                    print(
                        f"{split_ts}  "
                        f"MAE power={mae_pwr:8.2f}  saved→{os.path.basename(f_pwr)} | "
                        f"MAE runtime={mae_rt:8.2f}  saved→{os.path.basename(f_rt)}"
                    )
                    results.append(res)
            except Exception as exc:
                print(f"{st} raised {exc}")

    if results:
        p_mae = [r[2] for r in results]
        r_mae = [r[4] for r in results]
        print("\n=== Overall summary ===")
        print(f"Power   : mean MAE={np.mean(p_mae):.2f}  median={np.median(p_mae):.2f}")
        print(f"Runtime : mean MAE={np.mean(r_mae):.2f}  median={np.median(r_mae):.2f}")


if __name__ == "__main__":
    main()
