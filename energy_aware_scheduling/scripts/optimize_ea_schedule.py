import os
import sys
import subprocess
import numpy as np
import pandas as pd
import optuna
from datetime import timedelta, datetime

import controller

def load_and_prepare_series(power_csv_path: str, re_csv_path: str):
    """
    Load two CSV files and prepare the timeseries.
    - power_csv_path: CSV file with columns "time" and "power_usage".
    - re_csv_path: CSV file with columns "time" and "re_availability".
    
    Filters the data to include only records between '2024-09-01' and '2024-09-16'.
    Then sets the time column as a DatetimeIndex and resamples to a 1-second resolution with interpolation.
    
    Returns:
      Tuple of two DataFrames: (df_power, df_re)
    """
    # Read the CSV files.
    df_power = pd.read_csv(power_csv_path)
    df_re = pd.read_csv(re_csv_path)
    
    # Define the desired date range.
    start_date = pd.Timestamp('2024-09-01')
    end_date = pd.Timestamp('2024-09-16')
    
    # Process power usage data.
    if not pd.api.types.is_numeric_dtype(df_power["time"]):
        df_power["time"] = pd.to_datetime(df_power["time"])
    else:
        # If already numeric, assume they are Unix seconds and convert to datetime.
        df_power["time"] = pd.to_datetime(df_power["time"], unit='s')
    
    # Filter the power data to the desired date range.
    df_power = df_power[(df_power["time"] >= start_date) & (df_power["time"] <= end_date)]
    
    # Process RE availability data.
    if not pd.api.types.is_numeric_dtype(df_re["time"]):
        df_re["time"] = pd.to_datetime(df_re["time"])
    else:
        df_re["time"] = pd.to_datetime(df_re["time"], unit='s')
    
    # Filter the RE data to the desired date range.
    df_re = df_re[(df_re["time"] >= start_date) & (df_re["time"] <= end_date)]
    
    # Set the "time" column as the index.
    df_power.set_index("time", inplace=True)
    df_re.set_index("time", inplace=True)
    
    # Resample both DataFrames to a 1-second resolution and interpolate missing values.
    df_power = df_power.resample("1s").mean().interpolate()
    df_re = df_re.resample("1s").mean().interpolate()
    
    return df_power, df_re

def compute_re_utilization_ratio(power_usage: pd.Series, re_availability: pd.Series) -> float:
    """
    Compute the Renewable Energy Utilization Ratio (REUR):
      REUR = (Sum over time of [power_usage * re_availability]) / (Total power_usage)
    """
    combined = pd.concat([power_usage, re_availability], axis=1, join='inner')
    combined.columns = ['power_usage', 're_availability']
    weighted_energy = (combined['power_usage'] * combined['re_availability']).sum()
    total_energy = combined['power_usage'].sum()
    return weighted_energy / total_energy if total_energy != 0 else np.nan

def get_avg_wait_time(results_fp):
    """
    Load the pickle file (results_fp) containing job-level data,
    adjust time fields as needed, and compute the average wait time (in hours).
    """
    # Code redacted due to proprietary information
    return wait_mean

# Baseline constants from your baseline scheduling runs.
RE_BASELINE = 0.4    # Example: baseline RE utilization ratio (0 to 1)
WT_BASELINE = 2.0    # Example: baseline average wait time in hours

def objective(trial: optuna.trial.Trial) -> float:
    """
    Optuna objective function. Suggests 'weight', 'gamma', 'blend_exponent',
    'time_boost_start', and 'time_boost_end' parameters, calls the simulator via
    the command line, and evaluates the resulting metrics.
    """
    # Suggest parameters to optimize.
    alpha = trial.suggest_float("alpha", 0, 6.0)
    beta = trial.suggest_float("beta", 0, 6.0)
    gamma = trial.suggest_float("gamma", 0, 1.0)
    time_boost_start = trial.suggest_float("time_boost_start", 4, 17.9)
    time_boost_end = trial.suggest_float("time_boost_end", time_boost_start + .1, 18)
    
    # Construct file paths for simulation output.
    results_pickle = (f"../optuna/trials/alpha_{alpha:.4f}_beta_{beta:.4f}_gamma_{gamma:.4f}_"
                      f"start_{time_boost_start:.4f}_end_{time_boost_end:.4f}.pkl")
    power_csv = (f"../optuna/trials/alpha_{alpha:.4f}_beta_{beta:.4f}_gamma_{gamma:.4f}_"
                 f"start_{time_boost_start:.4f}_end_{time_boost_end:.4f}.csv")
    
    # Construct the simulator command.
    cmd = [
        "python", "main.py",
        # Some fields redacted due to proprietary information
        "--power_alpha", str(alpha),
        "--power_beta", str(beta),
        "--power_gamma", str(gamma),
        "--power_tbs", str(time_boost_start),
        "--power_tbe", str(time_boost_end)
    ]
    
    # Run the simulator.
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        trial.set_user_attr("sim_error", result.stderr)
        raise RuntimeError(f"Simulation failed: {result.stderr}")
    
    # Load the simulation results.
    try:
        avg_wait = get_avg_wait_time(results_pickle)
    except Exception as e:
        trial.set_user_attr("wait_time_error", str(e))
        raise
    
    try:
        df_power, df_re = load_and_prepare_series(power_csv, "../data/re_availability.csv") # This RE data not actually provided
        re_util = compute_re_utilization_ratio(df_power["power_usage"], df_re["re_availability"])
    except Exception as e:
        trial.set_user_attr("re_util_error", str(e))
        raise
    
    
    # Define the objective: maximize RE utilization.
    objective_value = re_util
    
    trial.set_user_attr("avg_wait", avg_wait)
    trial.set_user_attr("re_util", re_util)
    
    print(f"Trial: weight={weight:.4f}, alpha={alpha:.4f}, beta={beta:.4f}, gamma={gamma:.4f}, "
          f"time_boost_start={time_boost_start:.4f}, time_boost_end={time_boost_end:.4f}, "
          f"avg_wait={avg_wait:.4f}h, re_util={re_util:.4f}, objective={objective_value:.4f}")
    
    return objective_value

if __name__ == "__main__":
    # Use SQLite storage for persistent study results.
    os.makedirs("../data/optuna", exist_ok=True)
    storage_url = "sqlite:///../optuna/optuna_study.db"
    study_name = "energy_aware_sim"
    
    study = optuna.create_study(
        direction="maximize",
        storage=storage_url,
        study_name=study_name,
        load_if_exists=True
    )
    
    # Run the optimization loop.
    study.optimize(objective, n_trials=100, n_jobs=8)
    
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  alpha: {best_trial.params['alpha']}")
    print(f"  beta: {best_trial.params['beta']}")
    print(f"  gamma: {best_trial.params['gamma']}")
    print(f"  time_boost_start: {best_trial.params['time_boost_start']}")
    print(f"  time_boost_end: {best_trial.params['time_boost_end']}")
    print(f"  Objective: {best_trial.value}")