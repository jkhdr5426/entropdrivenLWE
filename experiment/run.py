#!/usr/bin/env python3
"""
run.py — automated experiment runner (modified per request)

Features:
 - Runs run_grid_experiment.py over a grid of transforms, n, sigma, repeats
 - Aggregates summary CSVs recursively
 - Aggregates sample CSVs (raw per-trial data) recursively into one raw CSV
 - Computes descriptive stats (mean/std) grouped by transform,n,sigma
 - Performs pairwise t-tests on attacker success between transforms
 - Produces the requested plots:
     * Honest vs Attacker per Transform (grouped bar)
     * Effect of Noise on Honest Success (line)
     * Time per Decapsulation (box plot by transform)
     * Linearization Dimension vs Attacker Success (scatter)
 - Saves aggregated CSVs, grouped stats, t-tests, combined raw samples, and plots
 - Place this script alongside run_grid_experiment.py and run it:
     python run.py
Requirements:
  pandas, numpy, matplotlib, scipy
"""
import os
import subprocess
import glob
import argparse
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy import stats

# --------- Configuration (adjust as needed) ----------
DEFAULT_TRANSFORMS = ["quadratic", "cubic", "eo_lwe"]
DEFAULT_N = [16, 32, 64]
DEFAULT_Q = 3329
DEFAULT_SIGMA = [1.0, 2.0, 3.0]   # multiple sigma values for noise sweep
DEFAULT_TRIALS = 200
DEFAULT_ALPHA_RANGE = 8
DEFAULT_RUNS_PER_SETTING = 3     # repeats per cell to estimate variance
RESULTS_ROOT = "results_example2"
PLOTS_DIR = os.path.join(RESULTS_ROOT, "plots")
CSV_AGG = os.path.join(RESULTS_ROOT, "aggregated_results.csv")
SAMPLES_RAW_CSV = os.path.join(RESULTS_ROOT, "all_samples_raw.csv")
STATS_CSV = os.path.join(RESULTS_ROOT, "pairwise_ttests.csv")
GROUPED_STATS_CSV = os.path.join(RESULTS_ROOT, "grouped_stats.csv")

# --------- Helper functions ----------
def ensure_dirs():
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

def run_single(exp, out_dir):
    cmd = [
        "python", "run_grid_experiment.py",
        "--transform", exp["transform"],
        "--n", str(exp["n"]),
        "--q", str(exp["q"]),
        "--sigma", str(exp["sigma"]),
        "--trials", str(exp["trials"]),
        "--alpha_range", str(exp["alpha_range"]),
        "--out_dir", out_dir
    ]
    subprocess.run(cmd, check=True)

def collect_summary_csvs(root):
    pattern = os.path.join(root, "**", "summary_*.csv")
    files = sorted(glob.glob(pattern, recursive=True))
    return files

def collect_sample_csvs(root):
    pattern = os.path.join(root, "**", "samples_*.csv")
    files = sorted(glob.glob(pattern, recursive=True))
    return files

def load_csv_safe(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def aggregate_summaries(summary_files):
    dfs = []
    for f in summary_files:
        df = load_csv_safe(f)
        if df is None:
            continue
        folder = os.path.basename(os.path.dirname(f))
        df = df.copy()
        df["source_file"] = f
        df["source_folder"] = folder
        # try to infer transform,n,sigma from filename or folder
        # leave as-is; run_grid_experiment writes these fields in the CSV
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def aggregate_samples(sample_files):
    # sample CSVs expected columns: ['trial','a_hex','b','m','alpha','e']
    rows = []
    for f in sample_files:
        df = load_csv_safe(f)
        if df is None:
            continue
        folder = os.path.basename(os.path.dirname(f))
        # try to attach folder metadata (which encodes transform_n...)
        df = df.copy()
        df["source_file"] = f
        df["source_folder"] = folder
        # ensure columns exist
        for col in ["trial","a_hex","b","m","alpha","e"]:
            if col not in df.columns:
                df[col] = np.nan
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

def compute_group_stats(df, group_cols=["transform","n","sigma"]):
    # Defensive: ensure columns exist
    expected = ["honest_dsr","attacker_success_regression","bkw_score","linearization_dim","time_s"]
    for c in expected:
        if c not in df.columns:
            df[c] = np.nan
    agg = df.groupby(group_cols).agg(
        honest_mean = ("honest_dsr","mean"),
        honest_std  = ("honest_dsr","std"),
        attacker_mean = ("attacker_success_regression","mean"),
        attacker_std  = ("attacker_success_regression","std"),
        bkw_mean = ("bkw_score","mean"),
        bkw_std = ("bkw_score","std"),
        lin_dim = ("linearization_dim","mean"),
        time_mean = ("time_s","mean"),
        time_std = ("time_s","std"),
        runs = ("honest_dsr","count")
    ).reset_index()
    return agg

def pairwise_ttests(df, metric="attacker_success_regression"):
    rows = []
    # ensure numeric
    df = df.copy()
    if metric not in df.columns:
        return pd.DataFrame()
    groups = df.groupby(["n","sigma"])
    for (n,sigma), sub in groups:
        transforms = sub["transform"].unique()
        for a,b in combinations(transforms,2):
            vals_a = sub[sub["transform"]==a][metric].dropna().values
            vals_b = sub[sub["transform"]==b][metric].dropna().values
            if len(vals_a) < 2 or len(vals_b) < 2:
                tstat = np.nan
                pval = np.nan
            else:
                tstat, pval = stats.ttest_ind(vals_a, vals_b, equal_var=False)
            rows.append({
                "n": n,
                "sigma": sigma,
                "metric": metric,
                "transform_a": a,
                "transform_b": b,
                "t_stat": (float(tstat) if not np.isnan(tstat) else None),
                "p_value": (float(pval) if not np.isnan(pval) else None),
                "mean_a": (float(np.mean(vals_a)) if vals_a.size>0 else None),
                "mean_b": (float(np.mean(vals_b)) if vals_b.size>0 else None),
                "count_a": int(len(vals_a)),
                "count_b": int(len(vals_b))
            })
    return pd.DataFrame(rows)

# plotting helpers
def plot_honest_vs_attacker_bar(agg_df):
    # aggregated across n and sigma: group by transform
    df = agg_df.groupby("transform").agg(
        honest_mean = ("honest_mean","mean"),
        honest_std = ("honest_mean","std"),
        attacker_mean = ("attacker_mean","mean"),
        attacker_std = ("attacker_mean","std")
    ).reset_index()
    if df.empty:
        return
    x = np.arange(len(df))
    width = 0.35
    plt.figure(figsize=(8,6))
    plt.bar(x - width/2, df["honest_mean"], width, yerr=df["honest_std"], capsize=4, label="Honest DSR")
    plt.bar(x + width/2, df["attacker_mean"], width, yerr=df["attacker_std"], capsize=4, label="Attacker Success")
    plt.xticks(x, df["transform"])
    plt.ylabel("Success rate")
    plt.ylim(0,1)
    plt.title("Honest vs Attacker per Transform (aggregated)")
    plt.legend()
    out = os.path.join(PLOTS_DIR, "honest_vs_attacker_bar.png")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def plot_noise_effect_line(stats_df):
    # stats_df must contain columns transform,n,sigma,honest_mean
    if stats_df.empty:
        return
    plt.figure(figsize=(8,6))
    for t in sorted(stats_df['transform'].unique()):
        sub = stats_df[stats_df['transform']==t].sort_values('sigma')
        plt.plot(sub['sigma'], sub['honest_mean'], marker='o', label=t)
    plt.xlabel("Sigma (noise)")
    plt.ylabel("Honest DSR")
    plt.title("Effect of Noise on Honest Success")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    out = os.path.join(PLOTS_DIR, "noise_vs_honest_line.png")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def plot_time_boxplot(agg_df):
    # Use time_mean aggregated per run; but build a list of times per transform across runs
    if agg_df.empty:
        return
    data = []
    labels = []
    for t in sorted(agg_df['transform'].unique()):
        vals = agg_df[agg_df['transform']==t]['time_mean'].dropna().values
        if vals.size > 0:
            data.append(vals)
            labels.append(t)
    if not data:
        return
    plt.figure(figsize=(8,6))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("Time (s)")
    plt.title("Time per Decapsulation by Transform (across runs)")
    out = os.path.join(PLOTS_DIR, "decapsulation_time_boxplot.png")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def plot_lin_dim_vs_attacker(agg_df):
    if agg_df.empty:
        return
    plt.figure(figsize=(8,6))
    for t in sorted(agg_df['transform'].unique()):
        sub = agg_df[agg_df['transform']==t]
        plt.scatter(sub['lin_dim'], sub['attacker_mean'], label=t)
    plt.xlabel("Linearization Dimension")
    plt.ylabel("Attacker Success Rate")
    plt.title("Linearization Dimension vs Attacker Success")
    plt.legend()
    out = os.path.join(PLOTS_DIR, "linearization_vs_attacker.png")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

# --------- Main execution ----------
def main(args):
    ensure_dirs()

    # Build experiment cells
    experiments = []
    for transform in args.transforms:
        for n in args.n_values:
            for sigma in args.sigma_values:
                experiments.append({
                    "transform": transform,
                    "n": n,
                    "q": args.q,
                    "sigma": sigma,
                    "trials": args.trials,
                    "alpha_range": args.alpha_range
                })

    print(f"Total experiment cells: {len(experiments)}; repeats per cell: {args.runs_per_setting}")

    # Run experiments repeated times
    for exp in experiments:
        for run_i in range(args.runs_per_setting):
            out_dir = os.path.join(RESULTS_ROOT, f"{exp['transform']}_n{exp['n']}_s{str(exp['sigma']).replace('.','p')}_run{run_i+1}")
            os.makedirs(out_dir, exist_ok=True)
            print(f"\nRunning: transform={exp['transform']} n={exp['n']} sigma={exp['sigma']} run={run_i+1}")
            run_single(exp, out_dir)
            time.sleep(0.05)  # small pause

    # Collect all summary CSVs
    summary_files = collect_summary_csvs(RESULTS_ROOT)
    if not summary_files:
        print("No summary CSVs found. Exiting.")
        return

    print(f"\nFound {len(summary_files)} summary CSV files. Aggregating...")
    agg_df = aggregate_summaries(summary_files)
    if agg_df.empty:
        print("Aggregation produced empty DataFrame. Exiting.")
        return

    # Normalize types
    for col in ["n","q","trials"]:
        if col in agg_df.columns:
            agg_df[col] = pd.to_numeric(agg_df[col], errors='coerce').astype(pd.Int64Dtype())
    for col in ["sigma","honest_dsr","attacker_success_regression","bkw_score","linearization_dim","time_s"]:
        if col in agg_df.columns:
            agg_df[col] = pd.to_numeric(agg_df[col], errors='coerce')

    # Save aggregated summary CSV (raw)
    agg_df.to_csv(CSV_AGG, index=False)
    print(f"Aggregated summary CSV written to {CSV_AGG}")

    # Collect & combine raw sample CSVs
    sample_files = collect_sample_csvs(RESULTS_ROOT)
    combined_samples_df = aggregate_samples(sample_files)
    if not combined_samples_df.empty:
        combined_samples_df.to_csv(SAMPLES_RAW_CSV, index=False)
        print(f"Combined raw samples CSV written to {SAMPLES_RAW_CSV}")
    else:
        print("No sample CSVs found to combine.")

    # Compute grouped descriptive stats
    # We expect agg_df to have columns: transform,n,sigma,...
    # Sometimes transform/n/sigma are inside the CSV rows already; ensure they exist
    if 'transform' not in agg_df.columns:
        # try to parse from source_folder (best-effort)
        agg_df['transform'] = agg_df['source_folder'].str.split('_').str[0]
    if 'n' not in agg_df.columns:
        # try to parse n from folder name
        agg_df['n'] = agg_df['source_folder'].str.extract(r'_n(\d+)').astype(float)
    if 'sigma' not in agg_df.columns:
        agg_df['sigma'] = agg_df['source_folder'].str.extract(r'_s([0-9p]+)')[0]
        # convert '2p0' back to 2.0
        agg_df['sigma'] = agg_df['sigma'].str.replace('p','.', regex=False).astype(float)

    # Compute group stats
    stats_df = compute_group_stats(agg_df, group_cols=["transform","n","sigma"])
    stats_df.to_csv(GROUPED_STATS_CSV, index=False)
    print(f"Grouped stats written to {GROUPED_STATS_CSV}")

    # Pairwise t-tests (attacker success)
    ttests_df = pairwise_ttests(agg_df, metric="attacker_success_regression")
    ttests_df.to_csv(STATS_CSV, index=False)
    print(f"Pairwise t-tests written to {STATS_CSV}")

    # Plotting
    plot_honest_vs_attacker_bar(stats_df)
    plot_noise_effect_line(stats_df)
    plot_time_boxplot(stats_df)
    plot_lin_dim_vs_attacker(stats_df)

    print("\nPlots saved to", PLOTS_DIR)
    print("Done.")

# --------- Argument parsing ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full experiment grid, aggregate results, compute stats, and plot.")
    parser.add_argument("--transforms", nargs="+", default=DEFAULT_TRANSFORMS,
                        help="List of transforms to test (space separated).")
    parser.add_argument("--n_values", nargs="+", type=int, default=DEFAULT_N,
                        help="List of n (dimensions) to sweep.")
    parser.add_argument("--q", type=int, default=DEFAULT_Q, help="Modulus q.")
    parser.add_argument("--sigma_values", nargs="+", type=float, default=DEFAULT_SIGMA,
                        help="List of sigma (noise) values to sweep.")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Trials per run.")
    parser.add_argument("--alpha_range", type=int, default=DEFAULT_ALPHA_RANGE, help="Alpha range used in transforms.")
    parser.add_argument("--runs_per_setting", type=int, default=DEFAULT_RUNS_PER_SETTING,
                        help="Number of repeated runs for each experiment cell (to compute variance).")
    args = parser.parse_args()

    # adapt parsed args into expected names
    args.runs_per_setting = args.runs_per_setting
    args.sigma_values = args.sigma_values
    args.n_values = args.n_values
    args.transforms = args.transforms

    main(args)
