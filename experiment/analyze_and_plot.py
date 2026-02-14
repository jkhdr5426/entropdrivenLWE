import pandas as pd
import matplotlib.pyplot as plt
import glob, os
import numpy as np

def summarize_results(results_dir):
    files = glob.glob(os.path.join(results_dir, 'summary_*.csv'))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    if not dfs:
        print("No result files found in", results_dir)
        return
    all_df = pd.concat(dfs, ignore_index=True)
    print(all_df)
    # Simple plots: honest_dsr vs n
    for transform in all_df['transform'].unique():
        df = all_df[all_df['transform']==transform]
        plt.plot(df['n'], df['honest_dsr'], marker='o', label=f"{transform} DSR")
    plt.xlabel('n')
    plt.ylabel('Honest DSR')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir,'dsr_plot.png'))
    plt.clf()
    # attacker success
    for transform in all_df['transform'].unique():
        df = all_df[all_df['transform']==transform]
        plt.plot(df['n'], df['attacker_success_regression'], marker='o', label=f"{transform} attacker")
    plt.xlabel('n')
    plt.ylabel('Attacker regression success')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir,'attacker_plot.png'))
    print("Plots written to", results_dir)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='results')
    args = parser.parse_args()
    summarize_results(args.results_dir)
