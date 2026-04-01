"""Extract and summarize results from CRL experiment logs."""
import re
import os
import numpy as np

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")

def parse_log(filepath):
    """Parse evaluation metrics from a log file."""
    results = {}
    with open(filepath) as f:
        for line in f:
            # Parse eval metrics
            m = re.search(r'step: (\d+), (eval/\S+): ([\d.]+(?:e[+-]?\d+)?)', line)
            if m:
                step = int(m.group(1))
                metric = m.group(2)
                value = float(m.group(3))
                if metric not in results:
                    results[metric] = []
                results[metric].append((step, value))
            # Parse training metrics
            m = re.search(r'step: (\d+), (training/\S+): ([\d.]+(?:e[+-]?\d+)?)', line)
            if m:
                step = int(m.group(1))
                metric = m.group(2)
                value = float(m.group(3))
                if metric not in results:
                    results[metric] = []
                results[metric].append((step, value))
    return results

def summarize_env(env_name, seeds=(1, 2, 3)):
    """Summarize results across seeds for an environment."""
    print(f"\n{'='*60}")
    print(f" {env_name.upper()} - CRL (bwd_infonce, norm)")
    print(f"{'='*60}")

    all_final = {}
    for seed in seeds:
        logfile = os.path.join(LOG_DIR, f"{env_name}_s{seed}.log")
        if not os.path.exists(logfile) or os.path.getsize(logfile) < 1000:
            print(f"  Seed {seed}: log missing or too small")
            continue

        results = parse_log(logfile)
        print(f"\n  Seed {seed}:")

        for metric in ["eval/episode_success_any", "eval/episode_success",
                        "eval/episode_success_easy", "eval/episode_dist",
                        "training/sps", "training/critic_loss"]:
            if metric in results and results[metric]:
                steps, values = zip(*results[metric])
                final_val = values[-1]
                max_val = max(values)
                if metric not in all_final:
                    all_final[metric] = []
                all_final[metric].append(final_val)
                print(f"    {metric}: final={final_val:.4f}, max={max_val:.4f} (at step {steps[values.index(max_val)]})")

    if all_final:
        print(f"\n  Summary across seeds:")
        for metric, vals in all_final.items():
            vals = np.array(vals)
            print(f"    {metric}: mean={vals.mean():.4f} +/- {vals.std():.4f}")

if __name__ == "__main__":
    for env in ["ant", "arm_reach", "arm_push_easy"]:
        summarize_env(env)
