import argparse
import random
import sys
import os
import time

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from deel.puncc.api.RCPS import RCPS
from lambda_predictor import LambdaPredictorFromSoftmax
import wmloss

parser = argparse.ArgumentParser(description='RCPS experiment on ImageNet')
parser.add_argument('--data', default='imagenet_val_softmax_scores.npz', help='path to imagenet_val_softmax_scores.npz')
parser.add_argument('--n_runs', type=int, default=100, help='number of experiment runs')
parser.add_argument('--n_calib', type=int, default=30000, help='calibration set size')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--out_dir', default='results', help='directory for summary and plots')

ALPHA   = 0.1
DELTA   = 0.1
METHODS = ['clt', 'hoeffding_bentkus', 'wsr']

# Lambda grid: lam in [-1, 0], threshold = -lam in [0, 1]
# lam = -1 -> threshold 1 -> empty sets
# lam =  0 -> threshold 0 -> all classes
LAMBDA_GRID = tuple(np.linspace(-1.0, 0.0, 200))


if __name__ == "__main__":
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    wmloss.init(args.seed)

    # Load pre-computed softmax scores and labels
    data = np.load(args.data)
    all_scores = data['scores']   # (50000, 1000)
    all_labels = data['labels']   # (50000,)
    n_total = len(all_labels)
    n_val = n_total - args.n_calib

    print(f"Loaded {n_total} samples. Calibration: {args.n_calib}, Validation: {n_val}")
    print(f"Running {args.n_runs} experiments...\n")

    # Accumulators: results[method] = list of (mean_risk, lhat, set_sizes) per run
    results = {method: [] for method in METHODS}

    for run in range(args.n_runs):
        t_start = time.time()

        # Random 30k/20k split
        perm = np.random.permutation(n_total)
        calib_idx, val_idx = perm[:args.n_calib], perm[args.n_calib:]

        calib_scores = all_scores[calib_idx]
        calib_labels = all_labels[calib_idx]
        val_scores   = all_scores[val_idx]
        val_labels   = all_labels[val_idx]

        for method in METHODS:
            print(f"  Calibrating RCPS (method={method}, delta={DELTA})...")
            rcps = RCPS(
                model=LambdaPredictorFromSoftmax(),
                loss_function=wmloss.weighted_miscoverage_loss,
                loss_function_upper_bound=1,
            )
            rcps.calibrate(calib_scores, calib_labels, delta=DELTA, method=method)

            # Predict on validation set
            print(f"  Predicting with RCPS-selected lambda...")
            val_sets, lhat = rcps.predict(val_scores, alpha=ALPHA, lambda_grid=LAMBDA_GRID)

            # Metrics
            print(f"  Evaluating empirical risk and set sizes...")
            mean_risk = wmloss.weighted_miscoverage_loss(val_sets, val_labels).mean()
            set_sizes = np.array([len(S) for S in val_sets])

            results[method].append((mean_risk, lhat, set_sizes))

        elapsed = time.time() - t_start
        print(f"Run {run + 1}/{args.n_runs} completed in {elapsed:.1f}s")

    # --- Output directory ---
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Summary text file ---
    summary_path = os.path.join(args.out_dir, "summary.txt")
    summary_lines = [
        f"RCPS ImageNet experiment",
        f"  n_runs={args.n_runs}, n_calib={args.n_calib}, n_val={n_val}",
        f"  alpha={ALPHA}, delta={DELTA}",
        f"  seed={args.seed}",
        "",
    ]
    for method in METHODS:
        risks = np.array([r[0] for r in results[method]])
        lhats = np.array([r[1] for r in results[method]])
        sizes = np.concatenate([r[2] for r in results[method]])
        summary_lines += [
            f"{method.upper()}",
            f"  Mean risk:        {risks.mean():.4f} +/- {risks.std():.4f}  (target <= {ALPHA})",
            f"  Risk violations:  {(risks > ALPHA).sum()} / {args.n_runs}",
            f"  Mean lambda_hat:  {lhats.mean():.4f} +/- {lhats.std():.4f}",
            f"  Mean set size:    {sizes.mean():.2f} +/- {sizes.std():.2f}",
            "",
        ]

    summary_text = "\n".join(summary_lines)
    print("\n--- Summary ---\n" + summary_text)
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"Summary written to {summary_path}")

    # --- Histograms ---
    fig, axes = plt.subplots(2, len(METHODS), figsize=(5 * len(METHODS), 8))
    fig.suptitle(
        f"RCPS on ImageNet  —  {args.n_runs} runs, "
        f"n_calib={args.n_calib}, α={ALPHA}, δ={DELTA}",
        fontsize=13,
    )

    for col, method in enumerate(METHODS):
        risks = np.array([r[0] for r in results[method]])
        sizes = np.concatenate([r[2] for r in results[method]])

        # Row 0: histogram of mean empirical risks across runs
        ax_risk = axes[0, col]
        ax_risk.hist(risks, bins=20, edgecolor='black')
        ax_risk.axvline(ALPHA, color='red', linestyle='--', label=f'α={ALPHA}')
        ax_risk.set_title(f"{method.upper()}\nMean risk per run")
        ax_risk.set_xlabel("Empirical risk")
        ax_risk.set_ylabel("Count")
        ax_risk.legend()

        # Row 1: histogram of set sizes (all runs pooled)
        ax_size = axes[1, col]
        ax_size.hist(sizes, bins=50, edgecolor='black')
        ax_size.set_title(f"{method.upper()}\nSet sizes (all runs)")
        ax_size.set_xlabel("Prediction set size")
        ax_size.set_ylabel("Count")

    plt.tight_layout()
    plot_path = os.path.join(args.out_dir, "histograms.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Histograms saved to {plot_path}")
