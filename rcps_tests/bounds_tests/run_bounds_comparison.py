"""
Reproduces a Figure-4-style numerical comparison of the RCPS paper's
upper confidence bounds (Bates, Angelopoulos, Lei, Malik & Jordan 2021,
Section 3.1.4): for synthetic Bernoulli/Beta-distributed losses at a grid
of sample sizes n and means mu, we measure each method's empirical
coverage P(UCB >= mu) and median gap (UCB - mu).

This is a corrected, standalone extraction of bounds_tests.ipynb, fixing
two bugs found while extracting it:
  - `plt` (matplotlib.pyplot) was used in the plotting cell but never
    imported.
  - The per-row `ucbs` dict computed "tight-Hoef" and "Bentkus" entries
    that are not in METHODS (which only tracks the paper's five Figure-4
    methods: Bernstein, CLT, HB, sim-Hoef, WSR); `covered[method] += ...`
    would raise a KeyError on those two extra, untracked keys the moment
    the cell was actually run.

The tail-probability-based methods (HB) and WSR search over a grid of
candidate risk values R for each row; the wrapper functions
(hoeffding_bentkus_ucb, tighter_hoeffding_ucb, bentkus_ucb) hard-code a
1000-point default grid with no way to override it, which makes reps in
the tens of thousands prohibitively slow. This script instead calls
tpb_to_ucb directly with a caller-configurable (smaller) R_grid, using
the same underlying tail-probability functions, so reps can be scaled up
while keeping runtime reasonable. --reps 1_000_000 with the default
R_grid would reproduce the paper's exact scale but is not the default
here for that reason.
"""
import argparse
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deel.puncc.api.RCPS import (
    bentkus_tpb,
    bernstein_ucb,
    clt_ucb,
    simplified_hoeffding_ucb,
    tighter_hoeffding_tpb,
    tpb_to_ucb,
    wsr_ucb,
)

DIST_SPECS = [
    ("Ber(mu)", None),
    ("Beta(0.1, .)", 0.1),
    ("Beta(1, . )", 1.0),
    ("Beta(10, .)", 10.0),
]

MUS = [0.1, 0.01, 0.001]

NS = [int(math.floor(10**r)) for r in [2, 2.5, 3, 3.5, 4]]

METHODS = ["Bernstein", "CLT", "HB", "sim-Hoef", "WSR"]


def draw_samples(rng, beta_a, mu, rows, n):
    """Draw loss samples with mean mu."""
    if beta_a is None:
        return rng.binomial(1, mu, size=(rows, n)).astype(float)

    beta_b = beta_a * (1.0 / mu - 1.0)
    return rng.beta(beta_a, beta_b, size=(rows, n))


def effective_batch_size(n, requested_batch, max_elements=4_000_000):
    """Keep memory moderate for large n."""
    return max(1, min(requested_batch, max_elements // n))


def hb_ucb(risk, delta, n, r_grid):
    tpb = lambda t, R: min(tighter_hoeffding_tpb(t, R, n), bentkus_tpb(t, R, n))
    return tpb_to_ucb(tpb, risk, delta, R_grid=r_grid)


def simulate_one_setting(rng, dist_label, beta_a, mu, n, delta, reps, batch, r_grid):
    gaps = {method: np.empty(reps, dtype=float) for method in METHODS}
    covered = {method: 0 for method in METHODS}

    pos = 0
    batch = effective_batch_size(n, batch)

    while pos < reps:
        rows = min(batch, reps - pos)

        losses = draw_samples(rng, beta_a, mu, rows, n)
        emp_risks = losses.mean(axis=1)
        risk_sd = losses.std(axis=1, ddof=1)

        ucbs = {
            "sim-Hoef": np.array(
                [
                    simplified_hoeffding_ucb(lambda lam, r=r: r, delta, n)(0.0)
                    for r in emp_risks
                ]
            ),
            "HB": np.array(
                [hb_ucb(lambda lam, r=r: r, delta, n, r_grid)(0.0) for r in emp_risks]
            ),
            "Bernstein": np.array(
                [
                    bernstein_ucb(
                        lambda lam, r=r: r, lambda lam, s=s: s, delta, n
                    )(0.0)
                    for r, s in zip(emp_risks, risk_sd)
                ]
            ),
            "CLT": np.array(
                [
                    clt_ucb(lambda lam, r=r: r, lambda lam, s=s: s, delta, n)(0.0)
                    for r, s in zip(emp_risks, risk_sd)
                ]
            ),
            "WSR": np.array(
                [
                    wsr_ucb([lambda lam, l=l: l for l in row], delta, R_grid=r_grid)(
                        0.0
                    )
                    for row in losses
                ]
            ),
        }

        for method, ucb in ucbs.items():
            covered[method] += int(np.sum(ucb >= mu))
            gaps[method][pos : pos + rows] = ucb - mu

        pos += rows

    records = []
    for method in METHODS:
        records.append(
            {
                "dist": dist_label,
                "mu": mu,
                "n": n,
                "method": method,
                "coverage": covered[method] / reps,
                "median_gap": float(np.median(gaps[method])),
            }
        )

    return records


def run_simulation(delta, reps, batch, seed, r_grid, ns=NS):
    rng = np.random.default_rng(seed)
    all_records = []

    total = len(DIST_SPECS) * len(MUS) * len(ns)
    done = 0
    t_start = time.time()

    for dist_label, beta_a in DIST_SPECS:
        for mu in MUS:
            for n in ns:
                done += 1
                print(
                    f"[{done:02d}/{total}] dist={dist_label:12s} "
                    f"mu={mu:g} n={n} reps={reps} "
                    f"({time.time() - t_start:.1f}s elapsed)"
                )
                all_records.extend(
                    simulate_one_setting(
                        rng=rng,
                        dist_label=dist_label,
                        beta_a=beta_a,
                        mu=mu,
                        n=n,
                        delta=delta,
                        reps=reps,
                        batch=batch,
                        r_grid=r_grid,
                    )
                )

    return pd.DataFrame(all_records)


def plot_panel_grid(df, y_col, ylabel, outpath, delta, ns, include_clt=True):
    methods = ["Bernstein", "CLT", "HB", "sim-Hoef", "WSR"]
    if not include_clt:
        methods = ["Bernstein", "HB", "sim-Hoef", "WSR"]

    colors = {
        "Bernstein": "black",
        "CLT": "teal",
        "HB": "blue",
        "sim-Hoef": "goldenrod",
        "WSR": "red",
    }
    linestyles = {
        "Bernstein": (0, (1, 3)),
        "CLT": (0, (3, 2, 1, 2)),
        "HB": (0, (5, 3)),
        "sim-Hoef": (0, (3, 1, 1, 1)),
        "WSR": "-",
    }

    fig, axes = plt.subplots(
        nrows=len(DIST_SPECS),
        ncols=len(MUS),
        figsize=(10.5, 8.0),
        sharex=True,
    )

    for r, (dist_label, _) in enumerate(DIST_SPECS):
        for c, mu in enumerate(MUS):
            ax = axes[r, c]

            for method in methods:
                sub = df[
                    (df["dist"] == dist_label)
                    & (df["mu"] == mu)
                    & (df["method"] == method)
                ].sort_values("n")

                ax.plot(
                    sub["n"],
                    sub[y_col],
                    label=method,
                    color=colors[method],
                    linestyle=linestyles[method],
                    linewidth=1.7,
                )

            ax.set_xscale("log", base=10)
            ax.set_xlim(min(ns) * 0.9, max(ns) * 1.1)

            if y_col == "coverage":
                ax.axhline(1.0 - delta, color="black", linewidth=0.8)
            else:
                ax.set_yscale("log", base=10)

            if r == 0:
                ax.set_title(f"mu = {mu:g}")

            if c == len(MUS) - 1:
                ax.text(
                    1.04,
                    0.5,
                    dist_label,
                    transform=ax.transAxes,
                    rotation=-90,
                    va="center",
                    ha="left",
                )

            if r == len(DIST_SPECS) - 1:
                ax.set_xlabel("n")

            if c == 0:
                ax.set_ylabel(ylabel)

            ax.grid(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(methods), frameon=False)

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Figure-4-style UCB bound comparison for RCPS"
    )
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument(
        "--reps",
        type=int,
        default=3000,
        help="Repetitions per (dist, mu, n) setting. The paper uses "
        "1_000_000; that scale is impractical here since HB/WSR search "
        "over an R grid per repetition rather than being vectorized "
        "across n -- see module docstring.",
    )
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--r_grid_size",
        type=int,
        default=40,
        help="Number of candidate R values searched for HB and WSR. "
        "Smaller is faster but coarser; the wrapper functions' own "
        "default (1000) is not used here for speed.",
    )
    parser.add_argument(
        "--ns",
        type=int,
        nargs="+",
        default=NS,
        help="Sample sizes to sweep (default matches the paper's grid).",
    )
    parser.add_argument("--out_dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).parent / "fig4_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    r_grid = np.linspace(0.0, 1.0, args.r_grid_size)

    df = run_simulation(
        delta=args.delta,
        reps=args.reps,
        batch=args.batch,
        seed=args.seed,
        r_grid=r_grid,
        ns=args.ns,
    )

    csv_path = out_dir / "figure4_simulation_summary.csv"
    df.to_csv(csv_path, index=False)

    coverage_path = out_dir / "fig4_coverage.png"
    gap_path = out_dir / "fig4_gap.png"

    plot_panel_grid(
        df=df,
        y_col="coverage",
        ylabel="Coverage",
        outpath=coverage_path,
        delta=args.delta,
        ns=args.ns,
        include_clt=True,
    )
    plot_panel_grid(
        df=df,
        y_col="median_gap",
        ylabel=r"$\hat{R}^{+} - R$",
        outpath=gap_path,
        delta=args.delta,
        ns=args.ns,
        include_clt=False,
    )

    print(f"Saved CSV to: {csv_path}")
    print(f"Saved coverage plot to: {coverage_path}")
    print(f"Saved gap plot to: {gap_path}")
