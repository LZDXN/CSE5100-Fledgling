#!/usr/bin/env python
# Render the aggregated robustness / observability JSON files produced by
# the two runner scripts into figures suitable for the report.

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def _bar(ax, labels, means, stds, colors=None):
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")


def plotTwoByTwo(twoByTwo: dict, outPath: str):
    # 2x2 grouped bar of mean reward across the four train/eval cells.
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    cells = ["evalClean", "evalNoisy"]
    trains = ["train_clean", "train_noisy"]
    width = 0.35
    x = np.arange(len(cells))
    for i, tc in enumerate(trains):
        means = [
            twoByTwo[tc][ev]["meanReward"]["meanAcrossSeeds"] for ev in cells
        ]
        stds = [twoByTwo[tc][ev]["meanReward"]["stdAcrossSeeds"] for ev in cells]
        ax[0].bar(
            x + (i - 0.5) * width, means, width, yerr=stds, capsize=4, label=tc
        )
        # Tracking error subplot
        means_te = [
            twoByTwo[tc][ev]["avgTrackingErr"]["meanAcrossSeeds"] for ev in cells
        ]
        stds_te = [
            twoByTwo[tc][ev]["avgTrackingErr"]["stdAcrossSeeds"] for ev in cells
        ]
        ax[1].bar(
            x + (i - 0.5) * width, means_te, width, yerr=stds_te, capsize=4, label=tc
        )
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(["Eval Clean", "Eval Noisy"])
    ax[0].set_ylabel("Mean eval reward")
    ax[0].set_title("2x2 train/eval ablation -- reward")
    ax[0].legend()
    ax[0].grid(axis="y", alpha=0.3)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(["Eval Clean", "Eval Noisy"])
    ax[1].set_ylabel("Mean tracking error")
    ax[1].set_title("2x2 train/eval ablation -- tracking error")
    ax[1].legend()
    ax[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outPath, dpi=150)
    plt.close(fig)


def plotFailure(failure: dict, outPath: str):
    # Grouped bar across {n_rotors} for each train/eval combination.
    nRotorsList = sorted(int(k.split("_")[-1]) for k in failure.keys())
    trainConds = ["train_clean", "train_failure"]
    evalConds = ["evalNoFailure", "evalFailure"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    width = 0.18
    x = np.arange(len(nRotorsList))

    # Reward subplot
    for i, tc in enumerate(trainConds):
        for j, ec in enumerate(evalConds):
            means = []
            stds = []
            for n in nRotorsList:
                cell = failure[f"n_rotors_{n}"][tc][ec]
                means.append(cell["meanReward"]["meanAcrossSeeds"])
                stds.append(cell["meanReward"]["stdAcrossSeeds"])
            offset = ((i * 2 + j) - 1.5) * width
            label = f"{tc.split('_')[-1]}/{ec.replace('eval','').lower()}"
            axes[0].bar(x + offset, means, width, yerr=stds, capsize=3, label=label)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"n={n}" for n in nRotorsList])
    axes[0].set_ylabel("Mean eval reward")
    axes[0].set_title("Rotor failure -- reward by config")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    # Tracking error subplot
    for i, tc in enumerate(trainConds):
        for j, ec in enumerate(evalConds):
            means = []
            stds = []
            for n in nRotorsList:
                cell = failure[f"n_rotors_{n}"][tc][ec]
                means.append(cell["avgTrackingErr"]["meanAcrossSeeds"])
                stds.append(cell["avgTrackingErr"]["stdAcrossSeeds"])
            offset = ((i * 2 + j) - 1.5) * width
            label = f"{tc.split('_')[-1]}/{ec.replace('eval','').lower()}"
            axes[1].bar(x + offset, means, width, yerr=stds, capsize=3, label=label)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"n={n}" for n in nRotorsList])
    axes[1].set_ylabel("Mean tracking error")
    axes[1].set_title("Rotor failure -- tracking error by config")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(outPath, dpi=150)
    plt.close(fig)


def plotObservability(regimes: dict, outPath: str):
    names = list(regimes.keys())
    means = [regimes[n]["meanReward"]["meanAcrossSeeds"] for n in names]
    stds = [regimes[n]["meanReward"]["stdAcrossSeeds"] for n in names]
    teMeans = [regimes[n]["avgTrackingErr"]["meanAcrossSeeds"] for n in names]
    teStds = [regimes[n]["avgTrackingErr"]["stdAcrossSeeds"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    _bar(axes[0], names, means, stds)
    axes[0].set_ylabel("Mean eval reward")
    axes[0].set_title("Partial observability -- reward by regime")
    axes[0].grid(axis="y", alpha=0.3)
    _bar(axes[1], names, teMeans, teStds)
    axes[1].set_ylabel("Mean tracking error")
    axes[1].set_title("Partial observability -- tracking error by regime")
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outPath, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robustness_json", default=None)
    p.add_argument("--observability_json", default=None)
    p.add_argument("--out_dir", default="data/aux_plots")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.robustness_json and os.path.exists(args.robustness_json):
        with open(args.robustness_json) as f:
            r = json.load(f)
        if "twoByTwo" in r:
            plotTwoByTwo(r["twoByTwo"], os.path.join(args.out_dir, "robustness_2x2.png"))
        if "failure" in r:
            plotFailure(r["failure"], os.path.join(args.out_dir, "robustness_failure.png"))

    if args.observability_json and os.path.exists(args.observability_json):
        with open(args.observability_json) as f:
            o = json.load(f)
        plotObservability(
            o["regimes"], os.path.join(args.out_dir, "observability_regimes.png")
        )

    print(f"Wrote plots to {args.out_dir}")


if __name__ == "__main__":
    main()
