#!/usr/bin/env python
# Section 5.2 -- Cross-configuration generalization.
#
# Trains the per-axis PPO controller across the Cartesian product of:
#   - rotor counts:  {4, 6, 8}
#   - axes:          {PITCHLON, ROLLLAT, VERT, YAWHDG}
# with `--seeds` independent runs per cell.  Identical hyperparameters are
# used across the configurations so the only architectural change is the
# actor's output dimension scaling with rotor count -- this is the structural
# property the report claims is sufficient for cross-config transfer.
#
# Aggregated per-cell metrics (mean +/- std across seeds) are written to
# crossconfig_results.json plus a Markdown summary table.
#
# Usage:
#   python scripts/runCrossConfigExperiments.py \
#       --out_dir data/aux_crossconfig \
#       --seeds 3 --n_batches 60

import argparse
import json
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _THIS_DIR)

import multiRotorPlant  # noqa: E402
import auxEval  # noqa: E402
from _runnerCommon import runMainSubprocess, aggregateOverSeeds  # noqa: E402


def runCell(args, nRotors: int, axisName: str) -> dict:
    perSeed = []
    plant = multiRotorPlant.multiRotor6DOFWithXYZPositionError_class(
        rotorCount_nr_int=nRotors
    )
    for seed in args.seedList:
        local = argparse.Namespace(**vars(args))
        local.seed = seed
        local.n_rotors = nRotors
        local.axis = axisName
        runDir = runMainSubprocess(
            _REPO_ROOT,
            local,
            f"crosscfg_n{nRotors}_{axisName.lower()}_s{seed}",
            extra=[],
        )
        metrics = auxEval.evaluatePolicy(
            runDir,
            plant,
            axisName=axisName,
            rCmd=1.0,
            dt=0.01,
            maxSteps=args.max_steps,
            hiddenDim=64,
            obsCfgEval=None,
            distCfgEval=None,
            nEpisodes=args.eval_episodes,
            seed=12345,
        )
        perSeed.append(metrics)
    return aggregateOverSeeds(perSeed)


def writeMarkdownSummary(out: dict, path: str):
    lines = ["# Section 5.2 cross-configuration aggregates", ""]
    lines.append("| n_rotors | axis | meanReward | trackErr | settlingTime | overshoot% |")
    lines.append("|---------|------|------------|----------|--------------|------------|")
    for nKey, axisBlock in out["cells"].items():
        n = nKey.split("_")[-1]
        for axisName, agg in axisBlock.items():
            if not agg:
                continue
            mr = agg.get("meanReward", {}).get("meanAcrossSeeds", float("nan"))
            sr = agg.get("meanReward", {}).get("stdAcrossSeeds", float("nan"))
            te = agg.get("avgTrackingErr", {}).get("meanAcrossSeeds", float("nan"))
            st = agg.get("settlingTime", {}).get("meanAcrossSeeds", float("nan"))
            os_ = agg.get("overshootPct", {}).get("meanAcrossSeeds", float("nan"))
            lines.append(
                f"| {n} | {axisName} | {mr:.3f} ± {sr:.3f} | {te:.3f} | {st:.3f} | {os_:.2f} |"
            )
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/aux_crossconfig")
    p.add_argument(
        "--rotor_counts",
        default="4,6,8",
        help="Comma-separated rotor counts to sweep over.",
    )
    p.add_argument(
        "--axes",
        default="VERT,PITCHLON,ROLLLAT,YAWHDG",
        help="Comma-separated axis names to sweep over.",
    )
    p.add_argument("--n_batches", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--eval_episodes", type=int, default=3)
    args = p.parse_args()

    args.out_dir = os.path.abspath(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    args.seedList = list(range(args.seeds))
    rotorCounts = [int(x) for x in args.rotor_counts.split(",")]
    axisNames = [a.strip().upper() for a in args.axes.split(",")]
    # axis is set per-cell; populate a placeholder so runMainSubprocess can
    # read args.axis when it builds each subprocess command line.
    args.axis = "VERT"
    args.n_rotors = rotorCounts[0]

    out = {
        "args": {k: v for k, v in vars(args).items() if k != "seedList"},
        "cells": {},
    }
    for nRotors in rotorCounts:
        cellKey = f"n_rotors_{nRotors}"
        out["cells"][cellKey] = {}
        for axisName in axisNames:
            out["cells"][cellKey][axisName] = runCell(args, nRotors, axisName)
            with open(os.path.join(args.out_dir, "crossconfig_results.json"), "w") as f:
                json.dump(out, f, indent=2)

    writeMarkdownSummary(out, os.path.join(args.out_dir, "crossconfig_summary.md"))
    print(f"\n[runner] Wrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
