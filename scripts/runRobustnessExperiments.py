#!/usr/bin/env python
# Section 5.3 -- Robustness to disturbance and noise.
#
# Reproduces three sub-experiments from the report:
#
#   (A) 2x2 train/eval ablation on the VERT axis of the four-rotor vehicle.
#       Cells:   train clean / eval clean    (baseline)
#                train clean / eval noisy    (clean policy brittleness)
#                train noisy / eval clean    (cost of robust training)
#                train noisy / eval noisy    (robustness payoff)
#       "Noisy" combines all three disturbance channels (process, actuator,
#       force) at their default sampling ranges -- see disturbance.py for
#       citations.  This mirrors the dynamics-randomization recipe of
#       Peng et al. 2018.
#
#   (B) Single-rotor failure across rotor counts {4, 6, 8} on the VERT axis.
#       Two training conditions per count -- clean training vs. training
#       with rotor_failure_prob=0.2 -- evaluated under no-failure and
#       single-rotor-failure conditions.  Tests redundancy benefit of
#       higher rotor counts; follows Sharma et al. 2021.
#
# Each ablation cell is run for `--seeds` random seeds.  Aggregated results
# (mean +/- std across seeds) are written to the output directory as
# robustness_2x2.json, robustness_failure.json, and a Markdown summary.
#
# Total runtime is dominated by the 2x2 ablation: with --n_batches 60
# (default), seed count 3, and 4 rotors on a single axis, expect roughly
# 2 minutes per train run on a modern laptop CPU.
#
# Usage:
#   python scripts/runRobustnessExperiments.py \
#       --out_dir data/aux_robustness \
#       --seeds 3 \
#       --n_batches 60

import argparse
import json
import os
import sys

import numpy as np

# Make the project root importable when this script is launched directly.
_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _THIS_DIR)

import multiRotorPlant  # noqa: E402
import auxEval  # noqa: E402
from disturbance import DisturbanceConfig  # noqa: E402
from observability import ObservabilityConfig  # noqa: E402  # noqa: F401
from _runnerCommon import runMainSubprocess, aggregateOverSeeds  # noqa: E402


def _trainCondition(args, condition: str, seed: int):
    # condition in {"clean", "noisy", "failure"}.
    args = argparse.Namespace(**vars(args))
    args.seed = seed
    extra = []
    if condition == "noisy":
        extra += [
            "--enable_process_noise",
            "--enable_actuator_noise",
            "--enable_force_disturbance",
        ]
    elif condition == "failure":
        extra += ["--rotor_failure_prob", "0.2"]
    elif condition != "clean":
        raise ValueError(f"unknown condition {condition!r}")
    runName = f"robust_{condition}_n{args.n_rotors}_s{seed}"
    return runMainSubprocess(_REPO_ROOT, args, runName, extra)


def _evalCells(args, runDir: str, plant):
    # Returns dict of cell -> aggregated metrics (mean +/- std across episodes).
    cleanCfg = None
    noisyCfg = DisturbanceConfig(
        enableProcessNoise=True,
        enableActuatorNoise=True,
        enableForceDisturbance=True,
    )
    cells = {
        "evalClean": cleanCfg,
        "evalNoisy": noisyCfg,
    }
    out = {}
    for name, distCfg in cells.items():
        out[name] = auxEval.evaluatePolicy(
            runDir,
            plant,
            axisName=args.axis,
            rCmd=1.0,
            dt=0.01,
            maxSteps=args.max_steps,
            hiddenDim=64,
            obsCfgEval=None,
            distCfgEval=distCfg,
            nEpisodes=args.eval_episodes,
            seed=12345,
        )
    return out


def _evalFailureCells(args, runDir: str, plant):
    # No-failure vs. forced-failure (probability 1.0 to make the eval deterministic).
    out = {}
    out["evalNoFailure"] = auxEval.evaluatePolicy(
        runDir,
        plant,
        axisName=args.axis,
        rCmd=1.0,
        dt=0.01,
        maxSteps=args.max_steps,
        hiddenDim=64,
        distCfgEval=None,
        nEpisodes=args.eval_episodes,
        seed=12345,
    )
    out["evalFailure"] = auxEval.evaluatePolicy(
        runDir,
        plant,
        axisName=args.axis,
        rCmd=1.0,
        dt=0.01,
        maxSteps=args.max_steps,
        hiddenDim=64,
        distCfgEval=DisturbanceConfig(rotorFailureProb=1.0),
        nEpisodes=args.eval_episodes,
        seed=12345,
    )
    return out


def runTwoByTwo(args):
    out = {}
    plant = multiRotorPlant.multiRotor6DOFWithXYZPositionError_class(
        rotorCount_nr_int=args.n_rotors
    )
    for trainCond in ("clean", "noisy"):
        perSeedByEval = {"evalClean": [], "evalNoisy": []}
        for seed in args.seedList:
            runDir = _trainCondition(args, trainCond, seed)
            cells = _evalCells(args, runDir, plant)
            for evalName, metrics in cells.items():
                perSeedByEval[evalName].append(metrics)
        out[f"train_{trainCond}"] = {
            ev: aggregateOverSeeds(perSeedByEval[ev]) for ev in perSeedByEval
        }
    return out


def runFailure(args):
    # Iterate over rotor counts.  For each, train a clean and a failure-augmented
    # policy, evaluate both under no-failure / forced-failure conditions.
    out = {}
    for nRotors in args.failureRotorCounts:
        rotorOut = {}
        local = argparse.Namespace(**vars(args))
        local.n_rotors = nRotors
        plant = multiRotorPlant.multiRotor6DOFWithXYZPositionError_class(
            rotorCount_nr_int=nRotors
        )
        for trainCond in ("clean", "failure"):
            perSeedByEval = {"evalNoFailure": [], "evalFailure": []}
            for seed in args.seedList:
                runDir = _trainCondition(local, trainCond, seed)
                cells = _evalFailureCells(local, runDir, plant)
                for evalName, metrics in cells.items():
                    perSeedByEval[evalName].append(metrics)
            rotorOut[f"train_{trainCond}"] = {
                ev: aggregateOverSeeds(perSeedByEval[ev]) for ev in perSeedByEval
            }
        out[f"n_rotors_{nRotors}"] = rotorOut
    return out


def writeMarkdownSummary(out: dict, path: str):
    # Compact human-readable rendering of the 2x2 + failure tables.
    lines = ["# Section 5.3 robustness aggregates", ""]
    if "twoByTwo" in out:
        lines.append("## 2x2 train/eval ablation (VERT axis)")
        lines.append("")
        lines.append("| train \\ eval  | meanReward (clean) | meanReward (noisy) | trackErr (clean) | trackErr (noisy) |")
        lines.append("|--------------|--------------------|--------------------|------------------|------------------|")
        for tc in ("train_clean", "train_noisy"):
            row = out["twoByTwo"].get(tc, {})
            mc = row.get("evalClean", {}).get("meanReward", {}).get("meanAcrossSeeds", float("nan"))
            sc = row.get("evalClean", {}).get("meanReward", {}).get("stdAcrossSeeds", float("nan"))
            mn = row.get("evalNoisy", {}).get("meanReward", {}).get("meanAcrossSeeds", float("nan"))
            sn = row.get("evalNoisy", {}).get("meanReward", {}).get("stdAcrossSeeds", float("nan"))
            tc_clean = row.get("evalClean", {}).get("avgTrackingErr", {}).get("meanAcrossSeeds", float("nan"))
            tc_noisy = row.get("evalNoisy", {}).get("avgTrackingErr", {}).get("meanAcrossSeeds", float("nan"))
            lines.append(
                f"| {tc} | {mc:.3f} ± {sc:.3f} | {mn:.3f} ± {sn:.3f} | {tc_clean:.3f} | {tc_noisy:.3f} |"
            )
        lines.append("")

    if "failure" in out:
        lines.append("## Single-rotor failure across rotor counts")
        lines.append("")
        lines.append("| n_rotors | train | eval | meanReward | trackErr | settlingTime |")
        lines.append("|---------|-------|------|------------|----------|--------------|")
        for rkey, rblock in out["failure"].items():
            n = rkey.split("_")[-1]
            for tcond in ("train_clean", "train_failure"):
                if tcond not in rblock:
                    continue
                for evcond in ("evalNoFailure", "evalFailure"):
                    cell = rblock[tcond].get(evcond, {})
                    if not cell:
                        continue
                    mr = cell.get("meanReward", {}).get("meanAcrossSeeds", float("nan"))
                    sr = cell.get("meanReward", {}).get("stdAcrossSeeds", float("nan"))
                    te = cell.get("avgTrackingErr", {}).get("meanAcrossSeeds", float("nan"))
                    st = cell.get("settlingTime", {}).get("meanAcrossSeeds", float("nan"))
                    lines.append(
                        f"| {n} | {tcond.split('_')[-1]} | {evcond.replace('eval', '').lower()} | {mr:.3f} ± {sr:.3f} | {te:.3f} | {st:.3f} |"
                    )
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/aux_robustness")
    p.add_argument("--axis", default="VERT")
    p.add_argument("--n_rotors", type=int, default=4)
    p.add_argument("--n_batches", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--eval_episodes", type=int, default=5)
    p.add_argument("--skip_failure", action="store_true")
    p.add_argument("--skip_2x2", action="store_true")
    p.add_argument(
        "--failure_rotor_counts",
        default="4,6,8",
        help="Comma-separated rotor counts for the failure ablation.",
    )
    args = p.parse_args()

    args.out_dir = os.path.abspath(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    args.seedList = list(range(args.seeds))
    args.failureRotorCounts = [int(x) for x in args.failure_rotor_counts.split(",")]

    # Load pre-existing results from a prior partial invocation, so that calling
    # this runner with --skip_2x2 first and then with --skip_failure (or vice
    # versa) accumulates into the same robustness_all.json instead of clobbering.
    allPath = os.path.join(args.out_dir, "robustness_all.json")
    if os.path.exists(allPath):
        with open(allPath) as f:
            out = json.load(f)
        out["args"] = {k: v for k, v in vars(args).items() if k != "seedList"}
    else:
        out = {"args": {k: v for k, v in vars(args).items() if k != "seedList"}}

    if not args.skip_2x2:
        out["twoByTwo"] = runTwoByTwo(args)
        with open(os.path.join(args.out_dir, "robustness_2x2.json"), "w") as f:
            json.dump({"args": out["args"], "twoByTwo": out["twoByTwo"]}, f, indent=2)
    if not args.skip_failure:
        out["failure"] = runFailure(args)
        with open(os.path.join(args.out_dir, "robustness_failure.json"), "w") as f:
            json.dump({"args": out["args"], "failure": out["failure"]}, f, indent=2)

    with open(allPath, "w") as f:
        json.dump(out, f, indent=2)
    writeMarkdownSummary(out, os.path.join(args.out_dir, "robustness_summary.md"))
    print(f"\n[runner] Wrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
