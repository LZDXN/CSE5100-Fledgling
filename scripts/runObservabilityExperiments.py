#!/usr/bin/env python
# Section 5.4 -- Restricted-observability ablation.
#
# Trains a separate per-axis policy under each of the partial-observability
# regimes documented in observability.REGIME_PRESETS:
#
#   (1) full state                                    [e_z, z, z_dot]      baseline
#   (2) inertial-only             (drop pos)          [e_z,    z_dot]
#   (3) position-only             (drop vel)          [e_z, z       ]
#   (4) noisy full state          (5% Gauss noise)
#   (5) delayed full state        (3-step lag)
#   (6) inertial-only + frame-stack(K=4) recovery
#   (7) delayed-full + frame-stack(K=4) recovery
#
# Following Mnih et al. 2015 (DQN frame stacking) and the partial-observability
# discussion of Peng et al. 2018, we use frame-stacking as the lightweight
# recovery mechanism in regimes (6)/(7) -- these are the rows that the
# Section 5.4 conclusion identifies as recoverable from a feedforward MLP.
#
# Per regime, train --seeds independent runs and record converged eval
# reward and time-domain metrics; aggregate across seeds.
#
# Usage:
#   python scripts/runObservabilityExperiments.py \
#       --out_dir data/aux_observability \
#       --seeds 3 \
#       --n_batches 60

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
from observability import REGIME_PRESETS  # noqa: E402
from _runnerCommon import runMainSubprocess, aggregateOverSeeds  # noqa: E402


def _presetToCli(preset: dict) -> list:
    extra = []
    if preset.get("obs_keep_idxs", "all") != "all":
        extra += ["--obs_keep_idxs", preset["obs_keep_idxs"]]
    if preset.get("obs_noise_sigma", 0.0):
        extra += ["--obs_noise_sigma", str(preset["obs_noise_sigma"])]
    if preset.get("obs_delay_steps", 0):
        extra += ["--obs_delay_steps", str(preset["obs_delay_steps"])]
    if preset.get("obs_history_len", 1) > 1:
        extra += ["--obs_history_len", str(preset["obs_history_len"])]
    return extra


def runRegime(args, regimeName: str, plant) -> dict:
    preset = REGIME_PRESETS[regimeName]
    extra = _presetToCli(preset)
    perSeed = []
    for seed in args.seedList:
        local = argparse.Namespace(**vars(args))
        local.seed = seed
        runDir = runMainSubprocess(
            _REPO_ROOT, local, f"obs_{regimeName}_n{args.n_rotors}_s{seed}", extra
        )
        # Evaluate under the same observability regime (no extra eval-time
        # noise / disturbance).  The actor's stored aux_config.json already
        # encodes the training-time obs config.
        metrics = auxEval.evaluatePolicy(
            runDir,
            plant,
            axisName=args.axis,
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
    lines = ["# Section 5.4 partial-observability aggregates", ""]
    lines.append("| regime | meanReward | trackErr | settlingTime | overshoot% |")
    lines.append("|--------|------------|----------|--------------|------------|")
    for name, agg in out["regimes"].items():
        if not agg:
            continue
        mr = agg.get("meanReward", {}).get("meanAcrossSeeds", float("nan"))
        sr = agg.get("meanReward", {}).get("stdAcrossSeeds", float("nan"))
        te = agg.get("avgTrackingErr", {}).get("meanAcrossSeeds", float("nan"))
        st = agg.get("settlingTime", {}).get("meanAcrossSeeds", float("nan"))
        os_ = agg.get("overshootPct", {}).get("meanAcrossSeeds", float("nan"))
        lines.append(
            f"| {name} | {mr:.3f} ± {sr:.3f} | {te:.3f} | {st:.3f} | {os_:.2f} |"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/aux_observability")
    p.add_argument("--axis", default="VERT")
    p.add_argument("--n_rotors", type=int, default=4)
    p.add_argument("--n_batches", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--eval_episodes", type=int, default=5)
    p.add_argument(
        "--regimes",
        default="full,inertial_only,position_only,noisy_full,delayed_full,inertial_only_stacked,delayed_full_stacked",
        help="Comma-separated regime names from observability.REGIME_PRESETS.",
    )
    args = p.parse_args()

    args.out_dir = os.path.abspath(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    args.seedList = list(range(args.seeds))

    plant = multiRotorPlant.multiRotor6DOFWithXYZPositionError_class(
        rotorCount_nr_int=args.n_rotors
    )

    out = {
        "args": {k: v for k, v in vars(args).items() if k != "seedList"},
        "regimes": {},
    }
    for name in args.regimes.split(","):
        name = name.strip()
        if name not in REGIME_PRESETS:
            print(f"[runner] skipping unknown regime: {name}")
            continue
        out["regimes"][name] = runRegime(args, name, plant)
        # Incrementally persist after every regime so a long sweep can be
        # interrupted without losing prior results.
        with open(os.path.join(args.out_dir, "observability_results.json"), "w") as f:
            json.dump(out, f, indent=2)

    writeMarkdownSummary(out, os.path.join(args.out_dir, "observability_summary.md"))
    print(f"\n[runner] Wrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
