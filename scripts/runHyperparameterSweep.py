#!/usr/bin/env python
# Section 5.5 -- Hyperparameter sensitivity (one-at-a-time).
#
# For each (knob, setting) pair where `setting != default`, train --seeds
# policies on the VERT axis of the four-rotor vehicle and record converged
# eval reward + time-domain metrics.  The default cell is also run once so
# the sweep table includes a reference point for each knob.
#
# Per the report Section 5.5, the one-at-a-time design is chosen over Latin
# hypercube so per-knob sensitivity can be read directly from the table.
#
# Usage:
#   python scripts/runHyperparameterSweep.py \
#       --out_dir data/aux_hparams \
#       --seeds 3 --n_batches 60

import argparse
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _THIS_DIR)

import multiRotorPlant  # noqa: E402
import auxEval  # noqa: E402
from _runnerCommon import runMainSubprocess, aggregateOverSeeds  # noqa: E402


# Knob -> (cli_flag, [low, default, high]).  Order matters for table layout.
HPARAM_GRID = {
    "lr":               ("--lr",                  [1e-4, 3e-4, 1e-3]),
    "gamma":            ("--gamma",               [0.95, 0.99, 0.995]),
    "lam":              ("--lam",                 [0.9, 0.95, 0.97]),
    "clip_eps":         ("--clip_eps",            [0.1, 0.2, 0.3]),
    "entropy_coeff":    ("--entropy_coeff",       [0.0, 0.01, 0.05]),
    "hidden_dim":       ("--hidden_dim",          [32, 64, 128]),
    "batch_size":       ("--batch_size",          [1024, 2048, 4096]),
    "vel_penalty":      ("--vel_penalty",         [0.15, 0.3, 0.6]),
    "overshoot_weight": ("--overshoot_weight",    [1.0, 3.0, 6.0]),
    "rotor_balance":    ("--rotor_balance_coeff", [0.0, 0.05, 0.1]),
    "energy":           ("--energy_coeff",        [0.0, 0.05, 0.1]),
}

# Knobs that must be passed as int on the CLI.
_INT_KNOBS = {"hidden_dim", "batch_size"}


def _settingToString(knob: str, value) -> str:
    if knob in _INT_KNOBS:
        return str(int(value))
    return f"{value:g}"


def _runOneCell(args, knob: str, settingIdx: int, plant) -> dict:
    flag, settings = HPARAM_GRID[knob]
    setting = settings[settingIdx]
    perSeed = []
    extraBase = [flag, _settingToString(knob, setting)]
    # batch_size affects --batch_size on main.py, which we also use to pass
    # to the runner -- so keep them consistent for the subprocess command.
    if knob == "batch_size":
        local_batch = int(setting)
    else:
        local_batch = args.batch_size

    for seed in args.seedList:
        local = argparse.Namespace(**vars(args))
        local.seed = seed
        local.batch_size = local_batch
        runDir = runMainSubprocess(
            _REPO_ROOT,
            local,
            f"hparam_{knob}_{settingIdx}_s{seed}",
            extra=extraBase,
        )
        metrics = auxEval.evaluatePolicy(
            runDir,
            plant,
            axisName=args.axis,
            rCmd=1.0,
            dt=0.01,
            maxSteps=args.max_steps,
            hiddenDim=64 if knob != "hidden_dim" else int(setting),
            obsCfgEval=None,
            distCfgEval=None,
            nEpisodes=args.eval_episodes,
            seed=12345,
        )
        perSeed.append(metrics)
    agg = aggregateOverSeeds(perSeed)
    agg["setting"] = setting
    return agg


def runSweep(args) -> dict:
    plant = multiRotorPlant.multiRotor6DOFWithXYZPositionError_class(
        rotorCount_nr_int=args.n_rotors
    )
    out = {}
    for knob in args.knobList:
        if knob not in HPARAM_GRID:
            print(f"[runner] skipping unknown knob: {knob}")
            continue
        settings = HPARAM_GRID[knob][1]
        knobOut = {"low": None, "default": None, "high": None}
        labelByIdx = {0: "low", 1: "default", 2: "high"}
        for settingIdx in range(len(settings)):
            label = labelByIdx[settingIdx]
            knobOut[label] = _runOneCell(args, knob, settingIdx, plant)
        out[knob] = knobOut
    return out


def writeMarkdownSummary(out: dict, path: str):
    lines = ["# Section 5.5 hyperparameter sensitivity (one-at-a-time)", ""]
    lines.append("| knob | low (value, reward) | default (value, reward) | high (value, reward) |")
    lines.append("|------|---------------------|-------------------------|----------------------|")
    for knob, knobOut in out["sweep"].items():
        cells = []
        for label in ("low", "default", "high"):
            cell = knobOut.get(label) or {}
            val = cell.get("setting", float("nan"))
            mr = cell.get("meanReward", {}).get("meanAcrossSeeds", float("nan"))
            sr = cell.get("meanReward", {}).get("stdAcrossSeeds", float("nan"))
            cells.append(f"{val:g}, {mr:.3f}±{sr:.3f}")
        lines.append(f"| {knob} | {cells[0]} | {cells[1]} | {cells[2]} |")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/aux_hparams")
    p.add_argument("--axis", default="VERT")
    p.add_argument("--n_rotors", type=int, default=4)
    p.add_argument("--n_batches", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--eval_episodes", type=int, default=3)
    p.add_argument(
        "--knobs",
        default=",".join(HPARAM_GRID.keys()),
        help="Comma-separated knob names from HPARAM_GRID.",
    )
    args = p.parse_args()

    args.out_dir = os.path.abspath(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    args.seedList = list(range(args.seeds))
    args.knobList = [k.strip() for k in args.knobs.split(",")]

    out = {
        "args": {k: v for k, v in vars(args).items() if k != "seedList"},
        "sweep": {},
    }
    sweep = runSweep(args)
    out["sweep"] = sweep
    with open(os.path.join(args.out_dir, "hparams_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    writeMarkdownSummary(out, os.path.join(args.out_dir, "hparams_summary.md"))
    print(f"\n[runner] Wrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
