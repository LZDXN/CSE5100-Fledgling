# Shared helpers for the auxiliary-experiment sweep runners.
#
# Both scripts/runRobustnessExperiments.py and scripts/runObservabilityExperiments.py
# launch main.py as a subprocess per training run, then aggregate eval-time
# metrics across seeds.  Earlier this logic was duplicated; it now lives here.

import os
import subprocess
import sys
import time

import numpy as np


def _findExistingRun(outDir: str, runName: str, axisDirName: str) -> str:
    # Look for a previously-completed run with this runName.  A run counts
    # as "complete" iff it has summary.json AND at least one actor checkpoint
    # in the per-axis subdirectory.  Returns the axis dir, or "" if none found.
    if not os.path.isdir(outDir):
        return ""
    for d in sorted(
        d for d in os.listdir(outDir) if d.endswith("_" + runName)
    ):
        axisDir = os.path.join(outDir, d, axisDirName)
        if not os.path.isdir(axisDir):
            continue
        hasSummary = os.path.exists(os.path.join(axisDir, "summary.json"))
        hasCkpt = os.path.exists(
            os.path.join(axisDir, "actor_best.pth")
        ) or os.path.exists(os.path.join(axisDir, "actor_final.pth"))
        if hasSummary and hasCkpt:
            return axisDir
    return ""


def runMainSubprocess(
    repoRoot: str,
    args,
    runName: str,
    extra: list,
    skipExisting: bool = True,
) -> str:
    # Invoke main.py as a subprocess so each training run is fully isolated
    # (own RNG state, own torch graph, own logger).  Returns the path to the
    # per-axis output directory (the <Axis> subdirectory).
    #
    # When skipExisting=True (default), reuse any prior completed run with
    # the same runName -- summary.json + checkpoint must both exist.  This
    # makes the backlog orchestrator resumable without manual bookkeeping.
    axisDirName = args.axis.capitalize()
    if skipExisting:
        existing = _findExistingRun(args.out_dir, runName, axisDirName)
        if existing:
            print(f"[runner] {runName}: SKIP (reusing {existing})", flush=True)
            return existing
    # eval_every must be <= n_batches so at least one eval fires during the
    # run (otherwise actor_best.pth is never written and downstream loadActor
    # has nothing to load).  Default heuristic: half-way through, but never
    # less than 1 and never more than n_batches.
    evalEvery = min(max(args.n_batches // 2, 1), args.n_batches)

    cmd = [
        sys.executable,
        os.path.join(repoRoot, "main.py"),
        "--axis", args.axis,
        "--no_lqr",
        "--n_rotors", str(args.n_rotors),
        "--n_batches", str(args.n_batches),
        "--batch_size", str(args.batch_size),
        "--max_steps", str(args.max_steps),
        "--eval_every", str(evalEvery),
        "--exp_name", runName,
        "--seed", str(args.seed),
        "--data_path", args.out_dir,
    ] + extra

    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env.setdefault("MPLBACKEND", "Agg")

    t0 = time.time()
    print(f"[runner] {runName}: launching ...", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=repoRoot)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        raise RuntimeError(f"main.py failed for run {runName}")
    print(f"[runner] {runName}: done in {elapsed:.1f}s", flush=True)

    # Pick the most-recent matching dir (timestamp prefix sorts naturally).
    candidates = sorted(
        d for d in os.listdir(args.out_dir) if d.endswith("_" + runName)
    )
    if not candidates:
        raise RuntimeError(f"No run directory matched suffix _{runName}")
    runDir = os.path.join(args.out_dir, candidates[-1])
    axisDir = os.path.join(runDir, args.axis.capitalize())

    # Defensive: confirm the subprocess actually wrote a checkpoint before
    # downstream code tries to load it.  Otherwise the failure mode would be
    # an opaque FileNotFoundError several call levels deep.  Accept either
    # actor_best (preferred -- written by the live-eval callback) or
    # actor_final (always written at the end of training).
    bestCkpt = os.path.join(axisDir, "actor_best.pth")
    finalCkpt = os.path.join(axisDir, "actor_final.pth")
    if not (os.path.exists(bestCkpt) or os.path.exists(finalCkpt)):
        raise RuntimeError(
            f"main.py for {runName} produced no checkpoint in {axisDir}; "
            f"check the run dir for partial output."
        )
    return axisDir


def aggregateOverSeeds(perSeed: list) -> dict:
    # perSeed is a list[dict-of-metrics], one per seed.  For each numeric
    # metric (everything but "firstTraj" and "nEpisodes"), report mean across
    # seeds and standard deviation across seeds as the variance estimator.
    if not perSeed:
        return {}
    keys = [k for k in perSeed[0] if k != "firstTraj" and k != "nEpisodes"]
    out = {"nSeeds": len(perSeed), "nEpisodes": perSeed[0]["nEpisodes"]}
    for k in keys:
        means = np.array([p[k]["mean"] for p in perSeed])
        out[k] = {
            "meanAcrossSeeds": float(np.mean(means)),
            "stdAcrossSeeds": float(np.std(means)),
            "perSeedMeans": means.tolist(),
        }
    return out
