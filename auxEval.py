# Cross-condition policy evaluator for the auxiliary experiments.
#
# Loads a previously-trained per-axis actor and runs it deterministically
# against the discrete-time plant under a configurable observation regime
# and disturbance injector.  The evaluator is the workhorse of the 2x2
# train/eval ablation in Section 5.3 (because the off-diagonal cells re-use
# a clean-trained policy under disturbed evaluation, no retraining) and of
# the rotor-failure runs in the same section.
#
# References:
#   - The 2x2 train/eval split mirrors the ablation protocol of
#     Peng et al. 2018, who measure both the brittleness of clean-trained
#     policies and the robustness payoff of dynamics-randomized training.

# Global libraries
import json
import os
from dataclasses import asdict
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch

# Local project libraries
import multiRotorPlant
import nnController
import nnTrainingLoop
from disturbance import DisturbanceConfig, DisturbanceInjector
from observability import ObservabilityConfig, ObservationProcessor, buildRestrictedC


# ---------------------------------------------------------------------------
# Plant / actor reconstruction from a saved run directory
# ---------------------------------------------------------------------------


_AXIS_BY_NAME = {
    "PITCHLON": multiRotorPlant.axisEnum_enumClass.PITCHLON,
    "ROLLLAT": multiRotorPlant.axisEnum_enumClass.ROLLLAT,
    "VERT": multiRotorPlant.axisEnum_enumClass.VERT,
    "YAWHDG": multiRotorPlant.axisEnum_enumClass.YAWHDG,
}


def loadActor(runDir: str, plant, axisName: str, hiddenDim: int, tag: str = "best"):
    # Reconstruct the actor (architecture must match what was trained), then
    # load saved weights.  baseObsLen is determined from the saved aux_config
    # if present (so partial-obs runs use the right C dim), else falls back
    # to the plant default.
    auxPath = os.path.join(runDir, "aux_config.json")
    if os.path.exists(auxPath):
        with open(auxPath) as f:
            aux = json.load(f)
        keepIdxs = aux["observability"].get("keepIdxs")
        historyLen = aux["observability"].get("historyLen", 1)
    else:
        keepIdxs = None
        historyLen = 1

    axis = _AXIS_BY_NAME[axisName]
    fullStateDim = plant.plantAxisHandler(axis=axis, obsIdxs=None)[0].shape[0]
    cMatrix = buildRestrictedC(fullStateDim, keepIdxs)
    obsLen = int(cMatrix.shape[0]) * max(historyLen, 1)

    actor = nnController.ActorMLP(
        obs_dim=obsLen, action_dim=plant.rotorCount_nr_int, hidden=hiddenDim
    )
    weightPath = os.path.join(runDir, f"actor_{tag}.pth")
    actor.load_state_dict(torch.load(weightPath, weights_only=True))
    actor.eval()
    return actor, cMatrix, historyLen


# ---------------------------------------------------------------------------
# Cross-condition deterministic evaluation
# ---------------------------------------------------------------------------


def evaluatePolicy(
    runDir: str,
    plant,
    axisName: str,
    rCmd: float = 1.0,
    dt: float = 0.01,
    maxSteps: int = 500,
    hiddenDim: int = 64,
    obsCfgEval: Optional[ObservabilityConfig] = None,
    distCfgEval: Optional[DisturbanceConfig] = None,
    nEpisodes: int = 1,
    seed: Optional[int] = None,
) -> dict:
    # Run `nEpisodes` deterministic-eval rollouts of a previously-trained
    # actor under the supplied evaluation-time observability/disturbance
    # configs.  Returns aggregate metrics (mean +/- std across episodes) plus
    # the first-episode trajectory for plotting.

    # Build plant matrices using the *training-time* obs mask (so the actor
    # sees the obs shape it was trained on), regardless of what eval-time
    # obs noise/delay we layer on.
    actor, cMatrixTrain, historyLenTrain = loadActor(
        runDir, plant, axisName, hiddenDim
    )
    axis = _AXIS_BY_NAME[axisName]

    # Use train-time C for state -> obs projection.  Eval-time obs noise /
    # delay are layered on top via obsCfgEval (which inherits keepIdxs and
    # historyLen from training, but may add noise/delay).
    A, B, C, E = plant.discreteLQIPlant(
        plant.plantAxisHandler(axis=axis, obsIdxs=cMatrixTrain), dt
    )

    if obsCfgEval is None:
        # Default: same obs regime as training.
        obsCfgEval = ObservabilityConfig(
            keepIdxs=None,  # already applied via cMatrixTrain
            obsNoiseSigma=0.0,
            delaySteps=0,
            historyLen=historyLenTrain,
        )
    else:
        # Force history-len to match training so the actor's input dim aligns.
        obsCfgEval.historyLen = historyLenTrain
        # Train-time keepIdxs already restricted C; do not re-restrict here.
        obsCfgEval.keepIdxs = None

    rng = np.random.default_rng(seed)
    distInjector = (
        DisturbanceInjector(
            distCfgEval,
            nRotors=plant.rotorCount_nr_int,
            vehicleMass=plant.vehicleMass_mv_float,
            rng=rng,
        )
        if distCfgEval is not None and distCfgEval.isActive
        else None
    )

    # Build a SimpleNamespace mirroring the args contract of _runEval.
    args = SimpleNamespace(
        rCmd=rCmd,
        dt=dt,
        maxSteps=maxSteps,
        velPenalty=0.3,
        overshootWeight=3.0,
        rotorBalanceCoeff=0.05,
        energyCoeff=0.05,
        overshootTolPct=0.1,
    )
    hoverPct = plant.rotorHoverThrustPercent_fthov_float

    # Per-episode metrics
    meanRewards = []
    avgTrackingErrs = []
    finalTrackingErrs = []
    riseTimes = []
    settlingTimes = []
    overshootPcts = []
    undershootPcts = []
    integratedEffort = []  # int |u - hover| dt across all rotors

    firstTraj = None
    for epIdx in range(nEpisodes):
        # Re-seed disturbance per episode for variance estimation.
        traj = nnTrainingLoop._runEval(
            A,
            B,
            C,
            E,
            actor,
            args,
            hoverPct,
            axis,
            obsCfg=obsCfgEval,
            distInjector=distInjector,
        )
        if firstTraj is None:
            firstTraj = traj

        meanRewards.append(float(np.mean(traj["rewards"])))
        avgTrackingErrs.append(float(np.mean(np.abs(traj["pos_err"]))))
        finalTrackingErrs.append(float(traj["pos_err"][-1].item()))

        stepM = nnTrainingLoop.computeStepMetrics(traj, rCmd, dt)
        riseTimes.append(stepM["riseTime"])
        settlingTimes.append(stepM["settlingTime"])
        overshootPcts.append(stepM["overshoot"])
        undershootPcts.append(stepM["undershoot"])
        # Per-rotor effort: shape (nRotors, nSteps).  Sum |u - hover| across rotors then integrate.
        u = traj["u"]
        effort = float(np.sum(np.abs(u - hoverPct)) * dt)
        integratedEffort.append(effort)

    def _meanstd(xs):
        arr = np.array(xs, dtype=float)
        # Treat NaN settling/rise (no crossing) as the truncation horizon.
        if np.any(np.isnan(arr)):
            arr = np.where(np.isnan(arr), maxSteps * dt, arr)
        return float(np.mean(arr)), float(np.std(arr))

    mr_m, mr_s = _meanstd(meanRewards)
    at_m, at_s = _meanstd(avgTrackingErrs)
    ft_m, ft_s = _meanstd(finalTrackingErrs)
    rt_m, rt_s = _meanstd(riseTimes)
    st_m, st_s = _meanstd(settlingTimes)
    os_m, os_s = _meanstd(overshootPcts)
    us_m, us_s = _meanstd(undershootPcts)
    ef_m, ef_s = _meanstd(integratedEffort)

    return {
        "nEpisodes": nEpisodes,
        "meanReward": {"mean": mr_m, "std": mr_s},
        "avgTrackingErr": {"mean": at_m, "std": at_s},
        "finalTrackingErr": {"mean": ft_m, "std": ft_s},
        "riseTime": {"mean": rt_m, "std": rt_s},
        "settlingTime": {"mean": st_m, "std": st_s},
        "overshootPct": {"mean": os_m, "std": os_s},
        "undershootPct": {"mean": us_m, "std": us_s},
        "integratedEffort": {"mean": ef_m, "std": ef_s},
        "firstTraj": {
            "time": firstTraj["time"].tolist(),
            "pos": firstTraj["pos"].tolist(),
            "vel": firstTraj["vel"].tolist(),
            "pos_err": np.asarray(firstTraj["pos_err"]).reshape(-1).tolist(),
            "rewards": firstTraj["rewards"].tolist(),
        },
    }


def saveEvalResult(saveDir: str, name: str, result: dict, distCfg=None, obsCfg=None):
    os.makedirs(saveDir, exist_ok=True)
    payload = dict(result)
    if distCfg is not None:
        payload["distCfg"] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in asdict(distCfg).items()
            if k != "failedRotorIdx"
        }
    if obsCfg is not None:
        payload["obsCfg"] = asdict(obsCfg)
    with open(os.path.join(saveDir, f"eval_{name}.json"), "w") as f:
        json.dump(payload, f, indent=2)
