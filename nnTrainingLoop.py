# Code to encapsulate the nn controller training loop to involve simming the plant
# dynamics, determining the reward function in terms of plant outputs, and training the
# controller network.

# Global libraries
import os

import numpy as np

# Local project libraries
import multiRotorPlant

# import nnController
# import discreteTimeSim
import ppoTrainer
import trainingArtifacts
from utilsPlots import LivePlotter

# Auxiliary-experiment hooks. Both modules expose a configurable injector that
# is a no-op by default, so importing them does not change baseline behavior.
from disturbance import DisturbanceConfig, DisturbanceInjector
from observability import ObservabilityConfig, ObservationProcessor, buildRestrictedC


def scoreFunction(
    state,
    ref=0.0,
    actions=None,
    velPenalty=0.1,
    overshootWeight=3.0,
    rotorBalancePenalty=0.05,
    energyPenalty=0.05,
):
    # Reward for one timestep. Range (0, 1] -- 1 at perfect tracking, approaches 0 far away.
    # Works for all axes: state[1] is always the instantaneous position, state[2] is always
    # the rate (velocity or angular rate for the axis).
    #
    # Secondary shaping terms (require actions to be passed):
    #   rotorBalancePenalty: ||abs(a) - mean(abs(a))||² -- zero when all rotors equal magnitude
    #   energyPenalty:       ||a||²                     -- penalizes overall control effort (R-matrix analog)
    trackErr = state[1] - ref
    rate = state[2]

    # Asymmetric position penalty: crossing past the reference costs overshootWeight× more.
    posPenalty = (overshootWeight if trackErr > 0 else 1.0) * trackErr**2

    # Base return -- tracking + damping only. Uncomment to test without secondary terms.
    # return float(np.exp(-(posPenalty + velPenalty * rate**2)))

    imbalance = 0.0
    energy = 0.0
    if actions is not None:
        absAct = np.abs(actions)
        dev = absAct - np.mean(absAct)
        imbalance = float(np.dot(dev, dev))
        energy = float(np.dot(actions, actions))

    # return float(
    #     np.exp(-(trackErr**2 + velPenalty * rate**2 + rotorBalancePenalty * imbalance))
    # )
    return float(
        np.exp(
            -(
                trackErr**2
                + velPenalty * rate**2
                + rotorBalancePenalty * imbalance
                + energyPenalty * energy
            )
        )
    )


def _fmtMetric(v):
    return f"{v:.3f}" if not np.isnan(v) else "N/A"


def computeStepMetrics(traj, refCmd, dt, riseThreshold=0.9, settlingBand=0.02):
    pos = traj["pos"]
    t = traj["time"]

    riseTarget = riseThreshold * refCmd
    riseIdxs = np.where(pos >= riseTarget)[0]
    riseTime = t[riseIdxs[0]] if len(riseIdxs) > 0 else float("nan")

    band = settlingBand * abs(refCmd)
    outsideBand = np.where(np.abs(pos - refCmd) > band)[0]
    settlingTime = t[outsideBand[-1]] + dt if len(outsideBand) > 0 else 0.0

    overshoot = max(0.0, (np.max(pos) - refCmd) / abs(refCmd) * 100)

    firstReach = riseIdxs[0] if len(riseIdxs) > 0 else len(pos)
    undershoot = (
        max(0.0, -np.min(pos[:firstReach]) / abs(refCmd) * 100)
        if firstReach > 0
        else 0.0
    )

    return {
        "riseTime": riseTime,
        "settlingTime": settlingTime,
        "overshoot": overshoot,
        "undershoot": undershoot,
    }


def _actionToRotorThrusts(actions, hoverPct=0.5):
    # Map a per-rotor action vector to absolute thrust commands.
    # actions: array of shape (nRotors,), each element in [-1, 1]
    #   action_i =  0 -> hoverPct for rotor i (trim)
    #   action_i =  1 -> 2*hoverPct for rotor i (max)
    #   action_i = -1 -> 0 for rotor i (min)
    return np.clip(hoverPct + np.clip(actions, -1, 1) * hoverPct, 0, 1)


def train(plant, axis, actor, critic, trainer, args, writer=None):
    # PPO training loop for a single decoupled axis.
    # Runs the discrete-time plant step-by-step, stores experience in RolloutBuffer,
    # scores with scoreFunction, and updates the network via PPOTrainer each batch.
    # Plots update live every evalEvery batches.

    nRotors = plant.rotorCount_nr_int
    hoverPct = plant.rotorHoverThrustPercent_fthov_float

    # Pull auxiliary-experiment configuration from args.  Both default to
    # no-op so callers that do not opt in get the original training path.
    obsCfg = ObservabilityConfig.fromArgs(args)
    distCfg = DisturbanceConfig.fromArgs(args)

    # Build the per-axis (A, B, C, E).  When obsCfg restricts the rows of C,
    # we override the plant default with a row-restricted selector.
    fullStateDim = plant.plantAxisHandler(axis)[0].shape[0]
    cOverride = buildRestrictedC(fullStateDim, obsCfg.keepIdxs)
    A, B, C, E = plant.discreteLQIPlant(
        plant.plantAxisHandler(axis, obsIdxs=cOverride), args.dt
    )

    # Per-rollout observation/disturbance processors.  Reset on every episode
    # boundary inside the rollout loop below.
    obsProc = ObservationProcessor(obsCfg, baseObsDim=C.shape[0])
    distInj = DisturbanceInjector(
        distCfg,
        nRotors=nRotors,
        vehicleMass=plant.vehicleMass_mv_float,
    )

    refCmd = np.array([args.rCmd])
    x0 = np.zeros(A.shape[0])
    # Small perturbation scale for randomized episode resets (5% of reference command).
    # Helps the policy generalize beyond a single initial condition.
    _icPerturbScale = 0.05 * abs(args.rCmd)

    saveDir = getattr(args, "saveDir", None)
    plotter = LivePlotter(nRotors=nRotors, hoverPct=hoverPct, saveDir=saveDir)
    logger = trainingArtifacts.TrainingLogger(saveDir)

    if obsCfg.isActive or distCfg.isActive:
        trainingArtifacts.saveAuxConfig(saveDir, obsCfg, distCfg)

    batchRewards = []
    bestEvalReward = -np.inf

    print(
        f"\nStarting PPO training -- {args.nBatches} batches x {args.batchSize} steps\n",
        flush=True,
    )

    for batchIdx in range(args.nBatches):
        buffer = ppoTrainer.RolloutBuffer()
        x = x0 + np.random.uniform(-_icPerturbScale, _icPerturbScale, size=x0.shape)
        stepCount = 0
        done = False
        truncated = False
        episodeStartTime = 0.0  # seconds since current episode reset (for sinusoidal F_d)
        obsProc.reset()
        distInj.sampleEpisode()

        for step in range(args.batchSize):
            rawObs = (C @ x).astype(np.float32)
            obs = obsProc.process(rawObs)

            action, logProb = actor.getAction(obs)
            value = critic.getValue(obs)

            # Linear plant expects thrust *deviation* from hover (δu), not absolute thrust.
            # δu = u_abs - hoverPct  →  δu=0 at hover, B@δu=0 ↔ no net force/torque.
            actionForPlant = distInj.perturbAction(action)
            u_abs = _actionToRotorThrusts(actionForPlant, hoverPct)
            xNext = (
                A @ x
                + B @ (u_abs - hoverPct)
                + E @ refCmd
                + distInj.stateNoise(x.shape)
                + distInj.forceDelta(episodeStartTime, args.dt, x.shape)
            )

            reward = scoreFunction(
                xNext,
                ref=refCmd[0],
                actions=action,
                velPenalty=args.velPenalty,
                overshootWeight=args.overshootWeight,
                rotorBalancePenalty=args.rotorBalanceCoeff,
                energyPenalty=args.energyCoeff,
            )

            stepCount += 1
            # done = abs(xNext[1]) > args.yMax
            done |= xNext[1] > (
                refCmd * (1 + args.overshootTolPct)
            )  # overshoot reference command by tol % as primary early term condition
            if (
                axis == multiRotorPlant.axisEnum_enumClass.VERT
                or np.sign(refCmd[0]) > 0
            ):
                done |= (
                    xNext[1] < 0.0
                )  # Gone negative position with positive command or hit the ground for vert
            if (
                axis == multiRotorPlant.axisEnum_enumClass.PITCHLON
                or axis == multiRotorPlant.axisEnum_enumClass.ROLLLAT
            ):
                done |= (
                    np.abs(xNext[3]) > 25 * np.pi / 180
                )  # Small angle approximation divergence for translational axes

            truncated = stepCount >= args.maxSteps

            buffer.store(obs, action, reward, logProb, value, done or truncated)

            x = xNext
            episodeStartTime += args.dt

            if done or truncated:
                x = x0 + np.random.uniform(-_icPerturbScale, _icPerturbScale, size=x0.shape)
                stepCount = 0
                done = False
                truncated = False
                episodeStartTime = 0.0
                obsProc.reset()
                distInj.sampleEpisode()

        lastRawObs = (C @ x).astype(np.float32)
        lastObs = obsProc.process(lastRawObs)
        lastVal = 0.0 if (done or truncated) else critic.getValue(lastObs)
        buffer.finish(lastVal)

        losses = trainer.update(buffer)
        meanReward = buffer.meanReward()
        batchRewards.append(meanReward)

        print(
            f"Batch {batchIdx + 1:>4}/{args.nBatches}"
            f" reward={meanReward:.4f}"
            f" policyLoss={losses['policyLoss']:+.4f}"
            f" valueLoss={losses['valueLoss']:.4f}"
            f" entropy={losses['entropy']:.4f}",
            flush=True,
        )

        logger.logBatch(batchIdx, meanReward, losses)

        if writer is not None:
            writer.add_scalar("train/meanReward", meanReward, batchIdx)
            writer.add_scalar("train/policyLoss", losses["policyLoss"], batchIdx)
            writer.add_scalar("train/valueLoss", losses["valueLoss"], batchIdx)
            writer.add_scalar("train/entropy", losses["entropy"], batchIdx)
            writer.add_scalar("train/meanAdvantage", losses["meanAdvantage"], batchIdx)

        if (batchIdx + 1) % args.evalEvery == 0:
            evalTraj = _runEval(
                A, B, C, E, actor, args, hoverPct, axis, obsCfg=obsCfg
            )
            stepMetrics = computeStepMetrics(evalTraj, args.rCmd, args.dt)
            plotter.update(batchRewards, evalTraj, args.rCmd, stepMetrics=stepMetrics)

            evalMeanReward = float(np.mean(evalTraj["rewards"]))
            evalTrackingErr = float(np.mean(np.abs(evalTraj["pos_err"])))
            print(
                f"  [eval] meanReward={evalMeanReward:.4f}"
                f"  avgTrackingErr={evalTrackingErr:.4f}"
                f"  finalTrackingErr={evalTraj['pos_err'][-1].item():.4f}"
                f"  Tr={_fmtMetric(stepMetrics['riseTime'])}s"
                f"  Ts={_fmtMetric(stepMetrics['settlingTime'])}s"
                f"  OS={stepMetrics['overshoot']:.2f}%"
                f"  US={stepMetrics['undershoot']:.2f}%",
                flush=True,
            )

            if saveDir and evalMeanReward > bestEvalReward:
                bestEvalReward = evalMeanReward
                trainingArtifacts.saveNNCheckpoint(actor, critic, saveDir, tag="best")
                print(
                    f"  [best] New best eval reward={bestEvalReward:.4f} -- model checkpoint saved",
                    flush=True,
                )

            if writer:
                writer.add_scalar(
                    "eval/meanReward", float(np.mean(evalTraj["rewards"])), batchIdx
                )
                writer.add_scalar("eval/trackingError", evalTrackingErr, batchIdx)

    plotter.saveFinal()
    logger.close()

    if saveDir:
        trainingArtifacts.saveNNCheckpoint(actor, critic, saveDir, tag="final")

    if saveDir and os.path.exists(os.path.join(saveDir, "actor_best.pth")):
        trainingArtifacts.loadNNCheckpoint(actor, critic, saveDir, tag="best")
        bestFinalTraj = _runEval(
            A, B, C, E, actor, args, hoverPct, axis, obsCfg=obsCfg
        )
        bestMetrics = computeStepMetrics(bestFinalTraj, args.rCmd, args.dt)
        plotter.update(batchRewards, bestFinalTraj, args.rCmd, stepMetrics=bestMetrics)
        plotter.saveBest()
        bestTrackingErr = float(np.mean(np.abs(bestFinalTraj["pos_err"])))
        bestRewardFinal = float(np.mean(bestFinalTraj["rewards"]))
        print(
            f"\n[best model eval] meanReward={bestRewardFinal:.4f}"
            f"  avgTrackingErr={bestTrackingErr:.4f}"
            f"  finalTrackingErr={bestFinalTraj['pos_err'][-1].item():.4f}"
            f"  Tr={_fmtMetric(bestMetrics['riseTime'])}s"
            f"  Ts={_fmtMetric(bestMetrics['settlingTime'])}s"
            f"  OS={bestMetrics['overshoot']:.2f}%"
            f"  US={bestMetrics['undershoot']:.2f}%",
            flush=True,
        )

        trainingArtifacts.saveRunSummary(
            saveDir,
            {
                "axis": str(axis).split(".")[-1],
                "nBatches": args.nBatches,
                "batchSize": args.batchSize,
                "bestEvalMeanReward": bestRewardFinal,
                "bestAvgTrackingErr": bestTrackingErr,
                "bestFinalTrackingErr": float(bestFinalTraj["pos_err"][-1].item()),
                "riseTime": bestMetrics["riseTime"],
                "settlingTime": bestMetrics["settlingTime"],
                "overshootPct": bestMetrics["overshoot"],
                "undershootPct": bestMetrics["undershoot"],
            },
        )

    return batchRewards


def _runEval(A, B, C, E, actor, args, hoverPct, axis, obsCfg=None, distInjector=None):
    # obsCfg: ObservabilityConfig used at training time (frame-stack, masking,
    #         delay, noise) -- mirrored at eval so the actor sees an obs of
    #         the same shape it was trained on.
    # distInjector: optional DisturbanceInjector for evaluating a clean-trained
    #         policy under a noisy/disturbed environment (the off-diagonal
    #         cells of the 2x2 train-eval ablation in Section 5.3).
    refCmd = np.array([args.rCmd])
    x = np.zeros(A.shape[0])

    obsProc = ObservationProcessor(
        obsCfg if obsCfg is not None else ObservabilityConfig(),
        baseObsDim=C.shape[0],
    )
    obsProc.reset()
    if distInjector is not None:
        distInjector.sampleEpisode()

    timeHist = []
    posHist = []
    velHist = []
    posErrHist = []
    rewHist = []
    uHist = []

    for i in range(args.maxSteps):
        rawObs = (C @ x).astype(np.float32)
        obs = obsProc.process(rawObs)
        action = actor.getActionDeterministic(obs)
        actionForPlant = (
            distInjector.perturbAction(action) if distInjector is not None else action
        )
        u_abs = _actionToRotorThrusts(actionForPlant, hoverPct)
        x = A @ x + B @ (u_abs - hoverPct) + E @ refCmd
        if distInjector is not None:
            x = (
                x
                + distInjector.stateNoise(x.shape)
                + distInjector.forceDelta(i * args.dt, args.dt, x.shape)
            )

        timeHist.append(i * args.dt)
        posHist.append(x[1])
        velHist.append(x[2])
        posErrHist.append(refCmd - x[1])
        rewHist.append(
            scoreFunction(
                x,
                ref=refCmd[0],
                actions=action,
                velPenalty=args.velPenalty,
                overshootWeight=args.overshootWeight,
                rotorBalancePenalty=args.rotorBalanceCoeff,
                energyPenalty=args.energyCoeff,
            )
        )
        uHist.append(u_abs.copy())

        # if abs(x[1]) > args.yMax:
        #     break
        if x[1] > (
            refCmd * (1 + args.overshootTolPct)
        ):  # overshoot reference command by tol % as primary early term condition
            break
        if axis == multiRotorPlant.axisEnum_enumClass.VERT or np.sign(refCmd[0]) > 0:
            if (
                x[1] < 0.0
            ):  # Gone negative position with positive command or hit the ground for vert
                break
        if (
            axis == multiRotorPlant.axisEnum_enumClass.PITCHLON
            or axis == multiRotorPlant.axisEnum_enumClass.ROLLLAT
        ):
            if (
                np.abs(x[3]) > 25 * np.pi / 180
            ):  # Small angle approximation divergence for translational axes
                break

    return {
        "time": np.array(timeHist),
        "pos": np.array(posHist),
        "vel": np.array(velHist),
        "pos_err": np.array(posErrHist),
        "rewards": np.array(rewHist),
        # shape (nRotors, nSteps)
        "u": np.array(uHist).T,
    }
