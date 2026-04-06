# Code to encapsulate the nn controller training loop to involve simming the plant
# dynamics, determining the reward function in terms of plant outputs, and training the
# controller network.

# Global libraries
import numpy as np

# Local project libraries
import multiRotorPlant
import nnController
import discreteTimeSim
import ppoTrainer
import trainingArtifacts
from utilsPlots import LivePlotter


def scoreFunction(state, zRef, velPenalty=0.1):
    # Reward for one timestep. Range (0, 1] -- 1 at perfect hover, approaches 0 far away.
    # state is VERT axis: [z_error, z, z_dot]
    # state[1] = z position
    zError = zRef - state[1]
    # state[2] = z velocity
    zDot   = state[2]
    return float(np.exp(-(zError ** 2 + velPenalty * zDot ** 2)))


def _actionToRotorThrusts(action, nRotors, hoverPct=0.5):
    # Map scalar action in [-1, 1] to a per-rotor thrust vector.
    #   action =  0 -> hoverPct per rotor (steady hover)
    #   action =  1 -> 2*hoverPct per rotor (max climb)
    #   action = -1 -> 0 per rotor (free fall)
    scalar    = float(np.clip(action, -1, 1))
    uPerRotor = np.clip(hoverPct + scalar * hoverPct, 0, 1)
    return np.full(nRotors, uPerRotor)


def train(plant, actor, critic, trainer, args, writer=None):
    # PPO training loop for VERT (z-axis).
    # Runs the discrete-time plant step-by-step, stores experience in RolloutBuffer,
    # scores with scoreFunction, and updates the network via PPOTrainer each batch.
    # Plots update live every evalEvery batches.

    simAxis = multiRotorPlant.axisEnum_enumClass.VERT
    A, B, C, E = plant.discreteLQIPlant(
        plant.plantAxisHandler(simAxis), args.dt
    )
    nRotors  = plant.rotorCount_nr_int
    hoverPct = plant.rotorHoverThrustPercent_fthov_float

    refCmd = np.array([args.zCmd])
    x0     = np.zeros(A.shape[0])

    saveDir = getattr(args, "saveDir", None)
    plotter = LivePlotter(nRotors=nRotors, hoverPct=hoverPct, saveDir=saveDir)
    logger  = trainingArtifacts.TrainingLogger(saveDir)

    batchRewards = []

    print(f"\nStarting PPO training -- {args.nBatches} batches x {args.batchSize} steps\n", flush=True)

    for batchIdx in range(args.nBatches):
        buffer    = ppoTrainer.RolloutBuffer()
        x         = x0.copy()
        stepCount = 0
        done      = False
        truncated = False

        for step in range(args.batchSize):
            obs = (C @ x).astype(np.float32)

            action, logProb = actor.getAction(obs)
            value           = critic.getValue(obs)

            # Linear plant expects thrust *deviation* from hover (δu), not absolute thrust.
            # δu = u_abs - hoverPct  →  δu=0 at hover, B@δu=0 ↔ no net vertical force.
            u_abs = _actionToRotorThrusts(action, nRotors, hoverPct)
            xNext = A @ x + B @ (u_abs - hoverPct) + E @ refCmd

            reward = scoreFunction(xNext, args.zCmd)

            stepCount += 1
            zError    = args.zCmd - xNext[1]
            done      = abs(zError) > args.zMax
            truncated = stepCount >= args.maxSteps

            buffer.store(obs, action, reward, logProb, value, done or truncated)

            x = xNext

            if done or truncated:
                x         = x0.copy()
                stepCount = 0
                done      = False
                truncated = False

        lastObs = (C @ x).astype(np.float32)
        lastVal = 0.0 if (done or truncated) else critic.getValue(lastObs)
        buffer.finish(lastVal)

        losses     = trainer.update(buffer)
        meanReward = buffer.meanReward()
        batchRewards.append(meanReward)

        print(
            f"Batch {batchIdx + 1:>4}/{args.nBatches}"
            f"reward={meanReward:.4f}"
            f"policyLoss={losses['policyLoss']:+.4f}"
            f"valueLoss={losses['valueLoss']:.4f}"
            f"entropy={losses['entropy']:.4f}",
            flush=True,
        )

        logger.logBatch(batchIdx, meanReward, losses)

        if writer is not None:
            writer.add_scalar("train/meanReward",meanReward,batchIdx)
            writer.add_scalar("train/policyLoss",losses["policyLoss"],batchIdx)
            writer.add_scalar("train/valueLoss",losses["valueLoss"],batchIdx)
            writer.add_scalar("train/entropy",losses["entropy"],batchIdx)
            writer.add_scalar("train/meanAdvantage",losses["meanAdvantage"],batchIdx)

        if (batchIdx + 1) % args.evalEvery == 0:
            evalTraj = _runEval(A, B, C, E, actor, args, nRotors, hoverPct)
            plotter.update(batchRewards, evalTraj, args.zCmd)

            evalTrackingErr = float(np.mean(np.abs(args.zCmd - evalTraj["z"])))
            print(f"  [eval] meanReward={np.mean(evalTraj['rewards']):.4f}  trackingErr={evalTrackingErr:.4f} m", flush=True)

            if writer:
                writer.add_scalar("eval/meanReward",    float(np.mean(evalTraj["rewards"])), batchIdx)
                writer.add_scalar("eval/trackingError", evalTrackingErr,                     batchIdx)

    plotter.saveFinal()
    logger.close()

    if saveDir:
        trainingArtifacts.saveNNCheckpoint(actor, critic, saveDir, tag="final")

    return batchRewards


def _runEval(A, B, C, E, actor, args, nRotors, hoverPct):
    refCmd = np.array([args.zCmd])
    x      = np.zeros(A.shape[0])

    timeHist = []
    zHist    = []
    zDotHist = []
    rewHist  = []
    uHist    = []

    for i in range(args.nEvalSteps):
        obs    = (C @ x).astype(np.float32)
        action = actor.getActionDeterministic(obs)
        u_abs  = _actionToRotorThrusts(action, nRotors, hoverPct)
        x      = A @ x + B @ (u_abs - hoverPct) + E @ refCmd

        zError = args.zCmd - x[1]
        timeHist.append(i * args.dt)
        zHist.append(x[1])
        zDotHist.append(x[2])
        rewHist.append(scoreFunction(x, args.zCmd))
        uHist.append(u_abs.copy())

        if abs(zError) > args.zMax:
            break

    return {
        "time":    np.array(timeHist),
        "z":       np.array(zHist),
        "z_dot":   np.array(zDotHist),
        "rewards": np.array(rewHist),
        # shape (nRotors, nSteps)
        "u":       np.array(uHist).T,
    }


# def scoreFunction():
#     # TODO: produce score function for controller based on minimizing error relative
#     # to command, and/or other factors
#     # Gets called in train()
#     raise NotImplementedError


# def train():
#     # TODO: loop training for loop for number of training cycles set by arg in main().
#     # Invoke discrete time sim simRun() for some number of steps for a batch of
#     # data, discard data if sim time history goes out of "bounds" for some arbitrary
#     # conditions, and re-run simRun() for remaining steps to batch size. Invoke
#     # scoreFunction() to get Q values or etc. for training, update the network for
#     # whatever DRL method is being used, then loop.
#     raise NotImplementedError
