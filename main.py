# Global libraries
import argparse
import json
import random
import time
import os
import numpy as np
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# Local project libraries
import multiRotorPlant
import nnController
import humanLQI
import nnTrainingLoop
import ppoTrainer
import observability


def main():
    parser_argParser = argparse.ArgumentParser()
    parser_argParser.add_argument("--data_path", type=str, default="./data")
    # parser_argParser.add_argument("--exp_name", type=str, required=True)
    parser_argParser.add_argument("--exp_name", type=str, default="test")
    parser_argParser.add_argument("--no_lqr", action="store_true")
    # -1 = unseeded (non-deterministic). Any non-negative value seeds numpy/torch/random.
    parser_argParser.add_argument("--seed", type=int, default=-1)

    # Axis
    parser_argParser.add_argument("--axis", type=str, default="ALL")
    # parser_argParser.add_argument("--axis", type=str, default="VERT")

    # Plant
    parser_argParser.add_argument("--n_rotors", type=int, default=4)
    parser_argParser.add_argument("--mass", type=float, default=1.0)
    parser_argParser.add_argument("--dt", type=float, default=0.01)

    # Training
    parser_argParser.add_argument("--n_batches", type=int, default=200)
    parser_argParser.add_argument("--batch_size", type=int, default=2048)
    parser_argParser.add_argument("--eval_every", type=int, default=20)
    parser_argParser.add_argument("--max_steps", type=int, default=500)
    parser_argParser.add_argument("--r_cmd", type=float, default=1.0)
    parser_argParser.add_argument("--rotor_balance_coeff", type=float, default=0.05)
    parser_argParser.add_argument("--energy_coeff", type=float, default=0.05)
    parser_argParser.add_argument("--vel_penalty", type=float, default=0.3)
    parser_argParser.add_argument("--overshoot_tol_pct", type=float, default=0.1)
    parser_argParser.add_argument("--overshoot_weight", type=float, default=3.0)
    parser_argParser.add_argument("--hidden_dim", type=int, default=64)

    # PPO
    parser_argParser.add_argument("--lr", type=float, default=3e-4)
    parser_argParser.add_argument("--gamma", type=float, default=0.99)
    parser_argParser.add_argument("--lam", type=float, default=0.95)
    parser_argParser.add_argument("--clip_eps", type=float, default=0.2)
    parser_argParser.add_argument("--n_epochs", type=int, default=5)
    parser_argParser.add_argument("--entropy_coeff", type=float, default=0.01)

    # ----- Auxiliary-experiment knobs (Section 5.3 robustness, 5.4 partial obs) -----
    # All default to no-op so omitting them reproduces the baseline pipeline.
    parser_argParser.add_argument(
        "--enable_process_noise",
        action="store_true",
        help="Inject state-side process noise (Peng et al. 2018 dynamics randomization).",
    )
    parser_argParser.add_argument(
        "--enable_actuator_noise",
        action="store_true",
        help="Inject Gaussian per-rotor actuator noise (Chen et al. 2025 ESC jitter analog).",
    )
    parser_argParser.add_argument(
        "--enable_force_disturbance",
        action="store_true",
        help="Inject external sinusoidal force disturbance (wind-gust analog, Wang et al. 2021).",
    )
    parser_argParser.add_argument(
        "--rotor_failure_prob",
        type=float,
        default=0.0,
        help="Probability of zeroing one rotor at episode start (Sharma et al. 2021).",
    )
    parser_argParser.add_argument(
        "--obs_keep_idxs",
        type=str,
        default="all",
        help="Comma-separated state indices to keep in the obs (e.g. '0,2'). 'all' = full state.",
    )
    parser_argParser.add_argument(
        "--obs_noise_sigma",
        type=float,
        default=0.0,
        help="Std of additive Gaussian noise on the obs vector.",
    )
    parser_argParser.add_argument(
        "--obs_delay_steps",
        type=int,
        default=0,
        help="Observation delay in simulator steps.",
    )
    parser_argParser.add_argument(
        "--obs_history_len",
        type=int,
        default=1,
        help="Frame-stack length for the observation passed to the actor (Mnih et al. 2015).",
    )

    args_namespace = parser_argParser.parse_args()

    if args_namespace.seed >= 0:
        random.seed(args_namespace.seed)
        np.random.seed(args_namespace.seed)
        torch.manual_seed(args_namespace.seed)

    # Map to the attribute names used in nnTrainingLoop
    args_namespace.nBatches = args_namespace.n_batches
    args_namespace.batchSize = args_namespace.batch_size
    args_namespace.evalEvery = args_namespace.eval_every
    args_namespace.rCmd = args_namespace.r_cmd
    args_namespace.maxSteps = args_namespace.max_steps
    args_namespace.overshootTolPct = args_namespace.overshoot_tol_pct
    args_namespace.rotorBalanceCoeff = args_namespace.rotor_balance_coeff
    args_namespace.energyCoeff = args_namespace.energy_coeff
    args_namespace.velPenalty = args_namespace.vel_penalty
    args_namespace.overshootWeight = args_namespace.overshoot_weight
    args_namespace.hiddenDim = args_namespace.hidden_dim

    dataPath_str = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), args_namespace.data_path
    )
    if not (os.path.exists(dataPath_str)):
        os.makedirs(dataPath_str)

    # logdir_str = args_namespace.exp_name + "_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir_str = time.strftime("%Y-%m-%d_%H-%M-%S") + "_" + args_namespace.exp_name
    logdir_str = os.path.join(dataPath_str, logdir_str)
    if not os.path.exists(logdir_str):
        os.makedirs(logdir_str)

    with open(os.path.join(logdir_str, "args.json"), "w") as f:
        json.dump(vars(args_namespace), f, indent=2)

    match args_namespace.axis:
        case "pitchlon" | "PITCHLON" | "x" | "X":
            axis = [multiRotorPlant.axisEnum_enumClass.PITCHLON]
            ax_str = ["Pitchlon"]
        case "rolllat" | "ROLLLAT" | "y" | "Y":
            axis = [multiRotorPlant.axisEnum_enumClass.ROLLLAT]
            ax_str = ["Rolllat"]
        case "vert" | "VERT" | "z" | "Z":
            axis = [multiRotorPlant.axisEnum_enumClass.VERT]
            ax_str = ["Vert"]
        case "yawhdg" | "YAWHDG" | "yaw" | "YAW" | "psi" | "PSI":
            axis = [multiRotorPlant.axisEnum_enumClass.YAWHDG]
            ax_str = ["Yawhdg"]
        case "all" | "ALL" | _:
            axis = [
                multiRotorPlant.axisEnum_enumClass.PITCHLON,
                multiRotorPlant.axisEnum_enumClass.ROLLLAT,
                multiRotorPlant.axisEnum_enumClass.VERT,
                multiRotorPlant.axisEnum_enumClass.YAWHDG,
            ]
            ax_str = [
                "Pitchlon",
                "Rolllat",
                "Vert",
                "Yawhdg",
            ]

    plant = multiRotorPlant.multiRotor6DOFWithXYZPositionError_class(
        rotorCount_nr_int=args_namespace.n_rotors
    )

    trainingStartTime = time.perf_counter()
    for axn, ax in enumerate(axis):
        axisStartTime = time.perf_counter()
        axdir_str = os.path.join(logdir_str, ax_str[axn])
        args_namespace.logdir = axdir_str
        args_namespace.saveDir = axdir_str
        if not os.path.exists(axdir_str):
            os.makedirs(axdir_str)

        # Perform LQI baseline for comparison data
        if not args_namespace.no_lqr:
            humanLQI.lqiWithLogRandomSearch(
                plant,
                ax,
                overshootHardRejectPercent_float=0.01,
                riseTimeHardRejectSeconds_float=5,
                settlingTimeHardRejectSeconds_float=5,
                saveDir=axdir_str,
            )

        # Build NN controller and trainer.
        # Observation dimension is determined jointly by:
        #   (a) the per-axis state-vector size projected through C
        #       (full state feedback: 3 for VERT/YAWHDG, 5 for PITCHLON/ROLLLAT)
        #   (b) the partial-observability mask --obs_keep_idxs, which restricts
        #       C to a subset of state rows for the Section 5.4 ablations
        #   (c) the frame-stack length --obs_history_len, which multiplies the
        #       per-step obs dimension when the policy sees a sliding window
        #       of past observations (Mnih et al. 2015 frame stacking)
        obsCfg = observability.ObservabilityConfig.fromArgs(args_namespace)
        fullStateDim = plant.plantAxisHandler(axis=ax, obsIdxs=None)[0].shape[0]
        cMatrix = observability.buildRestrictedC(fullStateDim, obsCfg.keepIdxs)
        baseObsLen = int(cMatrix.shape[0])
        obsLen = baseObsLen * max(obsCfg.historyLen, 1)
        actor = nnController.ActorMLP(
            obs_dim=obsLen,
            action_dim=plant.rotorCount_nr_int,
            hidden=args_namespace.hiddenDim,
        )
        critic = nnController.CriticMLP(obs_dim=obsLen, hidden=args_namespace.hiddenDim)

        trainer = ppoTrainer.PPOTrainer(
            actor,
            critic,
            lr=args_namespace.lr,
            gamma=args_namespace.gamma,
            lam=args_namespace.lam,
            clipEps=args_namespace.clip_eps,
            nEpochs=args_namespace.n_epochs,
            entropyCoeff=args_namespace.entropy_coeff,
        )

        # NN simulation & training loop
        batchRewards = nnTrainingLoop.train(
            plant, ax, actor, critic, trainer, args_namespace
        )
        axisEndTime = time.perf_counter()
        print(f"Axis training time: {axisEndTime - axisStartTime:.1f} seconds")

    trainingEndTime = time.perf_counter()
    print(f"Total elapsed time: {trainingEndTime - trainingStartTime:.1f} seconds")
    print("\nDone.")
    # input("Press Enter to close plots...")


# Old, pre-merge code
#     # plant = multiRotorPlant.multiRotor6DOFWithXYZPositionError_class(
#     #     rotorCount_nr_int=6
#     # )
#     # plant = multiRotorPlant.multiRotor6DOFWithXYZPositionError_class(
#     #     rotorCount_nr_int=8
#     # )
#
#     # controller = #pytorch or however controller network is initialized
#
#     # Perform LQI
#     humanLQI.lqiWithLogRandomSearch(
#         plant,
#         multiRotorPlant.axisEnum_enumClass.VERT,
#         overshootHardRejectPercent_float=0.01,
#         riseTimeHardRejectSeconds_float=5,
#         settlingTimeHardRejectSeconds_float=5,
#     )
#
#     # humanLQI.lqiWithLogRandomSearch(
#     #     plant,
#     #     multiRotorPlant.axisEnum_enumClass.PITCHLON,
#     #     overshootHardRejectPercent_float=0.01,
#     #     riseTimeHardRejectSeconds_float=5,
#     #     settlingTimeHardRejectSeconds_float=5,
#     # )
#
#     # humanLQI.lqiWithLogRandomSearch(
#     #     plant,
#     #     multiRotorPlant.axisEnum_enumClass.ROLLLAT,
#     #     overshootHardRejectPercent_float=0.01,
#     #     riseTimeHardRejectSeconds_float=5,
#     #     settlingTimeHardRejectSeconds_float=5,
#     # )
#
#     # humanLQI.lqiWithLogRandomSearch(
#     #     plant,
#     #     multiRotorPlant.axisEnum_enumClass.YAWHDG,
#     #     overshootHardRejectPercent_float=0.01,
#     #     riseTimeHardRejectSeconds_float=5,
#     #     settlingTimeHardRejectSeconds_float=5,
#     # )
#
#     # Perform the NN simulation & training loop for the Z-axis
#     # nnTrainingLoop.train() # TODO: contents to train function
#
#     # DEBUG
#     import discreteTimeSim
#
#     discreteTimeSim.simRun(plant, multiRotorPlant.axisEnum_enumClass.VERT)
#     # DEBUG


if __name__ == "__main__":
    main()
