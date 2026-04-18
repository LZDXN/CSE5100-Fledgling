# Global libraries
import argparse
import time
import os
import numpy as np

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


def main():
    parser_argParser = argparse.ArgumentParser()
    parser_argParser.add_argument("--data_path", type=str, default="./data")
    # parser_argParser.add_argument("--exp_name", type=str, required=True)
    parser_argParser.add_argument("--exp_name", type=str, default="test")
    parser_argParser.add_argument("--no_lqr", action="store_true")

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
    parser_argParser.add_argument("--n_eval_steps", type=int, default=500)
    parser_argParser.add_argument("--r_cmd", type=float, default=1.0)
    parser_argParser.add_argument("--max_steps", type=int, default=500)
    parser_argParser.add_argument("--rotor_balance_coeff", type=float, default=0.05)
    parser_argParser.add_argument("--energy_coeff", type=float, default=0.05)
    parser_argParser.add_argument("--overshoot_tol_pct", type=float, default=0.1)

    # PPO
    parser_argParser.add_argument("--lr", type=float, default=3e-4)
    parser_argParser.add_argument("--gamma", type=float, default=0.99)
    parser_argParser.add_argument("--lam", type=float, default=0.95)
    parser_argParser.add_argument("--clip_eps", type=float, default=0.2)
    parser_argParser.add_argument("--n_epochs", type=int, default=10)
    parser_argParser.add_argument("--entropy_coeff", type=float, default=0.01)

    args_namespace = parser_argParser.parse_args()

    # Map to the attribute names used in nnTrainingLoop
    args_namespace.nBatches = args_namespace.n_batches
    args_namespace.batchSize = args_namespace.batch_size
    args_namespace.evalEvery = args_namespace.eval_every
    args_namespace.nEvalSteps = args_namespace.n_eval_steps
    args_namespace.rCmd = args_namespace.r_cmd
    args_namespace.maxSteps = args_namespace.max_steps
    args_namespace.overshootTolPct = args_namespace.overshoot_tol_pct
    args_namespace.rotorBalanceCoeff = args_namespace.rotor_balance_coeff
    args_namespace.energyCoeff = args_namespace.energy_coeff

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

    for axn, ax in enumerate(axis):
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

        # Build NN controller and trainer
        # Observation dimensions is up to the below for full state feedback/but for
        # systems with limited measurement (IMU only, for example) observation
        # dimension is nonzero rows of plant C matrix
        # obs_dim = up to 5 for PITCHLON axis: [x_error, x, x_dot, theta, theta_dot]
        # obs_dim = up to 5 for ROLLLAT axis: [y_error, y, y_dot, phi, phi_dot]
        # obs_dim = up to 3 for VERT axis: [z_error, z, z_dot]
        # obs_dim = up to 3 for YAWHDG axis: [psi_error, psi, psi_dot]
        # obsLen = sum of nonzero rows in plant C matrix, which for full state feedback is as above
        obsLen = np.sum(
            np.count_nonzero(plant.plantAxisHandler(axis=ax, obsIdxs=None)[2], axis=1)
        )
        actor = nnController.ActorMLP(
            obs_dim=obsLen, action_dim=plant.rotorCount_nr_int
        )
        critic = nnController.CriticMLP(obs_dim=obsLen)

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
