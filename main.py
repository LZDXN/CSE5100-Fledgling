import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS",     "1")
os.environ.setdefault("MKL_NUM_THREADS",     "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")

import matplotlib
# matplotlib.use("MacOSX")

# Global libraries
import argparse
import time

import humanLQI

# Local project libraries
import multiRotorPlant
import nnController
import nnTrainingLoop
import ppoTrainer


def main():
    parser_argParser = argparse.ArgumentParser()
    parser_argParser.add_argument("--data_path",  type=str,   default="./data")
    parser_argParser.add_argument("--exp_name",   type=str,   default="test")
    parser_argParser.add_argument("--no_lqr",     action="store_true")

    # Plant
    parser_argParser.add_argument("--n_rotors",   type=int,   default=4)
    parser_argParser.add_argument("--mass",       type=float, default=1.0)
    parser_argParser.add_argument("--dt",         type=float, default=0.01)

    # Training
    parser_argParser.add_argument("--n_batches",    type=int,   default=200)
    parser_argParser.add_argument("--batch_size",   type=int,   default=2048)
    parser_argParser.add_argument("--eval_every",   type=int,   default=20)
    parser_argParser.add_argument("--n_eval_steps", type=int,   default=500)
    parser_argParser.add_argument("--z_cmd",        type=float, default=1.0)
    parser_argParser.add_argument("--max_steps",    type=int,   default=500)
    parser_argParser.add_argument("--z_max",        type=float, default=10.0)

    # PPO
    parser_argParser.add_argument("--lr",           type=float, default=3e-4)
    parser_argParser.add_argument("--gamma",        type=float, default=0.99)
    parser_argParser.add_argument("--lam",          type=float, default=0.95)
    parser_argParser.add_argument("--clip_eps",     type=float, default=0.2)
    parser_argParser.add_argument("--n_epochs",     type=int,   default=10)
    parser_argParser.add_argument("--entropy_coeff",type=float, default=0.01)

    args_namespace = parser_argParser.parse_args()

    # Map to the attribute names used in nnTrainingLoop
    args_namespace.nBatches    = args_namespace.n_batches
    args_namespace.batchSize   = args_namespace.batch_size
    args_namespace.evalEvery   = args_namespace.eval_every
    args_namespace.nEvalSteps  = args_namespace.n_eval_steps
    args_namespace.zCmd        = args_namespace.z_cmd
    args_namespace.maxSteps    = args_namespace.max_steps
    args_namespace.zMax        = args_namespace.z_max

    dataPath_str = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), args_namespace.data_path
    )
    if not os.path.exists(dataPath_str):
#     parser_argParser.add_argument("--data_path", type=str, default="./.data")
#     # parser_argParser.add_argument("--exp_name", type=str, required=True)
#     parser_argParser.add_argument("--exp_name", type=str, default="test")
#     args_namespace = parser_argParser.parse_args()

#     dataPath_str = os.path.join(
#         os.path.dirname(os.path.realpath(__file__)), args_namespace.data_path
#     )

#     if not (os.path.exists(dataPath_str)):
        os.makedirs(dataPath_str)

    logdir_str = args_namespace.exp_name + "_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir_str = os.path.join(dataPath_str, logdir_str)
    args_namespace.logdir  = logdir_str
    args_namespace.saveDir = logdir_str
    if not os.path.exists(logdir_str):
        os.makedirs(logdir_str)

    plant = multiRotorPlant.multiRotor6DOFWithXYZPositionError_class(
        vehicleMass_mv_float=args_namespace.mass,
        rotorCount_nr_int=args_namespace.n_rotors,
    )

    # Perform LQI baseline on VERT axis
    if not args_namespace.no_lqr:
        humanLQI.lqiWithLogRandomSearch(
            plant,
            multiRotorPlant.axisEnum_enumClass.VERT,
            overshootHardRejectPercent_float=0.01,
            riseTimeHardRejectSeconds_float=5,
            settlingTimeHardRejectSeconds_float=5,
            saveDir=logdir_str,
        )

    # Build NN controller and trainer
    # obs_dim = 3 for VERT axis: [z_error, z, z_dot]
    obsLen = len(list(plant.vertIdxs_slice))
    actor  = nnController.ActorMLP(obs_dim=obsLen)
    critic = nnController.CriticMLP(obs_dim=obsLen)

    trainer = ppoTrainer.PPOTrainer(
        actor, critic,
        lr=args_namespace.lr,
        gamma=args_namespace.gamma,
        lam=args_namespace.lam,
        clipEps=args_namespace.clip_eps,
        nEpochs=args_namespace.n_epochs,
        entropyCoeff=args_namespace.entropy_coeff,
    )

    # NN simulation & training loop
    batchRewards = nnTrainingLoop.train(
        plant, actor, critic, trainer, args_namespace
    )

    print("\nDone.")
    # input("Press Enter to close plots...")
#     args_namespace.logdir = logdir_str
#     if not (os.path.exists(logdir_str)):
#         os.makedirs(logdir_str)

#     plant = multiRotorPlant.multiRotor6DOFWithXYZPositionError_class()
#     # plant = multiRotorPlant.multiRotor6DOFWithXYZPositionError_class(
#     #     rotorCount_nr_int=6
#     # )
#     # plant = multiRotorPlant.multiRotor6DOFWithXYZPositionError_class(
#     #     rotorCount_nr_int=8
#     # )

#     # controller = #pytorch or however controller network is initialized

#     # Perform LQI
#     humanLQI.lqiWithLogRandomSearch(
#         plant,
#         multiRotorPlant.axisEnum_enumClass.VERT,
#         overshootHardRejectPercent_float=0.01,
#         riseTimeHardRejectSeconds_float=5,
#         settlingTimeHardRejectSeconds_float=5,
#     )

#     # humanLQI.lqiWithLogRandomSearch(
#     #     plant,
#     #     multiRotorPlant.axisEnum_enumClass.PITCHLON,
#     #     overshootHardRejectPercent_float=0.01,
#     #     riseTimeHardRejectSeconds_float=5,
#     #     settlingTimeHardRejectSeconds_float=5,
#     # )

#     # humanLQI.lqiWithLogRandomSearch(
#     #     plant,
#     #     multiRotorPlant.axisEnum_enumClass.ROLLLAT,
#     #     overshootHardRejectPercent_float=0.01,
#     #     riseTimeHardRejectSeconds_float=5,
#     #     settlingTimeHardRejectSeconds_float=5,
#     # )

#     # humanLQI.lqiWithLogRandomSearch(
#     #     plant,
#     #     multiRotorPlant.axisEnum_enumClass.YAWHDG,
#     #     overshootHardRejectPercent_float=0.01,
#     #     riseTimeHardRejectSeconds_float=5,
#     #     settlingTimeHardRejectSeconds_float=5,
#     # )

#     # Perform the NN simulation & training loop for the Z-axis
#     # nnTrainingLoop.train() # TODO: contents to train function

#     # DEBUG
#     import discreteTimeSim

#     discreteTimeSim.simRun(plant, multiRotorPlant.axisEnum_enumClass.VERT)
#     # DEBUG


if __name__ == "__main__":
    main()
