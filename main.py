# Main

# Global libraries
import argparse
import os
import time

# # Local project libraries
# import multiRotorPlant
# import nnController
# import humanLQI
# import nnTrainingLoop

# DEBUG
# Local project libraries
import multiRotorPlant
import humanLQI

# DEBUG


def main():
    parser_argParser = argparse.ArgumentParser()
    parser_argParser.add_argument("--data_path", type=str, default="./.data")
    # parser_argParser.add_argument("--exp_name", type=str, required=True)
    parser_argParser.add_argument("--exp_name", type=str, default="test")
    args_namespace = parser_argParser.parse_args()

    dataPath_str = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), args_namespace.data_path
    )

    if not (os.path.exists(dataPath_str)):
        os.makedirs(dataPath_str)

    logdir_str = args_namespace.exp_name + "_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir_str = os.path.join(dataPath_str, logdir_str)
    args_namespace.logdir = logdir_str
    if not (os.path.exists(logdir_str)):
        os.makedirs(logdir_str)

    plant = multiRotorPlant.multiRotor6DOFWithXYZPositionError_class()
    # plant = multiRotorPlant.multiRotor6DOFWithXYZPositionError_class(
    #     rotorCount_nr_int=6
    # )
    # plant = multiRotorPlant.multiRotor6DOFWithXYZPositionError_class(
    #     rotorCount_nr_int=8
    # )

    # controller = #pytorch or however controller network is initialized

    # Perform LQI
    humanLQI.lqiWithLogRandomSearch(
        plant,
        multiRotorPlant.axisEnum_enumClass.VERT,
        overshootHardRejectPercent_float=0.01,
        riseTimeHardRejectSeconds_float=5,
        settlingTimeHardRejectSeconds_float=5,
    )

    # humanLQI.lqiWithLogRandomSearch(
    #     plant,
    #     multiRotorPlant.axisEnum_enumClass.PITCHLON,
    #     overshootHardRejectPercent_float=0.01,
    #     riseTimeHardRejectSeconds_float=5,
    #     settlingTimeHardRejectSeconds_float=5,
    # )

    # humanLQI.lqiWithLogRandomSearch(
    #     plant,
    #     multiRotorPlant.axisEnum_enumClass.ROLLLAT,
    #     overshootHardRejectPercent_float=0.01,
    #     riseTimeHardRejectSeconds_float=5,
    #     settlingTimeHardRejectSeconds_float=5,
    # )

    # humanLQI.lqiWithLogRandomSearch(
    #     plant,
    #     multiRotorPlant.axisEnum_enumClass.YAWHDG,
    #     overshootHardRejectPercent_float=0.01,
    #     riseTimeHardRejectSeconds_float=5,
    #     settlingTimeHardRejectSeconds_float=5,
    # )

    # Perform the NN simulation & training loop for the Z-axis
    # nnTrainingLoop.train() # TODO: contents to train function


if __name__ == "__main__":
    main()
