# Main

# Global libraries
import argparse
import os
import time

# Local project libraries
import multiRotorPlant
import nnController
import humanLQI
import nnTrainingLoop


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
    # controller = #pytorch or however controller network is initialized

    # Perform LQI on the Z-axis
    humanLQI.lqiWithLogRandomSearch(plant, multiRotorPlant.axisEnum_enumClass.VERT)

    # Perform the NN simulation & training loop for the Z-axis
    # nnTrainingLoop.train() # TODO: contents to train function


if __name__ == "__main__":
    main()
