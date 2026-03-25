# Main

import argparse
import os
import time


def Train(args_namespace):
    pass


def main():
    parser_argParser = argparse.ArgumentParser()
    parser_argParser.add_argument()
    parser_argParser.add_argument("--data_path", type=str, default="./data")
    # parser_argParser.add_argument("--exp_name", type=str, required=True)
    parser_argParser.add_argument("--exp_name", type=str, default="test")
    # parser_argParser.add_argument("--n_iter", "-n", type=int, default=200)
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

    Train(args_namespace)


if __name__ == "__main__":
    main()
