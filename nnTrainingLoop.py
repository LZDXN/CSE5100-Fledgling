# Code to encapsulate the nn controller training loop to involve simming the plant
# dynamics, determining the reward function in terms of plant outputs, and training the
# controller network.

# TODO

# Global libraries

# Local project libraries
import multiRotorPlant
import nnController
import discreteTimeSim


def scoreFunction():
    # TODO: produce score function for controller based on minimizing error relative
    # to command, and/or other factors
    # Gets called in train()
    raise NotImplementedError


def train():
    # TODO: loop training for loop for number of training cycles set by arg in main().
    # Invoke discrete time sim simRun() for some number of steps for a batch of
    # data, discard data if sim time history goes out of "bounds" for some arbitrary
    # conditions, and re-run simRun() for remaining steps to batch size. Invoke
    # scoreFunction() to get Q values or etc. for training, update the network for
    # whatever DRL method is being used, then loop.
    raise NotImplementedError
