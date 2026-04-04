# Discrete-time simulating functions for deep RL training loop

# Global libraries
import control as ct
import numpy as np
import torch  # ???

# Local project libraries
import multiRotorPlant


def simStep(
    plant,
    controller,
    referenceCommand,
    xdata,
    ydata,
    udata,
    step=0,
):
    A, B, C, E = plant
    ydata[step] = C @ xdata[step]

    # TODO: Make properly interface with controller code
    with torch.no_grad():
        udata[step] = u = controller(ydata[step])

    xdata[step + 1] = A @ xdata[step] + B @ u + E @ referenceCommand[step]


def simRun(
    plantObject: multiRotorPlant.multiRotor6DOFWithXYZPositionError_class,
    simAxis: multiRotorPlant.axisEnum_enumClass,
    controller: torch.nn.Module,
    initialConditions=None,
    referenceCommand=None,
    simFreq_f_Hz_int=1000,
    simTime_T_s_float=10,
    simDiscreteSteps_N_U_int=None,
):
    simTimeStep_dt_sinv_float = 1 / simFreq_f_Hz_int
    if simDiscreteSteps_N_U_int is None:
        simDiscreteSteps_N_U_int = int(simTime_T_s_float / simTimeStep_dt_sinv_float)
    else:
        simDiscreteSteps_N_U_int = simDiscreteSteps_N_U_int
        simTime_T_s_float = simDiscreteSteps_N_U_int * simFreq_f_Hz_int

    discretePlant = plantObject.discreteLQIPlant(
        plantObject.plantAxisHandler(simAxis), simTimeStep_dt_sinv_float
    )

    if initialConditions is None:
        initialConditions = np.zeros_like(
            plantObject.plantAxisHandler(simAxis)[0].shape
        )

    if referenceCommand is None:
        referenceCommand = np.ones((simDiscreteSteps_N_U_int, 1))

    tdata = np.zeros((simDiscreteSteps_N_U_int, 1))
    xdata = np.zeros(
        (simDiscreteSteps_N_U_int, plantObject.plantAxisHandler(simAxis[0].shape[0]))
    )
    ydata = np.zeros(
        (simDiscreteSteps_N_U_int, plantObject.plantAxisHandler(simAxis[2].shape[0]))
    )
    udata = np.zeros(
        (simDiscreteSteps_N_U_int, plantObject.plantAxisHandler(simAxis[1].shape[0]))
    )

    xdata[0, :] = initialConditions

    for step in range(simDiscreteSteps_N_U_int):
        simStep(discretePlant, controller, referenceCommand, xdata, ydata, udata, step)
