# Discrete-time simulating functions for deep RL training loop

# Global libraries
import control as ct
import numpy as np

# import torch  # ???

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
    maxSteps=1,
):
    A, B, C, E = plant
    ydata[step] = C @ xdata[step]

    # # TODO: Make properly interface with controller code
    # with torch.no_grad():
    #     udata[step] = u = controller(ydata[step])

    u = -controller @ xdata[step]

    if step < maxSteps - 1:
        xdata[step + 1] = A @ xdata[step] + B @ u + E @ referenceCommand[step]


def simRun(
    plant_plant: multiRotorPlant.multiRotor6DOFWithXYZPositionError_class,
    axis_axisEnum: multiRotorPlant.axisEnum_enumClass,
    # controller: torch.nn.Module,
    # DEBUG
    controller=None,
    # DEBUG
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

    discretePlant = plant_plant.discreteLQIPlant(
        plant_plant.plantAxisHandler(axis_axisEnum), simTimeStep_dt_sinv_float
    )

    if initialConditions is None:
        initialConditions = np.zeros(
            (plant_plant.plantAxisHandler(axis_axisEnum)[0].shape[0],)
        )

    if referenceCommand is None:
        referenceCommand = np.ones((simDiscreteSteps_N_U_int, 1))

    tdata = np.arange(0, simDiscreteSteps_N_U_int)
    xdata = np.zeros(
        (
            simDiscreteSteps_N_U_int,
            plant_plant.plantAxisHandler(axis_axisEnum)[0].shape[0],
        )
    )
    ydata = np.zeros(
        (
            simDiscreteSteps_N_U_int,
            plant_plant.plantAxisHandler(axis_axisEnum)[2].shape[0],
        )
    )
    udata = np.zeros(
        (
            simDiscreteSteps_N_U_int,
            plant_plant.plantAxisHandler(axis_axisEnum)[1].shape[1],
        )
    )

    xdata[0, :] = initialConditions

    # DEBUG
    import humanLQI

    Qstar, Rstar = humanLQI.lqiWithLogRandomSearch(
        plant_plant,
        axis_axisEnum,
        overshootHardRejectPercent_float=0.01,
        riseTimeHardRejectSeconds_float=5,
        settlingTimeHardRejectSeconds_float=5,
    )

    controller, *_ = ct.dlqr(discretePlant[0], discretePlant[1], Qstar, Rstar)
    # DEBUG

    for step in range(simDiscreteSteps_N_U_int):
        simStep(
            discretePlant,
            controller,
            referenceCommand,
            xdata,
            ydata,
            udata,
            step,
            simDiscreteSteps_N_U_int,
        )

    # DEBUG
    from matplotlib import pyplot as plt

    u_tilde = controller @ xdata.T
    u = u_tilde + plant_plant.rotorHoverThrustPercent_fthov_float

    fig, ax = plt.subplots()
    ax.plot(tdata, ydata.T[1], "b-", linewidth=2)
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.5, label="Step Command")
    ax.set_title(f"Forced Response U = {referenceCommand[0].item()}")
    ax.set_xlabel("Sim Steps")
    ax.set_ylabel("Output")
    ax.legend()
    # ax.grid(True, alpha=0.4)
    ax.grid()
    # plt.show()

    nrows = int(np.ceil(np.sqrt(plant_plant.rotorCount_nr_int)))
    ncols = int(np.ceil(plant_plant.rotorCount_nr_int / nrows))

    # fig, axes = plt.subplots(
    #     nrows=int(np.ceil(np.sqrt(plant_plant.rotorCount_nr_int))),
    #     ncols=int(np.floor(np.sqrt(plant_plant.rotorCount_nr_int))),
    # )
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    axes = axes.flatten()
    for rotor in range(plant_plant.rotorCount_nr_int):
        # axes[rotor].plot(T, u[rotor, :], label=f"Rotor {rotor+1}", color=f"C{rotor}")
        axes[rotor].plot(tdata, u[rotor, :], color=f"C{rotor}")
        axes[rotor].axhline(
            plant_plant.rotorHoverThrustPercent_fthov_float,
            color="k",
            linestyle="--",
            alpha=0.5,
            label=f"Hover ({plant_plant.rotorHoverThrustPercent_fthov_float})",
        )
        axes[rotor].set_title(f"Rotor {rotor+1}")
        axes[rotor].legend()
        axes[rotor].grid()
    for j in range(rotor + 1, len(axes)):
        axes[j].axis("off")
    fig.supxlabel("Sim Steps")
    fig.supylabel("Rotor Input")
    fig.suptitle("Rotor Commands")
    plt.show()
