# Plotting utilities for LQI baseline results

from matplotlib import pyplot as plt
import numpy as np
import os

# plt.rcParams["text.usetex"] = True
# plt.rcParams["figure.constrained_layout.use"] = True


def plotLqiDebug(T, Y, u, stepInfo, plant, saveDir="/tmp"):
    # Step response plot
    fig, ax = plt.subplots()
    ax.plot(T, Y, "b-", linewidth=2)
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.5, label="Step Command")
    ax.set_title(
        f"Forced Response U = 1\n"
        f"Tr={stepInfo['RiseTime']:.3f}s  "
        f"Ts={stepInfo['SettlingTime']:.3f}s  "
        f"OS={stepInfo['Overshoot']:.2f}%"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Output")
    ax.legend()
    ax.grid()
    fig.set_size_inches(11, 8.5)
    fig.savefig(os.path.join(saveDir, "lqi_step_response.png"), dpi=150)
    plt.close(fig)

    # Rotor commands plot
    nRotors = plant.rotorCount_nr_int
    nrows = int(np.ceil(np.sqrt(nRotors)))
    ncols = int(np.ceil(nRotors / nrows))

    fig2, axes = plt.subplots(nrows=nrows, ncols=ncols)
    axes = axes.flatten()
    for rotor in range(nRotors):
        axes[rotor].plot(T, u[rotor, :], color=f"C{rotor}")
        axes[rotor].axhline(
            plant.rotorHoverThrustPercent_fthov_float,
            color="k",
            linestyle="--",
            alpha=0.5,
            label=f"Hover ({plant.rotorHoverThrustPercent_fthov_float})",
        )
        axes[rotor].set_title(f"Rotor {rotor + 1}")
        axes[rotor].legend()
        axes[rotor].grid()
    for j in range(nRotors, len(axes)):
        axes[j].axis("off")
    fig2.supxlabel("Time (s)")
    fig2.supylabel("Rotor Input")
    fig2.suptitle("Rotor Commands")
    fig2.set_size_inches(11, 8.5)
    fig2.savefig(os.path.join(saveDir, "lqi_rotor_commands.png"), dpi=150)
    plt.close(fig2)
