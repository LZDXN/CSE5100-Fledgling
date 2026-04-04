# Human-designed automated optimal control for ONE design point about hover
# linearization. Useful for baseline design metric comparison and to set expected
# performance for NN controller results

# Global libraries
import control as ct
import numpy as np
from matplotlib import pyplot as plt

# Local project libraries
import multiRotorPlant


def lqiWithLogRandomSearch(
    plant_plant: multiRotorPlant.multiRotor6DOFWithXYZPositionError_class,
    axis_axisEnum: multiRotorPlant.axisEnum_enumClass,
    numLQISamples_int: int = 1000,
    # TODO: Make each Q term an input?
    inLQILogSearchPowerLower_int: int = -2,
    inLQILogSearchPowerUpper_int: int = 2,
    # By default reject 1% overshoot, but make tailorable, pass 0 to pass all
    overshootHardRejectPercent_float: float = 0.01,
    riseTimeHardRejectSeconds_float: float = 5,
    settlingTimeHardRejectSeconds_float: float = 10,
):
    # Full plant is already in terms of Awiggle, so section out portion relevant from axis as open-loop system matrices
    (
        errorAugmentedStateMatrix_Awig_matrixFloat,
        errorAugmentedInputMatrix_Bwig_matrixFloat,
        _,
        errorReferenceVector_Ewig_vectorFloat,
    ) = plant_plant.plantAxisHandler(axis_axisEnum)

    # TODO: make input parameter
    lqiMetricTuningWeights_dictFloat = {
        "riseTime": 1,
        "settlingTime": 1,
        "overshoot": 1,
        "undershoot": 1,
        "normGainPenalty": 1,
    }

    bestLQIMetricScore_float = np.inf
    bestLQIQ_Qstar_matrixFloat = None
    bestLQIR_Rstar_matrixFloat = None

    for lqiRun in range(numLQISamples_int):
        # Base cost matrices, default penalty values to 1
        lqiStatePenalty_Q_matrixFloat_statesxstates = np.eye(
            errorAugmentedStateMatrix_Awig_matrixFloat.shape[0]
        )

        # errorStatePenalty_qqe_float
        lqiStatePenalty_Q_matrixFloat_statesxstates[0, 0] = 10 ** np.random.uniform(
            inLQILogSearchPowerLower_int, inLQILogSearchPowerUpper_int
        )
        # positionStatePenalty_qqx_float
        lqiStatePenalty_Q_matrixFloat_statesxstates[1, 1] = 10 ** np.random.uniform(
            inLQILogSearchPowerLower_int, inLQILogSearchPowerUpper_int
        )
        # velocityStatePenalty_qqxdot_float
        lqiStatePenalty_Q_matrixFloat_statesxstates[2, 2] = 10 ** np.random.uniform(
            inLQILogSearchPowerLower_int, inLQILogSearchPowerUpper_int
        )
        if (
            axis_axisEnum == multiRotorPlant.axisEnum_enumClass.PITCHLON
            or axis_axisEnum == multiRotorPlant.axisEnum_enumClass.ROLLLAT
        ):
            # poseAngleStatePenalty_qqp_float
            lqiStatePenalty_Q_matrixFloat_statesxstates[3, 3] = 10 ** np.random.uniform(
                inLQILogSearchPowerLower_int, inLQILogSearchPowerUpper_int
            )
            # poseRateStatePenalty_qqpdot_float
            lqiStatePenalty_Q_matrixFloat_statesxstates[4, 4] = 10 ** np.random.uniform(
                inLQILogSearchPowerLower_int, inLQILogSearchPowerUpper_int
            )

        lqiActionPenalty_R_matrixFloat_inputsxinputs = np.eye(
            plant_plant.rotorCount_nr_int
        ) * 10 ** np.random.uniform(
            inLQILogSearchPowerLower_int + 1, inLQILogSearchPowerUpper_int + 1
        )

        try:
            lqiGains_K_matrixFloat_3xinputs, *_ = ct.lqr(
                errorAugmentedStateMatrix_Awig_matrixFloat,
                errorAugmentedInputMatrix_Bwig_matrixFloat,
                lqiStatePenalty_Q_matrixFloat_statesxstates,
                lqiActionPenalty_R_matrixFloat_inputsxinputs,
            )
        except:
            continue

        # Skip absurdly large gains
        if np.linalg.norm(lqiGains_K_matrixFloat_3xinputs[0]) > 100:
            continue

        closedLoopStatesMatrix_Acl_matrixFloat = (
            errorAugmentedStateMatrix_Awig_matrixFloat
            - errorAugmentedInputMatrix_Bwig_matrixFloat
            @ lqiGains_K_matrixFloat_3xinputs
        )

        # Closed-loop stability check
        if not np.all(
            np.real(np.linalg.eigvals(closedLoopStatesMatrix_Acl_matrixFloat)) < 0
        ):
            continue

        closedLoopInputsMatrix_Bcl_matrixFloat = errorReferenceVector_Ewig_vectorFloat  # Reference is the only external input now

        closedLoopOutputsVector_Ccl_vectorFloat = np.zeros(
            (errorAugmentedStateMatrix_Awig_matrixFloat.shape[0], 1)
        ).squeeze()
        # Only care to observe position output of system # TODO?: can be modified for what states get fed to NN controller
        closedLoopOutputsVector_Ccl_vectorFloat[1] = 1

        closedLoopSystem_sysCL_ssSys = ct.ss(
            closedLoopStatesMatrix_Acl_matrixFloat,
            closedLoopInputsMatrix_Bcl_matrixFloat,
            closedLoopOutputsVector_Ccl_vectorFloat,
            0,
        )

        stepData_dict = ct.step_info(closedLoopSystem_sysCL_ssSys)

        # try:
        #     stepData_dict = ct.step_info(closedLoopSystem_sysCL_ssSys)
        # except:
        #     continue

        # Firm overshoot limits
        # Allow up to 2x allowable overshoot percent, but heavily penalize
        # OR...(currently using)
        # Hard performance limits, discard any system that can't meet it
        if (
            stepData_dict["Overshoot"] > overshootHardRejectPercent_float * 100
            or stepData_dict["RiseTime"] > riseTimeHardRejectSeconds_float
            or stepData_dict["SettlingTime"] > settlingTimeHardRejectSeconds_float
        ):
            continue

        # for gaink in range(len(lqiGains_K_matrixFloat_3xinputs)):
        #     # TODO: make in put parameter
        #     lqiMetricTuningWeights_dictFloat[f"K{gaink}"] = 1

        # Can preferential score if desired
        currentMetricScore_float = (
            stepData_dict["RiseTime"] * lqiMetricTuningWeights_dictFloat["riseTime"]
            + stepData_dict["SettlingTime"]
            * lqiMetricTuningWeights_dictFloat["settlingTime"]
            + stepData_dict["Overshoot"] * lqiMetricTuningWeights_dictFloat["overshoot"]
            + stepData_dict["Undershoot"]
            * lqiMetricTuningWeights_dictFloat["undershoot"]
            # # Gain penalty
            # + np.linalg.norm(abs(lqiGains_K_matrixFloat_3xinputs))
            # * lqiMetricTuningWeights_dictFloat["normGainPenalty"]
        )

        # for gaink in range(len(lqiGains_K_matrixFloat_3xinputs)):
        #     currentMetricScore_float += (
        #         lqiGains_K_matrixFloat_3xinputs[gaink]
        #         * lqiMetricTuningWeights_dictFloat[f"K{gaink}"]
        #     )

        if currentMetricScore_float < bestLQIMetricScore_float:
            bestLQIMetricScore_float = currentMetricScore_float
            bestLQIQ_Qstar_matrixFloat = lqiStatePenalty_Q_matrixFloat_statesxstates
            bestLQIR_Rstar_matrixFloat = lqiActionPenalty_R_matrixFloat_inputsxinputs

    if bestLQIMetricScore_float == np.inf:
        raise ValueError(
            "Unable to find acceptable gain solution to meet desired performance parameters with random search!"
        )

    lqiGainsFinal_Kstar_matrixFloat_3xinputs, *_ = ct.lqr(
        errorAugmentedStateMatrix_Awig_matrixFloat,
        errorAugmentedInputMatrix_Bwig_matrixFloat,
        bestLQIQ_Qstar_matrixFloat,
        bestLQIR_Rstar_matrixFloat,
    )

    closedLoopStatesMatrix_Acl_matrixFloat = (
        errorAugmentedStateMatrix_Awig_matrixFloat
        - errorAugmentedInputMatrix_Bwig_matrixFloat
        @ lqiGainsFinal_Kstar_matrixFloat_3xinputs
    )

    closedLoopInputsMatrix_Bcl_matrixFloat = errorReferenceVector_Ewig_vectorFloat  # Reference is the only external input now

    closedLoopOutputsVector_Ccl_vectorFloat = np.zeros(
        (errorAugmentedStateMatrix_Awig_matrixFloat.shape[0], 1)
    ).squeeze()
    closedLoopOutputsVector_Ccl_vectorFloat[1] = (
        1  # Only care to observe position output of system # TODO: can be modified for what states get fed to NN controller
    )

    # closedLoopSystem_sysCL_ssSys
    bestClosedLoopSystem_sysCLStar_sysCLCT = ct.ss(
        closedLoopStatesMatrix_Acl_matrixFloat,
        closedLoopInputsMatrix_Bcl_matrixFloat,
        closedLoopOutputsVector_Ccl_vectorFloat,
        0,
    )

    # return (
    #     ct.step_info(bestClosedLoopSystem_sysCLStar_sysCLCT),
    #     lqiGainsFinal_Kstar_matrixFloat_3xinputs,
    # )

    # DEBUG

    T = np.linspace(0, 10, 10000)
    U = 1 * np.ones_like(T)
    X0 = np.zeros(errorAugmentedStateMatrix_Awig_matrixFloat.shape[0])

    T, Y, X = ct.forced_response(
        bestClosedLoopSystem_sysCLStar_sysCLCT, T, U, X0, return_x=True
    )

    u_tilde = lqiGainsFinal_Kstar_matrixFloat_3xinputs @ X
    u = u_tilde + plant_plant.rotorHoverThrustPercent_fthov_float

    # T, Y = ct.step_response(bestClosedLoopSystem_sysCLStar_sysCLCT)
    stepInfo = ct.step_info(bestClosedLoopSystem_sysCLStar_sysCLCT)

    fig, ax = plt.subplots()
    ax.plot(T, Y, "b-", linewidth=2)
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.5, label="Step Command")
    ax.set_title(
        f"Forced Response U = {U[0]}\n"
        f"Tr={stepInfo['RiseTime']:.3f}s  "
        f"Ts={stepInfo['SettlingTime']:.3f}s  "
        f"OS={stepInfo['Overshoot']:.2f}%"
    )
    ax.set_xlabel("Time (s)")
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
        axes[rotor].plot(T, u[rotor, :], color=f"C{rotor}")
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
    fig.supxlabel("Time (s)")
    fig.supylabel("Rotor Input")
    fig.suptitle("Rotor Commands")
    plt.show()
    # DEBUG

    # # DEBUG GPT plotting code
    # import matplotlib.pyplot as plt

    # # ── 7b.  Optimal system: step / bode / pole-zero ─────────────────────────────
    # fig_sys, axes_sys = plt.subplots(2, 2, figsize=(14, 10))
    # fig_sys.suptitle(
    #     f"Optimal Closed-Loop System)",
    #     fontsize=14,
    #     fontweight="bold",
    # )

    # # Step response
    # T_step, y_step = ct.step_response(bestClosedLoopSystem_sysCLStar_sysCLCT)
    # axes_sys[0, 0].plot(T_step, y_step.flatten(), "b-", linewidth=2)
    # axes_sys[0, 0].axhline(1.0, color="k", linestyle="--", alpha=0.5, label="Setpoint")
    # info_opt = ct.step_info(bestClosedLoopSystem_sysCLStar_sysCLCT)
    # axes_sys[0, 0].set_title(
    #     f"Step Response\n"
    #     f"Tr={info_opt['RiseTime']:.3f}s  "
    #     f"Ts={info_opt['SettlingTime']:.3f}s  "
    #     f"OS={info_opt['Overshoot']:.2f}%"
    # )
    # axes_sys[0, 0].set_xlabel("Time (s)")
    # axes_sys[0, 0].set_ylabel("Output")
    # axes_sys[0, 0].legend()
    # axes_sys[0, 0].grid(True, alpha=0.4)

    # # Bode – magnitude
    # omega = np.logspace(-2, 3, 600)
    # mag, phase, omega_out = ct.bode(
    #     bestClosedLoopSystem_sysCLStar_sysCLCT, omega=omega, plot=False
    # )
    # mag_db = 20 * np.log10(mag.flatten())
    # phase_deg = np.degrees(phase.flatten())

    # axes_sys[0, 1].semilogx(omega_out, mag_db, "b-", linewidth=2)
    # axes_sys[0, 1].axhline(-3, color="k", linestyle=":", alpha=0.5, label="−3 dB")
    # axes_sys[0, 1].set_title("Bode – Magnitude")
    # axes_sys[0, 1].set_xlabel("Frequency (rad/s)")
    # axes_sys[0, 1].set_ylabel("Magnitude (dB)")
    # axes_sys[0, 1].legend(fontsize=9)
    # axes_sys[0, 1].grid(True, which="both", alpha=0.4)

    # # Bode – phase
    # axes_sys[1, 0].semilogx(omega_out, phase_deg, "r-", linewidth=2)
    # axes_sys[1, 0].axhline(-180, color="k", linestyle="--", alpha=0.5, label="−180°")
    # axes_sys[1, 0].set_title("Bode – Phase")
    # axes_sys[1, 0].set_xlabel("Frequency (rad/s)")
    # axes_sys[1, 0].set_ylabel("Phase (deg)")
    # axes_sys[1, 0].legend(fontsize=9)
    # axes_sys[1, 0].grid(True, which="both", alpha=0.4)

    # # Pole-zero map  (closed-loop poles only; no explicit zeros for ss(A_cl,B,C,0))
    # poles = np.linalg.eigvals(closedLoopStatesMatrix_Acl_matrixFloat)
    # axes_sys[1, 1].axvline(0, color="k", linewidth=0.8, alpha=0.4)
    # axes_sys[1, 1].axhline(0, color="k", linewidth=0.8, alpha=0.4)
    # axes_sys[1, 1].scatter(
    #     np.real(poles),
    #     np.imag(poles),
    #     marker="x",
    #     s=120,
    #     linewidths=2,
    #     color="red",
    #     zorder=5,
    #     label="Poles",
    # )
    # for p in poles:
    #     axes_sys[1, 1].annotate(
    #         f"({p.real:.2f}{p.imag:+.2f}j)",
    #         xy=(p.real, p.imag),
    #         xytext=(6, 6),
    #         textcoords="offset points",
    #         fontsize=7,
    #     )
    # axes_sys[1, 1].set_title("Pole-Zero Map")
    # axes_sys[1, 1].set_xlabel("Real")
    # axes_sys[1, 1].set_ylabel("Imaginary")
    # axes_sys[1, 1].legend()
    # axes_sys[1, 1].grid(True, alpha=0.4)

    # plt.tight_layout()
    # # plt.savefig("lqr_optimal_system.png", dpi=150, bbox_inches="tight")
    # # print("Saved: lqr_optimal_system.png")

    # plt.show()
    # # DEBUG
