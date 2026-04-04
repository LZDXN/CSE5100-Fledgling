# Human-designed automated optimal control for ONE design point about hover
# linearization. Useful for baseline design metric comparison and to set expected
# performance for NN controller results

# Global libraries
import control as ct
import numpy as np
from collections import defaultdict

# Local project libraries
import multiRotorPlant


def minMaxNormalize(arr: np.ndarray) -> np.ndarray:
    arr = np.abs(np.array(arr, dtype=float))
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if (hi - lo) < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def lqiWithLogRandomSearch(
    inPlant_plant: multiRotorPlant.multiRotor6DOFWithXYZPositionError_class,
    inAxis_axisEnum: multiRotorPlant.axisEnum_enumClass,
    inNumLQISamples_int: int = 1000,
    # TODO: Make each Q term an input?
    inLQILogSearchPowerLower_int: int = -2,
    inLQILogSearchPowerUpper_int: int = 2,
    # By default reject 1% overshoot, but make tailorable, pass 0 to pass all
    inOvershootHardRejectPercent_float: float = 0.01,
):
    # Full plant is already in terms of Awiggle, so section out portion relevant from axis as open-loop system matrices
    (
        errorAugmentedStateMatrix_Awig_matrixFloat,
        errorAugmentedInputMatrix_Bwig_matrixFloat,
        errorAugmentedOutputVector_Cwig_vectorfloat,
        errorReferenceVector_Ewig_vectorFloat,
    ) = inPlant_plant.plantAxisHandler(inAxis_axisEnum)
    errorReferenceVector_Ewig_vectorFloat[0] = 1

    # Base cost matrices, default penalty values to 1
    lqiStatePenalty_Q_matrixFloat_statesxstates = np.eye(
        errorAugmentedStateMatrix_Awig_matrixFloat.shape[0]
    )
    lqiActionPenalty_R_matrixFloat_inputsxinputs = np.eye(
        inPlant_plant.rotorCount_nr_int
    )

    lqiMetricData_dict = defaultdict(list)  # per-metric lists, keyed by metric name

    # TODO: make in put parameter
    lqiMetricTuningWeights_dictFloat = {
        "rise_time": 1,
        "settling_time": 1,
        "overshoot": 1,
        "undershoot": 1,
    }

    bestLQIMetricScore_float = np.inf
    bestLQIMetric_dictFloat = None

    for lqiRun in range(inNumLQISamples_int):
        # errorStatePenalty_qqe_float
        lqiStatePenalty_Q_matrixFloat_statesxstates[0, 0] = 10 ** np.random.uniform(
            inLQILogSearchPowerLower_int, inLQILogSearchPowerUpper_int
        )
        # positionStatePenalty_qqe_float
        lqiStatePenalty_Q_matrixFloat_statesxstates[1, 1] = 10 ** np.random.uniform(
            inLQILogSearchPowerLower_int, inLQILogSearchPowerUpper_int
        )
        # velocityStatePenalty_qqe_float
        lqiStatePenalty_Q_matrixFloat_statesxstates[2, 2] = 10 ** np.random.uniform(
            inLQILogSearchPowerLower_int, inLQILogSearchPowerUpper_int
        )
        if (
            inAxis_axisEnum == multiRotorPlant.axisEnum_enumClass.PITCHLON
            or inAxis_axisEnum == multiRotorPlant.axisEnum_enumClass.ROLLLAT
        ):
            # poseAngleStatePenalty_qqe_float
            lqiStatePenalty_Q_matrixFloat_statesxstates[3, 3] = 10 ** np.random.uniform(
                inLQILogSearchPowerLower_int, inLQILogSearchPowerUpper_int
            )
            # poseRateStatePenalty_qqe_float
            lqiStatePenalty_Q_matrixFloat_statesxstates[4, 4] = 10 ** np.random.uniform(
                inLQILogSearchPowerLower_int, inLQILogSearchPowerUpper_int
            )

        lqiActionPenalty_R_matrixFloat_inputsxinputs *= 10 ** np.random.uniform(
            inLQILogSearchPowerLower_int, inLQILogSearchPowerUpper_int
        )

        lqiGains_K_matrixFloat_3xinputs, *_ = ct.lqr(
            errorAugmentedStateMatrix_Awig_matrixFloat,
            errorAugmentedInputMatrix_Bwig_matrixFloat,
            lqiStatePenalty_Q_matrixFloat_statesxstates,
            lqiActionPenalty_R_matrixFloat_inputsxinputs,
        )

        closedLoopStatesMatrix_Acl_matrixFloat = (
            errorAugmentedStateMatrix_Awig_matrixFloat
            - errorAugmentedInputMatrix_Bwig_matrixFloat
            @ lqiGains_K_matrixFloat_3xinputs
        )

        eigValsOfAcl_vectorFloatReIm = np.linalg.eigvals(
            closedLoopStatesMatrix_Acl_matrixFloat
        )

        if not np.all(np.real(eigValsOfAcl_vectorFloatReIm) < 0):
            continue

        closedLoopInputsMatrix_Bcl_matrixFloat = errorReferenceVector_Ewig_vectorFloat  # Reference is the only external input now

        closedLoopOutputsVector_Ccl_vectorFloat = np.zeros(
            (errorAugmentedStateMatrix_Awig_matrixFloat.shape[0], 1)
        ).squeeze()
        closedLoopOutputsVector_Ccl_vectorFloat[1] = (
            1  # Only care to observe position output of system # TODOL can be modified for what states get fed to NN controller
        )

        closedLoopSystem_sysCL_ssSys = ct.ss(
            closedLoopStatesMatrix_Acl_matrixFloat,
            closedLoopInputsMatrix_Bcl_matrixFloat,
            closedLoopOutputsVector_Ccl_vectorFloat,
            0,
        )

        for gaink in range(len(lqiGains_K_matrixFloat_3xinputs)):
            # TODO: make in put parameter
            lqiMetricTuningWeights_dictFloat[f"K{gaink}"] = 1

        stepData_dict = ct.step_info(closedLoopSystem_sysCL_ssSys)

        currentMetricScore_float = (
            stepData_dict["RiseTime"] * lqiMetricData_dict["riseTime"]
        )
