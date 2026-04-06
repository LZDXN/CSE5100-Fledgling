# Generalized multirotor plant model for arbitrary number of rotors linearized about
# hover with small-angle theorem coupling for lon/lat movement and init()-customizable
# parameters

# Global libraries
import control as ct
import numpy as np
from enum import Enum

# Local project libraries


class axisEnum_enumClass(Enum):
    PITCHLON = 0
    ROLLLAT = 1
    VERT = 2
    YAWHDG = 3


class multiRotor6DOFWithXYZPositionError_class:
    def __init__(
        self,
        envLocalGravity_g_float=9.81,
        vehicleMass_mv_float=1,
        payloadMass_mp_float=0,
        # vehicleCGOffsetLeverArm_dCG_matrixFloat_3x1=np.array([0, 0, 0], dtype=float),
        vehicleInertiaMoments_I_matrixFloat_3x3=np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=float,
        ),
        rotorCount_nr_int=4,
        rotorLeverArmRadiusToCG_drr_float=1,  # Assuming all rotors are equidistant, in plane with the main body, and fixed orientation with the main body for simplicity, TODO: Add rotor offset and orientation effects?
        # rotorOffsetAngles_aRA_matrixFloat_2x1=np.array([0, 0], dtype=float),
        # rotorOrientationRelToBody_oRO_matrixFloat_2x1=np.array([0, 0], dtype=float),
        # TODO: Add rotor spin ordering as definable parameter? Otherwise it's just assumed to alternate clockwise around the vehicle
        # rotorThrustCoefficient_kf_float=1,
        # rotorTorqueCoefficient_km_float=1,
        rotorAngularInertiaToThrustRatio_ktau_float=1,
        # Arbitrary parameters that normalize thrust to betweek 0 and 1 and
        # where hover is achieved at 50% thrust per rotor per default mass
        rotorMinThrust_ftmin_float=0,
        rotorMaxThrust_ftmax_float=1,
        rotorHoverThrustPercent_fthov_float=1 / 2,
    ):
        # General assumptions for this implementation:
        #  - No rotor dynamics, aerodynamics, or gyroscopic effects
        #  - No aerodynamic effects, and assumed rigid body motion
        #  - Small angle coupling and linearized about a hover
        #  - Inputs are directly commanded forces to induce accelerations->velocity->
        #       position change with intent to penalize error state in LQR

        # Vehicle
        self.vehicleMass_mv_float = vehicleMass_mv_float + payloadMass_mp_float
        self.vehicleInertiaMoments_I_matrixFloat_3x3 = (
            vehicleInertiaMoments_I_matrixFloat_3x3
        )

        # Rotor
        self.rotorCount_nr_int = rotorCount_nr_int
        # Assuming equal spaced rotors
        self.rotorEqualSpreadAngle_rphi_float = 360 / self.rotorCount_nr_int
        # Assuming "X" config for 4 rotors, and for odd number of pairs prefer extra
        # rotors along longitudinal axis for more "forward" authority
        if self.rotorCount_nr_int / 2 % 2 == 0:
            self.rotorFirstRotorAngleOffsetFromPlusX = (
                self.rotorEqualSpreadAngle_rphi_float / 2
            )
        else:
            self.rotorFirstRotorAngleOffsetFromPlusX = 0
        self.rotorLeverArmRadiusToCG_drr_float = rotorLeverArmRadiusToCG_drr_float

        # Build rotor information matrix in cartesian offsets in body frame in format:
        # [[X
        #   PHI
        #   Y
        #   THETA
        #   Z
        #   PSI
        #   Spin direction]xnrotors]
        # starting clockwise with rotor 0. Set up to handle rotors out of plane and
        # with orientations different from in-plane with the body, otherwise hard-
        # coded to 0s for now
        self.rotorInformationMatrix_matrixFloat = []
        for rotorNumber in range(self.rotorCount_nr_int):
            self.xOffset_float = self.rotorLeverArmRadiusToCG_drr_float * np.cos(
                (
                    self.rotorFirstRotorAngleOffsetFromPlusX
                    + rotorNumber * self.rotorEqualSpreadAngle_rphi_float
                )
                * np.pi
                / 180
            )
            self.yOffset_float = self.rotorLeverArmRadiusToCG_drr_float * np.sin(
                (
                    self.rotorFirstRotorAngleOffsetFromPlusX
                    + rotorNumber * self.rotorEqualSpreadAngle_rphi_float
                )
                * np.pi
                / 180
            )
            self.zOffset_float = -self.rotorLeverArmRadiusToCG_drr_float * np.sin(
                0 * np.pi / 180
            )  # Rotors in-plane with body
            self.rotorInformationMatrix_matrixFloat.append(
                [
                    self.xOffset_float,
                    0,  # Rotor roll relto body plane
                    self.yOffset_float,
                    0,  # Rotor pitch relto body plane
                    self.zOffset_float,
                    # self.rotorFirstRotorAngleOffsetFromPlusX
                    # + rotorNumber * self.rotorEqualSpreadAngle_rphi_float,
                    0,
                    -(
                        (-1) ** rotorNumber
                    ),  # Rotor 0 CCW for addressing even rotors for positive yaw
                ]
            )
        self.rotorInformationMatrix_matrixFloat = np.column_stack(
            self.rotorInformationMatrix_matrixFloat
        )

        # self.rotorThrustCoefficient_kf_float = rotorThrustCoefficient_kf_float
        # self.rotorTorqueCoefficient_km_float = rotorTorqueCoefficient_km_float
        self.rotorAngularInertiaToThrustRatio_ktau_float = (
            rotorAngularInertiaToThrustRatio_ktau_float
        )
        self.rotorMinThrust_ftmin_float = rotorMinThrust_ftmin_float
        self.rotorMaxThrust_ftmax_float = rotorMaxThrust_ftmax_float
        self.rotorHoverThrustPercent_fthov_float = rotorHoverThrustPercent_fthov_float

        self.ssStatesPositionFacingErrorAugmented_Apwig_matrixFloat_16x16 = np.zeros(
            shape=(16, 16), dtype=float
        )
        self.pitchLonIdxs_slice = range(0, 4 + 1)
        self.rollLatIdxs_slice = range(5, 9 + 1)
        self.vertIdxs_slice = range(10, 12 + 1)
        self.yawIdxs_slice = range(13, 15 + 1)

        # Position command tracking errors
        self.ssStatesPositionFacingErrorAugmented_Apwig_matrixFloat_16x16[
            self.pitchLonIdxs_slice[0], self.pitchLonIdxs_slice[1]
        ] = self.ssStatesPositionFacingErrorAugmented_Apwig_matrixFloat_16x16[
            self.rollLatIdxs_slice[0], self.rollLatIdxs_slice[1]
        ] = self.ssStatesPositionFacingErrorAugmented_Apwig_matrixFloat_16x16[
            self.vertIdxs_slice[0], self.vertIdxs_slice[1]
        ] = self.ssStatesPositionFacingErrorAugmented_Apwig_matrixFloat_16x16[
            self.yawIdxs_slice[0], self.yawIdxs_slice[1]
        ] = -1

        # Position Integrators
        self.ssStatesPositionFacingErrorAugmented_Apwig_matrixFloat_16x16[
            self.pitchLonIdxs_slice[1], self.pitchLonIdxs_slice[2]
        ] = self.ssStatesPositionFacingErrorAugmented_Apwig_matrixFloat_16x16[
            self.rollLatIdxs_slice[1], self.rollLatIdxs_slice[2]
        ] = self.ssStatesPositionFacingErrorAugmented_Apwig_matrixFloat_16x16[
            self.vertIdxs_slice[1], self.vertIdxs_slice[2]
        ] = self.ssStatesPositionFacingErrorAugmented_Apwig_matrixFloat_16x16[
            self.yawIdxs_slice[1], self.yawIdxs_slice[2]
        ] = 1

        # Small-angle coupling
        self.ssStatesPositionFacingErrorAugmented_Apwig_matrixFloat_16x16[
            self.pitchLonIdxs_slice[2], self.pitchLonIdxs_slice[3]
        ] = self.ssStatesPositionFacingErrorAugmented_Apwig_matrixFloat_16x16[
            self.rollLatIdxs_slice[2], self.rollLatIdxs_slice[3]
        ] = envLocalGravity_g_float
        # self.ssStatesPositionFacingErrorAugmented_Apwig_matrixFloat_16x16[
        #     self.pitchLonIdxs_slice[2], self.pitchLonIdxs_slice[3]
        # ] = envLocalGravity_g_float
        # self.ssStatesPositionFacingErrorAugmented_Apwig_matrixFloat_16x16[
        #     self.rollLatIdxs_slice[2], self.rollLatIdxs_slice[3]
        # ] = -envLocalGravity_g_float

        # Pose Integrators
        self.ssStatesPositionFacingErrorAugmented_Apwig_matrixFloat_16x16[
            self.pitchLonIdxs_slice[3], self.pitchLonIdxs_slice[4]
        ] = self.ssStatesPositionFacingErrorAugmented_Apwig_matrixFloat_16x16[
            self.rollLatIdxs_slice[3], self.rollLatIdxs_slice[4]
        ] = 1

        self.ssInputsPositionFacingErrorAugmented_Bpwig_matrixFloat_16xnrotors = (
            np.zeros(shape=(16, self.rotorCount_nr_int), dtype=float)
        )
        # Using a decomposed virtual control scheme for decoupled training in the format:
        # [[pitching torque
        #   rolling torque
        #   thrust
        #   yawing torque]xnrotors]
        # starting from rotor 0 clockwise to match format of state matrix and rotor
        # information matrix

        # # Thrust (in units of force)
        # self.ssInputsPositionFacingErrorAugmented_Bpwig_matrixFloat_16xnrotors[
        #     12, :
        # ] = -(1 / self.vehicleMass_mv_float)

        # Thrust (in unitless ratio of hover thrust component per motor count)
        self.ssInputsPositionFacingErrorAugmented_Bpwig_matrixFloat_16xnrotors[
            12, :
        ] = -(
            envLocalGravity_g_float
            / (self.rotorCount_nr_int * rotorHoverThrustPercent_fthov_float)
        )

        # # Pitching moment (in units of force)
        # self.ssInputsPositionFacingErrorAugmented_Bpwig_matrixFloat_16xnrotors[4, :] = (
        #     self.rotorInformationMatrix_matrixFloat[0, :]  # x lever arm from CG
        #     / self.vehicleInertiaMoments_I_matrixFloat_3x3[1, 1]  # iyy
        # )

        # Pitching moment (in unitless ratio of hover thrust component per motor count)
        self.ssInputsPositionFacingErrorAugmented_Bpwig_matrixFloat_16xnrotors[4, :] = (
            self.vehicleMass_mv_float
            * envLocalGravity_g_float
            * self.rotorInformationMatrix_matrixFloat[0, :]  # x lever arm from CG
        ) / (
            self.rotorCount_nr_int
            * rotorHoverThrustPercent_fthov_float
            * self.vehicleInertiaMoments_I_matrixFloat_3x3[1, 1]  # iyy
        )

        # # Rolling moment (in units of force)
        # self.ssInputsPositionFacingErrorAugmented_Bpwig_matrixFloat_16xnrotors[9, :] = (
        #     -(
        #         self.rotorInformationMatrix_matrixFloat[2, :]  # y lever arm from CG
        #         / self.vehicleInertiaMoments_I_matrixFloat_3x3[0, 0]  # ixx
        #     )
        # )

        # Rolling moment (in unitless ratio of hover thrust component per motor count)
        self.ssInputsPositionFacingErrorAugmented_Bpwig_matrixFloat_16xnrotors[9, :] = (
            self.vehicleMass_mv_float
            * envLocalGravity_g_float
            * self.rotorInformationMatrix_matrixFloat[2, :]  # x lever arm from CG
        ) / (
            self.rotorCount_nr_int
            * rotorHoverThrustPercent_fthov_float
            * self.vehicleInertiaMoments_I_matrixFloat_3x3[0, 0]  # iyy
        )

        # # Yawing moment (in units of force)
        # self.ssInputsPositionFacingErrorAugmented_Bpwig_matrixFloat_16xnrotors[
        #     15, :
        # ] = (
        #     -self.rotorInformationMatrix_matrixFloat[  # plus CCW motors to positive yaw motion
        #         -1, :
        #     ]
        #     * self.rotorAngularInertiaToThrustRatio_ktau_float  # Because we're not counting rotor inertial dynamics directly due to only commanding forces, creating a ratio to reduce rotor effectiveness for yaw commands
        #     / (self.vehicleInertiaMoments_I_matrixFloat_3x3[2, 2])  # izz
        # )

        # Yawing moment (in unitless ratio of hover thrust component per motor count)
        self.ssInputsPositionFacingErrorAugmented_Bpwig_matrixFloat_16xnrotors[
            15, :
        ] = (
            self.rotorInformationMatrix_matrixFloat[  # plus CCW motors to positive yaw motion
                -1, :
            ]
            * self.vehicleMass_mv_float
            * envLocalGravity_g_float
            * self.rotorAngularInertiaToThrustRatio_ktau_float
        ) / (  # Because we're not counting rotor inertial dynamics directly due to only commanding forces, creating a ratio to reduce rotor effectiveness for yaw commands
            self.rotorCount_nr_int
            * rotorHoverThrustPercent_fthov_float
            * self.vehicleInertiaMoments_I_matrixFloat_3x3[2, 2]
        )  # izz

        # Full state truth visibility sufficient for academic purposes
        self.ssOutputsPositionFacingErrorAugmented_Cpwig_matrixFloat_16x16 = np.eye(16)
        # self.ssOutputsPositionFacingErrorAugmented_Cpwig_matrixFloat_16x16 = np.ones(16)

        # No D matrix

        # DEBUG
        print(
            ct.ss(
                self.ssStatesPositionFacingErrorAugmented_Apwig_matrixFloat_16x16,
                self.ssInputsPositionFacingErrorAugmented_Bpwig_matrixFloat_16xnrotors,
                self.ssOutputsPositionFacingErrorAugmented_Cpwig_matrixFloat_16x16,
                0,
            )
        )
        # DEBUG

    def plantAxisHandler(
        self, axis: axisEnum_enumClass, obsIdxs: np.array = None
#         self, axis: axisEnum_enumClass.name, obsIdxs: np.array = None
    ) -> tuple:
        match axis:
            case axisEnum_enumClass.PITCHLON:
                axisIdxs_listInt = list(self.pitchLonIdxs_slice)
            case axisEnum_enumClass.ROLLLAT:
                axisIdxs_listInt = list(self.rollLatIdxs_slice)
            case axisEnum_enumClass.VERT:
                axisIdxs_listInt = list(self.vertIdxs_slice)
            case axisEnum_enumClass.YAWHDG:
                axisIdxs_listInt = list(self.yawIdxs_slice)
            case _:
                raise ValueError(
                    "Invalid 6DOF Axis! Options are PITCHLON or 0, ROLLLAT or 1, VERT or 2, or YAWHDG or 3"
                )

        errorAugmentedStateMatrix_Awig_matrixFloat = (
            self.ssStatesPositionFacingErrorAugmented_Apwig_matrixFloat_16x16[
                np.ix_(axisIdxs_listInt, axisIdxs_listInt)
            ]
        )

        errorAugmentedInputMatrix_Bwig_matrixFloat = (
            self.ssInputsPositionFacingErrorAugmented_Bpwig_matrixFloat_16xnrotors[
                np.ix_(axisIdxs_listInt, range(self.rotorCount_nr_int))
            ]
        )

        # Allow tuning of C outputs when setting up plantwig object for other functions
        if obsIdxs is None:
            errorAugmentedOutputVector_Cwig_vectorfloat = (
                self.ssOutputsPositionFacingErrorAugmented_Cpwig_matrixFloat_16x16[
                    np.ix_(axisIdxs_listInt, axisIdxs_listInt)
                ]
            )
        else:
            errorAugmentedOutputVector_Cwig_vectorfloat = obsIdxs

        # No D for now

        # Error refernce input in first term for position error
        errorReferenceVector_Ewig_vectorFloat = np.zeros(
            (errorAugmentedStateMatrix_Awig_matrixFloat.shape[0], 1)
        )
        errorReferenceVector_Ewig_vectorFloat[0] = 1

        return (
            errorAugmentedStateMatrix_Awig_matrixFloat,
            errorAugmentedInputMatrix_Bwig_matrixFloat,
            errorAugmentedOutputVector_Cwig_vectorfloat,
            errorReferenceVector_Ewig_vectorFloat,
        )

    def discreteLQIPlant(self, plantMatrices: tuple, dt: int) -> tuple:
        Awig, Bwig, Cwig, Ewig = plantMatrices

        BECombinedForDiscrete = np.hstack([Bwig, Ewig])

        combinedCTSys = ct.ss(Awig, BECombinedForDiscrete, Cwig, 0)
        combinedDTsys = ct.c2d(combinedCTSys, dt)

        return (
            combinedDTsys.A,
            combinedDTsys.B[:, : self.rotorCount_nr_int],
            combinedDTsys.C,
            # No D
            combinedDTsys.B[:, self.rotorCount_nr_int :],  # E
        )
