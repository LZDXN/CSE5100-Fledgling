# Plant dynamics model code

import control as ct
import numpy as np


class SixDOFVehiclePlantModel:
    def __init__(self):
        # Vehicle
        self.vehicleBodyMass_m_kg_float = (
            4 / 9.81
        )  # Hypersimplified mass such that f = ma should yield a force thrust of 4 to hover, or in other words 1 per motor for a quadrotor # TODO Make an arg
        # self.vehicleBodyMomentX_jbx_kgm2_float = 1  # TODO Make an arg
        # self.vehicleBodyMomentY_jby_kgm2_float = 1  # TODO Make an arg
        # self.vehicleBodyMomentZ_jbz_kgm2_float = 1  # TODO Make an arg
        # Ignoring intertia cross-terms for simplicity
        # self.vehicleInertia_J_U_matrixFoat = np.array(
        #     [
        #         [self.vehicleBodyMomentX_jbx_kgm2_float, 0, 0],
        #         [0, self.vehicleBodyMomentY_jby_kgm2_float, 0],
        #         [0, 0, self.vehicleBodyMomentZ_jbz_kgm2_float],
        #     ]
        # )
        self.vehicleEZMomentArmLength_l_m_float = (
            1  # Hypersimplified moment arm length for rotor moments
        )

        # Rotors
        self.rotorMinThrust_ftmin_N_float = 0
        self.rotorMaxThrust_ftmax_N_float = 2  # Hypersimplified arbitrary parameter that hover is achieved at 50% thrust per rotor


class VerticalAxis(SixDOFVehiclePlantModel):
    def __init__(self):
        super().__init__()
        # States
        self.statesVerticalStatesErrorAugmented_Apwig_matrixFloat_3x3 = np.array(
            [[0, 0, -1], [0, 0, 1], [0, 0, 0]]
        )

        self.statesVerticalControlErrorAugmented_Bpwig_matrixFloat_3x4 = np.tile(
            np.array([0, 0, 1 / self.vehicleBodyMass_m_kg_float]), (1, 4)
        )  # TODO Make tile colum number an arg

        self.statesVerticalOutputErrorAugmented_Cpwig_matrixFloat_1x3 = np.array(
            [0, 1, 0]
        )

        self.statesVerticalFFErrorAugmented_Dpwig_matrixFloat = None

        # For traditional LQR
        self.statesVerticalControlErrorAugmentedVirtual_Bpwigvirt_matrixFloat_1x4 = (
            self.statesVerticalControlErrorAugmented_Bpwig_matrixFloat_3x4.sum(axis=1)
        )

        # Sys
        self.sysVerticalErrorAug_ssSys = ct.ss(
            self.statesVerticalStatesErrorAugmented_Apwig_matrixFloat_3x3,
            self.statesVerticalControlErrorAugmented_Bpwig_matrixFloat_3x4,
            self.statesVerticalOutputErrorAugmented_Cpwig_matrixFloat_1x3,
            0,
        )

        # For traditional LQR
        self.sysVerticalErrorAugVirtual_ssSys = ct.ss(
            self.statesVerticalStatesErrorAugmented_Apwig_matrixFloat_3x3,
            self.statesVerticalControlErrorAugmentedVirtual_Bpwigvirt_matrixFloat_1x4,
            self.statesVerticalOutputErrorAugmented_Cpwig_matrixFloat_1x3,
            0,
        )

        # Sysd
        self.sysdVerticalErrorAugDiscrete_ssSysD = ct.c2d(
            self.sysVerticalErrorAug_ssSys
        )
        self.sysdVerticalErrorAugDiscreteStates_Asysd_matrixFloat = np.array(
            self.sysdVerticalErrorAugDiscrete_ssSysD.A
        )
        self.sysdVerticalErrorAugDiscreteControls_Bsysd_matrixFloat = np.array(
            self.sysdVerticalErrorAugDiscrete_ssSysD.B
        )

        # For traditional LQR
        self.sysdVerticalErrorAugVirtualDiscrete_ssSysD = ct.c2d(
            self.sysVerticalErrorAugVirtual_ssSys
        )
        self.sysdVerticalErrorAugDiscreteStatesVirtual_Asysdv_matrixFloat = np.array(
            self.sysdVerticalErrorAugVirtualDiscrete_ssSysD.A
        )
        self.sysdVerticalErrorAugDiscreteControlsVirtual_Bsysdv_matrixFloat = np.array(
            self.sysdVerticalErrorAugVirtualDiscrete_ssSysD.B
        )
