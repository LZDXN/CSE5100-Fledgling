# General sim code

import plant


class Sim:
    def __init__(self):
        # Sim global parameters
        self.simFreq_f_Hz_int = 1000  # TODO Make an arg
        self.simTimeStep_dt_sinv_float = 1 / self.simFreq_f_Hz_int
        self.simTime_T_s_float = 10  # TODO Make an arg
        self.simDiscreteSteps_N_U_int = int(
            self.simTime_T_s_float / self.simTimeStep_dt_sinv_float
        )

        # Phys(ics)
        self.physAccGravity_g_mps2_float = 9.81

        # Model
        axis = plant.VerticalAxis()

        # Controller

    def simStep(self, step=0):
        pass

    def simRun(self):
        for step in range(self.simDiscreteSteps_N_U_int):
            self.simStep(step)
