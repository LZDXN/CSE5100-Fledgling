# Disturbance / robustness injection for the multirotor plant rollouts.
#
# Three independently-toggleable disturbance channels and one rotor-failure mode
# implement the robustness ablations in Section 5.3 of the report:
#
#   1. State-side process noise   w_t ~ N(0, sigma_w^2 I)         per-step on x
#      sigma_w is sampled log-uniform per episode in [1e-3, 1e-1].
#      Inspired by the dynamics-randomization recipe of Peng et al. 2018
#      ("Sim-to-Real Transfer of Robotic Control with Dynamics Randomization").
#
#   2. Actuator-side noise         n_t ~ N(0, sigma_a^2 I)         per-step on u
#      sigma_a sampled log-uniform per episode in [5e-3, 5e-2].  Models
#      electronic-speed-controller jitter.  See Chen et al. 2025
#      ("SimpleFlight") which identifies action smoothing/regularization
#      as one of the most impactful sim-to-real design choices.
#
#   3. External sinusoidal force   F_d(t) = A_d sin(2 pi f_d t + phi)
#      with A_d in [0, 0.3 m g] and f_d in [0.1, 5] Hz, sampled per episode.
#      Models steady-state wind-gust forcing.  Follows the disturbance
#      framing of Wang et al. 2021 ("Deterministic policy gradient with
#      integral compensator for robust quadrotor control").
#
#   4. Single-rotor failure        with prob p_fail at episode start, one
#      rotor's commanded thrust is zeroed for the duration of the episode.
#      Follows Sharma et al. 2021 ("Reinforcement learning-based fault-
#      tolerant control of quadrotor under one rotor failure").
#
# Each channel can be toggled independently from the CLI; an all-zero config
# reduces to the original noise-free training/eval path.

# Global libraries
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class DisturbanceConfig:
    # Master switches for each channel.
    enableProcessNoise: bool = False
    enableActuatorNoise: bool = False
    enableForceDisturbance: bool = False
    rotorFailureProb: float = 0.0

    # Process noise sigma_w log-uniform sampling range.
    processNoiseSigmaLogLo: float = -3.0  # 10^-3
    processNoiseSigmaLogHi: float = -1.0  # 10^-1

    # Actuator noise sigma_a log-uniform sampling range.
    actuatorNoiseSigmaLogLo: float = np.log10(5e-3)
    actuatorNoiseSigmaLogHi: float = np.log10(5e-2)

    # External sinusoidal force amplitude bound, expressed as fraction of (m g).
    # f_d is sampled uniformly in [forceFreqLo, forceFreqHi] Hz.
    forceAmplFracMgLo: float = 0.0
    forceAmplFracMgHi: float = 0.3
    forceFreqLo: float = 0.1
    forceFreqHi: float = 5.0

    # Indices in the per-axis state vector at which the velocity / rate is
    # stored.  Translational axes (PITCHLON, ROLLLAT, VERT) all keep velocity
    # at index 2; YAWHDG keeps yaw rate at index 2.  The disturbance force
    # acts on the velocity index, scaled by 1/m * dt (Euler increment of an
    # external acceleration over one simulator step).
    velocityIdx: int = 2

    # Rotor index that has failed for this episode (-1 means no failure).
    # Set internally by sampleEpisode().
    failedRotorIdx: int = field(default=-1, init=False)

    @classmethod
    def fromArgs(cls, args, rotorFailureProb: Optional[float] = None):
        # Pull the channel flags from a CLI-style namespace; missing attrs
        # default to disabled.  Used by main.py and the experiment runners.
        return cls(
            enableProcessNoise=getattr(args, "enable_process_noise", False),
            enableActuatorNoise=getattr(args, "enable_actuator_noise", False),
            enableForceDisturbance=getattr(args, "enable_force_disturbance", False),
            rotorFailureProb=rotorFailureProb
            if rotorFailureProb is not None
            else getattr(args, "rotor_failure_prob", 0.0),
        )

    @property
    def isActive(self) -> bool:
        return (
            self.enableProcessNoise
            or self.enableActuatorNoise
            or self.enableForceDisturbance
            or self.rotorFailureProb > 0.0
        )


class DisturbanceInjector:
    # Per-episode disturbance state.  Hold the sampled sigma_w, sigma_a,
    # F_d amplitude/frequency/phase, and rotor-failure index for the current
    # episode; resampled on every sampleEpisode() call.

    def __init__(
        self,
        config: DisturbanceConfig,
        nRotors: int,
        vehicleMass: float = 1.0,
        gravity: float = 9.81,
        rng: Optional[np.random.Generator] = None,
    ):
        self.cfg = config
        self.nRotors = nRotors
        self.vehicleMass = vehicleMass
        self.gravity = gravity
        self.rng = rng if rng is not None else np.random.default_rng()

        # Per-episode random parameters; populated by sampleEpisode().
        self._sigmaW = 0.0
        self._sigmaA = 0.0
        self._forceAmpl = 0.0
        self._forceFreq = 0.0
        self._forcePhase = 0.0
        self._failedRotor = -1  # -1 == no failure

    # -- per-episode parameter resampling -----------------------------------
    def sampleEpisode(self):
        # Resample sigma_w log-uniform.
        if self.cfg.enableProcessNoise:
            uw = self.rng.uniform(
                self.cfg.processNoiseSigmaLogLo, self.cfg.processNoiseSigmaLogHi
            )
            self._sigmaW = float(10**uw)
        else:
            self._sigmaW = 0.0

        # Resample sigma_a log-uniform.
        if self.cfg.enableActuatorNoise:
            ua = self.rng.uniform(
                self.cfg.actuatorNoiseSigmaLogLo, self.cfg.actuatorNoiseSigmaLogHi
            )
            self._sigmaA = float(10**ua)
        else:
            self._sigmaA = 0.0

        # Resample F_d amplitude (linear, fraction of m g), freq (uniform Hz),
        # phase (uniform [0, 2 pi)).
        if self.cfg.enableForceDisturbance:
            fracMg = self.rng.uniform(
                self.cfg.forceAmplFracMgLo, self.cfg.forceAmplFracMgHi
            )
            self._forceAmpl = float(fracMg * self.vehicleMass * self.gravity)
            self._forceFreq = float(
                self.rng.uniform(self.cfg.forceFreqLo, self.cfg.forceFreqHi)
            )
            self._forcePhase = float(self.rng.uniform(0.0, 2 * np.pi))
        else:
            self._forceAmpl = 0.0
            self._forceFreq = 0.0
            self._forcePhase = 0.0

        # Sample rotor-failure index with probability rotorFailureProb.
        if (
            self.cfg.rotorFailureProb > 0.0
            and self.rng.uniform(0.0, 1.0) < self.cfg.rotorFailureProb
        ):
            self._failedRotor = int(self.rng.integers(0, self.nRotors))
        else:
            self._failedRotor = -1

    # -- per-step injection -------------------------------------------------
    def perturbAction(self, action: np.ndarray) -> np.ndarray:
        # Add Gaussian actuator noise and zero-out failed rotor's command.
        # Returns a new action array; the caller is responsible for the
        # subsequent thrust-percent affine map and clipping into [-1, 1].
        out = action.astype(np.float64, copy=True)
        if self._sigmaA > 0.0:
            out = out + self.rng.normal(0.0, self._sigmaA, size=out.shape)
        if self._failedRotor >= 0:
            # Drive the failed rotor toward its zero-thrust extreme
            # (action=-1 maps to absolute thrust 0 via the hover-affine map).
            out[self._failedRotor] = -1.0
        return out

    def stateNoise(self, stateShape: Tuple[int, ...]) -> np.ndarray:
        # Return additive process noise w_t for one timestep, shape == state.
        if self._sigmaW <= 0.0:
            return np.zeros(stateShape)
        return self.rng.normal(0.0, self._sigmaW, size=stateShape)

    def forceDelta(self, t: float, dt: float, stateShape: Tuple[int, ...]) -> np.ndarray:
        # Return state-vector increment from external sinusoidal force over
        # one simulator step.  Force is mapped to the velocity index via a
        # first-order Euler increment: dx[v_idx] += (F_d / m) * dt.
        if self._forceAmpl <= 0.0:
            return np.zeros(stateShape)
        delta = np.zeros(stateShape)
        velIdx = self.cfg.velocityIdx
        if velIdx < delta.shape[0]:
            f_t = self._forceAmpl * np.sin(2 * np.pi * self._forceFreq * t + self._forcePhase)
            delta[velIdx] = (f_t / self.vehicleMass) * dt
        return delta

    # -- introspection ------------------------------------------------------
    def episodeRecord(self) -> dict:
        # Snapshot of the current episode's sampled disturbance parameters,
        # for logging into per-run summary JSON.
        return {
            "sigmaW": self._sigmaW,
            "sigmaA": self._sigmaA,
            "forceAmpl": self._forceAmpl,
            "forceFreq": self._forceFreq,
            "forcePhase": self._forcePhase,
            "failedRotor": self._failedRotor,
        }
