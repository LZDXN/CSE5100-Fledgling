# Restricted-observability transforms for the partial-observation ablations.
#
# Implements the five observation regimes in Section 5.4 of the report:
#   (1) Full state              [e_z, z, z_dot]                baseline
#   (2) Inertial-only           [e_z,    z_dot]                hide position
#   (3) Position-only           [e_z, z       ]                hide velocity
#   (4) Noisy full state                                       additive Gaussian
#   (5) Delayed full state                                     past observation
#
# A single ObservationProcessor wraps the per-axis output matrix C and applies
# (a) state masking via a row-restricted C, (b) additive observation noise,
# and (c) a fixed observation delay.  Stacking all three is allowed.
#
# References:
#   - The partial-observability framing follows Peng et al. 2018, who note
#     that recurrent policies can perform implicit system identification
#     through their hidden state.  Frame stacking via `historyLen > 1` is
#     the standard MDP-recovery technique introduced in Mnih et al. 2015
#     ("Human-level control through deep reinforcement learning") for DQN
#     and used here as the simpler alternative to a full GRU policy.
#   - Eschmann et al. 2025 ("Raptor") explore recurrent foundation policies
#     for cross-platform multirotor control; the historyLen>1 path here is
#     the lightweight analog of that approach.

# Global libraries
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ObservabilityConfig:
    # Row indices to keep from the per-axis state vector when projecting to
    # the controller observation.  None == use the plant default (full state).
    # Examples for VERT axis [e_z, z, z_dot]:
    #   keepIdxs=[0, 1, 2] -> full state (regime 1)
    #   keepIdxs=[0, 2]    -> inertial-only (regime 2)
    #   keepIdxs=[0, 1]    -> position-only (regime 3)
    keepIdxs: Optional[List[int]] = None

    # Additive Gaussian observation noise standard deviation (regime 4).
    # 0.0 disables.  Applied independently to each component of the obs.
    obsNoiseSigma: float = 0.0

    # Observation delay in simulator steps (regime 5).  0 disables.  When > 0,
    # the controller sees the observation taken `delaySteps` steps ago, with
    # zero-padding before the buffer fills.
    delaySteps: int = 0

    # Frame-stack history length.  When > 1 the observation passed to the
    # controller is the concatenation of the last `historyLen` (post-mask /
    # noise / delay) observations, oldest-first.  This is the partial-obs
    # recovery mechanism described in Section 5.4 ("recoverable by replacing
    # the feedforward MLP with a [recurrent / windowed] policy").
    historyLen: int = 1

    @classmethod
    def fromArgs(cls, args):
        # Parse from CLI: --obs_keep_idxs "0,2"  --obs_noise_sigma 0.05  ...
        keepStr = getattr(args, "obs_keep_idxs", None)
        if keepStr is None or keepStr == "" or keepStr == "all":
            keep = None
        else:
            keep = [int(x) for x in keepStr.split(",") if x.strip() != ""]
        return cls(
            keepIdxs=keep,
            obsNoiseSigma=float(getattr(args, "obs_noise_sigma", 0.0)),
            delaySteps=int(getattr(args, "obs_delay_steps", 0)),
            historyLen=int(getattr(args, "obs_history_len", 1)),
        )

    @property
    def isActive(self) -> bool:
        return (
            self.keepIdxs is not None
            or self.obsNoiseSigma > 0.0
            or self.delaySteps > 0
            or self.historyLen > 1
        )


# ---------------------------------------------------------------------------
# C-matrix builder
# ---------------------------------------------------------------------------


def buildRestrictedC(fullStateDim: int, keepIdxs: Optional[List[int]]) -> np.ndarray:
    # Build a row-restricted C matrix that picks out only the given state
    # indices.  Returns an identity-shaped (k, fullStateDim) selector.
    # When keepIdxs is None, returns a full-rank identity (== full obs).
    if keepIdxs is None:
        return np.eye(fullStateDim)
    C = np.zeros((len(keepIdxs), fullStateDim), dtype=float)
    for row, col in enumerate(keepIdxs):
        C[row, col] = 1.0
    return C


# ---------------------------------------------------------------------------
# Per-rollout processor (delay buffer + history stacker + noise sampler)
# ---------------------------------------------------------------------------


class ObservationProcessor:
    # Per-rollout stateful processor.  Owns the delay buffer (of post-mask,
    # post-noise observations) and the history-stack buffer (of post-delay
    # observations as they are returned to the controller).
    #
    # Call reset() at episode start, process(rawObs) at each step.

    def __init__(
        self,
        config: ObservabilityConfig,
        baseObsDim: int,
        rng: Optional[np.random.Generator] = None,
    ):
        self.cfg = config
        self.baseObsDim = baseObsDim  # dim of obs after C masking
        self.rng = rng if rng is not None else np.random.default_rng()

        # Delay buffer always holds at least 1 entry so we can read "current"
        # uniformly via .popleft() / [0] indexing.
        self._delayBuf = deque(maxlen=max(self.cfg.delaySteps + 1, 1))
        self._historyBuf = deque(maxlen=max(self.cfg.historyLen, 1))

    @property
    def outputObsDim(self) -> int:
        # The dim seen by the actor: history-stacked masked obs.
        return self.baseObsDim * max(self.cfg.historyLen, 1)

    def reset(self):
        # Zero-pad both buffers so that early-episode observations don't
        # carry stale history from the previous episode.
        self._delayBuf.clear()
        self._historyBuf.clear()
        zero = np.zeros(self.baseObsDim, dtype=np.float32)
        for _ in range(self._delayBuf.maxlen):
            self._delayBuf.append(zero.copy())
        for _ in range(self._historyBuf.maxlen):
            self._historyBuf.append(zero.copy())

    def process(self, rawObs: np.ndarray) -> np.ndarray:
        # rawObs: post-C, pre-noise observation, shape == (baseObsDim,)
        # Returns: history-stacked, post-delay, post-noise obs, shape ==
        #          (baseObsDim * historyLen,).

        # Optional additive noise (regime 4).
        if self.cfg.obsNoiseSigma > 0.0:
            rawObs = rawObs + self.rng.normal(
                0.0, self.cfg.obsNoiseSigma, size=rawObs.shape
            ).astype(rawObs.dtype)

        # Push into delay buffer; the oldest entry is what the controller
        # actually sees.  When delaySteps == 0 this devolves to identity.
        self._delayBuf.append(rawObs.astype(np.float32, copy=True))
        delayedObs = self._delayBuf[0].copy()

        # Push into history buffer and return the concatenation.
        self._historyBuf.append(delayedObs)
        if self.cfg.historyLen <= 1:
            return delayedObs
        return np.concatenate(list(self._historyBuf)).astype(np.float32)


# ---------------------------------------------------------------------------
# Named regime presets
# ---------------------------------------------------------------------------


REGIME_PRESETS = {
    # Each preset is a dict of CLI overrides for the partial-observability
    # ablation.  Index lists are written for the VERT axis state vector
    # [e_z, z, z_dot]; the runner script picks the right preset per axis.
    "full": {
        "obs_keep_idxs": "all",
        "obs_noise_sigma": 0.0,
        "obs_delay_steps": 0,
        "obs_history_len": 1,
    },
    "inertial_only": {
        "obs_keep_idxs": "0,2",
        "obs_noise_sigma": 0.0,
        "obs_delay_steps": 0,
        "obs_history_len": 1,
    },
    "position_only": {
        "obs_keep_idxs": "0,1",
        "obs_noise_sigma": 0.0,
        "obs_delay_steps": 0,
        "obs_history_len": 1,
    },
    "noisy_full": {
        "obs_keep_idxs": "all",
        "obs_noise_sigma": 0.05,
        "obs_delay_steps": 0,
        "obs_history_len": 1,
    },
    "delayed_full": {
        "obs_keep_idxs": "all",
        "obs_noise_sigma": 0.0,
        "obs_delay_steps": 3,
        "obs_history_len": 1,
    },
    # Recovery regime: identical to inertial_only but with frame-stacking
    # to give the MLP the past observations needed to recover the dropped
    # state.  Mnih et al. 2015 frame-stacking, scoped to the partial-obs
    # ablation.
    "inertial_only_stacked": {
        "obs_keep_idxs": "0,2",
        "obs_noise_sigma": 0.0,
        "obs_delay_steps": 0,
        "obs_history_len": 4,
    },
    "delayed_full_stacked": {
        "obs_keep_idxs": "all",
        "obs_noise_sigma": 0.0,
        "obs_delay_steps": 3,
        "obs_history_len": 4,
    },
}
