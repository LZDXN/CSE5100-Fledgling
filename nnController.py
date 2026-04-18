# File to handle the neural network and related functions

# Global libraries
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

# Local project libraries


def _orthogonalInit(layer, scale=1.0):
    # Orthogonal weight init: W drawn from the orthonormal basis of a random matrix,
    # scaled by `scale`.  Helps preserve gradient norms through deep networks.
    #   W ~ Orth(fan_in, fan_out) * scale,  b = 0
    nn.init.orthogonal_(layer.weight, gain=scale)
    nn.init.zeros_(layer.bias)


class ActorMLP(nn.Module):
    # Gaussian policy  π_θ(a | s) = N(μ_θ(s), σ²)
    #
    # Architecture:
    #   s ∈ R^obs_dim  →  FC(64) → Tanh → FC(64) → Tanh → FC(1)  →  μ_θ(s)
    #
    # σ is a single learnable scalar (state-independent):
    #   σ = exp(log_std),  log_std ∈ R  (clamped to [-4, 2])
    #
    # Action sampling during rollout:
    #   a ~ N(μ_θ(s), σ²),  then clipped to [-1, 1] before the plant
    #
    # log-probability of action a:
    #   log π(a|s) = -0.5 * ((a - μ)/σ)² - log σ - 0.5*log(2π)

    def __init__(self, obs_dim=3, action_dim=1, hidden=64):
        super().__init__()

        # μ_θ(s): unbounded mean, no final activation
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )
        # Hidden layers: scale = sqrt(2) recommended for Tanh activations
        # Output layer: scale = 0.01 → tiny initial mean → near-zero actions at init
        _orthogonalInit(self.net[0], scale=np.sqrt(2))
        _orthogonalInit(self.net[2], scale=np.sqrt(2))
        _orthogonalInit(self.net[4], scale=0.01)

        # log_std = -0.693  →  σ = exp(-0.693) ≈ 0.5  (moderate initial exploration)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.693))

    def _dist(self, obs):
        # Build N(μ_θ(s), σ²) for a batch of observations.
        # σ = exp(clamp(log_std, -4, 2))  →  σ ∈ [e^-4, e^2] ≈ [0.018, 7.39]
        mean = self.net(obs)
        std = self.log_std.clamp(-4, 2).exp().expand_as(mean)
        return Normal(mean, std)

    def forward(self, obs):
        # Returns (μ_θ(s), log_std) -- used by the optimizer, not directly by the sim loop
        dist = self._dist(obs)
        return dist.mean, self.log_std.clamp(-4, 2)

    def getAction(self, obs):
        # Stochastic rollout action:  a ~ N(μ_θ(s), σ²),  clipped to [-1, 1]
        # Returns (a, log π(a|s)) for storage in the rollout buffer.
        obs_t = torch.as_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            dist = self._dist(obs_t)
            action = dist.sample()
            # log π(a|s) = Σ_i log N(a_i; μ_i, σ_i)  (sum over action dims, here dim=1)
            log_prob = dist.log_prob(action).sum(-1)
        return action.squeeze(0).clamp(-1, 1).numpy(), float(log_prob)

    def getActionDeterministic(self, obs):
        # Deterministic evaluation action:  a = μ_θ(s),  no sampling
        # Used at eval time to measure true policy performance without noise.
        obs_t = torch.as_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            mean = self.net(obs_t)
        return mean.squeeze(0).clamp(-1, 1).numpy()

    def evaluateActions(self, obs_t, actions_t):
        # Re-evaluate stored (obs, action) pairs under the *current* policy θ.
        # Needed by PPO update to compute the probability ratio r_t(θ).
        #
        #   log π_θ(a|s)  -- new log-probs for the importance ratio
        #   H[π_θ(·|s)]   -- entropy bonus to encourage exploration
        #     H = 0.5 * log(2π e σ²) for a Gaussian
        dist = self._dist(obs_t)
        return dist.log_prob(actions_t).sum(-1), dist.entropy().sum(-1)


class CriticMLP(nn.Module):
    # State-value function  V_φ(s) ∈ R
    #
    # Architecture:
    #   s ∈ R^obs_dim  →  FC(64) → Tanh → FC(64) → Tanh → FC(1)  →  V_φ(s)
    #
    # V_φ(s) estimates the expected discounted return from state s:
    #   V_φ(s) ≈ E_π [ Σ_{k=0}^∞  γ^k r_{t+k} | s_t = s ]
    #
    # Used in GAE to compute advantages and as the critic target in the value loss.

    def __init__(self, obs_dim=3, hidden=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # Hidden: scale = sqrt(2) for Tanh; output: scale = 1.0 (standard value init)
        _orthogonalInit(self.net[0], scale=np.sqrt(2))
        _orthogonalInit(self.net[2], scale=np.sqrt(2))
        _orthogonalInit(self.net[4], scale=1.0)

    def forward(self, obs_t):
        # V_φ(s): raw scalar output, no activation (value can be any real number)
        return self.net(obs_t)

    def getValue(self, obs):
        # Single-observation inference.  Returns V_φ(s) as a Python float.
        obs_t = torch.as_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            v = self.net(obs_t)
        return float(v.squeeze())
