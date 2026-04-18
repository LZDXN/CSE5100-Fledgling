# Math references:
#   Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
#   Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2015)

# Global libraries
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

# Local project libraries
import nnController


class RolloutBuffer:
    # Stores one batch of experience collected by the sim loop.
    # Call store() each step, then finish() once at the end of the rollout.

    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.logProbs = []
        self.values = []
        self.dones = []
        self.lastValue = 0.0
        self.ready = False

    def store(self, obs, action, reward, logProb, value, done):
        # Append one transition (s_t, a_t, r_t, log π_θ_old(a_t|s_t), V_φ(s_t), done_t)
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.logProbs.append(float(logProb))
        self.values.append(float(value))
        self.dones.append(bool(done))

    def finish(self, lastValue):
        # lastValue = V_φ(s_T):  bootstrap value for the last state.
        # Set to 0 if the episode terminated on step T (no future return).
        self.lastValue = float(lastValue)
        self.ready = True

    def meanReward(self):
        return float(np.mean(self.rewards)) if self.rewards else 0.0


class PPOTrainer:

    def __init__(
        self,
        actor,
        critic,
        # TODO: Write arug pass through and change these parameters in config file
        lr=3e-4,
        gamma=0.99,  # γ: discount factor for future rewards
        lam=0.95,  # λ: GAE bias-variance trade-off (0 = TD, 1 = MC)
        clipEps=0.2,  # ε: PPO-clip radius — limits how far π_θ moves from π_θ_old
        valueCoeff=0.5,  # c_1: weight of value loss in the combined objective
        entropyCoeff=0.01,  # c_2: weight of entropy bonus (encourages exploration)
        nEpochs=10,
        miniBatchSize=64,
    ):
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.lam = lam
        self.clipEps = clipEps
        self.valueCoeff = valueCoeff
        self.entropyCoeff = entropyCoeff
        self.nEpochs = nEpochs
        self.miniBatchSize = miniBatchSize

        self.actorOpt = Adam(actor.parameters(), lr=lr)
        self.criticOpt = Adam(critic.parameters(), lr=lr)

    def _computeGae(self, buffer):
        # ── Generalized Advantage Estimation (GAE-λ) ────────────────────────
        # s
        # TD residual (one-step Bellman error):
        #   δ_t = r_t + γ · V(s_{t+1}) · (1 - done_t) - V(s_t)
        #
        # GAE advantage (exponentially-weighted sum of TD residuals):
        #   Â_t = Σ_{l=0}^∞ (γλ)^l · δ_{t+l}
        #
        # Computed backward to reuse the recurrence:
        #   Â_t = δ_t + γλ · (1 - done_t) · Â_{t+1}
        #
        # Critic return target (used as regression label for V_φ):
        #   G_t = Â_t + V(s_t)         (advantage + baseline = full return)
        # ────────────────────────────────────────────────────────────────────
        rewards = np.array(buffer.rewards)
        values = np.array(buffer.values)
        dones = np.array(buffer.dones, dtype=float)
        T = len(rewards)

        advantages = np.zeros(T)
        lastAdv = 0.0

        for t in reversed(range(T)):
            # V(s_{t+1}): use stored value, or bootstrap from lastValue at boundary
            nextVal = buffer.lastValue if t == T - 1 else values[t + 1]
            nextVal *= 1.0 - dones[t]  # zero out if episode ended at t
            delta = rewards[t] + self.gamma * nextVal - values[t]
            lastAdv = delta + self.gamma * self.lam * (1.0 - dones[t]) * lastAdv
            advantages[t] = lastAdv

        # G_t = Â_t + V(s_t)
        returns = advantages + values
        return returns, advantages

    def update(self, buffer):
        assert buffer.ready, "Call buffer.finish() before update()"

        returns, advantages = self._computeGae(buffer)

        # ── Advantage normalization ──────────────────────────────────────────
        # Normalize Â over the full rollout before any mini-batch split.
        # Stabilizes gradient magnitude regardless of reward scale.
        #   Â_normalized = (Â - mean(Â)) / (std(Â) + ε)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t = torch.tensor(np.array(buffer.obs))
        actions_t = torch.tensor(np.array(buffer.actions))
        oldLp_t = torch.tensor(np.array(buffer.logProbs))
        returns_t = torch.tensor(returns)
        adv_t = torch.tensor(advantages)

        T = len(buffer.rewards)
        indices = np.arange(T)

        totalPLoss = totalVLoss = totalEnt = 0.0
        nUpdates = 0

        self.actor.train()
        self.critic.train()

        for _ in range(self.nEpochs):
            # Shuffle transitions to break temporal correlations between mini-batches
            np.random.shuffle(indices)
            for start in range(0, T, self.miniBatchSize):
                idx = indices[start : start + self.miniBatchSize]
                if len(idx) < 2:
                    continue

                mbObs = obs_t[idx]
                mbActions = actions_t[idx]
                mbOldLp = oldLp_t[idx]
                mbRet = returns_t[idx]
                mbAdv = adv_t[idx]

                # ── Probability ratio ────────────────────────────────────────
                # r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)
                #        = exp( log π_θ - log π_θ_old )
                newLp, entropy = self.actor.evaluateActions(mbObs, mbActions)
                ratio = (newLp - mbOldLp).exp()

                # ── PPO-Clip policy loss ─────────────────────────────────────
                # Surrogate objectives:
                #   L_CPI  = r_t(θ) · Â_t                         (unclipped)
                #   L_CLIP = clip(r_t(θ), 1-ε, 1+ε) · Â_t        (clipped)
                #
                # Combined (pessimistic):
                #   L^CLIP(θ) = -E[ min(L_CPI, L_CLIP) ]
                # The min prevents overly large policy updates in either direction.
                surr1 = ratio * mbAdv
                surr2 = ratio.clamp(1.0 - self.clipEps, 1.0 + self.clipEps) * mbAdv
                policyLoss = -torch.min(surr1, surr2).mean()

                # ── Value (critic) loss ──────────────────────────────────────
                # Mean-squared Bellman error:
                #   L^VF(φ) = 0.5 · E[ (V_φ(s_t) - G_t)² ]
                valuePred = self.critic(mbObs).squeeze(-1)
                valueLoss = 0.5 * ((valuePred - mbRet) ** 2).mean()

                # ── Combined PPO objective ───────────────────────────────────
                # L(θ, φ) = L^CLIP(θ) + c_1 · L^VF(φ) - c_2 · H[π_θ(·|s_t)]
                #
                # The entropy term H encourages exploration by penalizing
                # policies that collapse to near-deterministic distributions.
                loss = (
                    policyLoss
                    + self.valueCoeff * valueLoss
                    - self.entropyCoeff * entropy.mean()
                )

                self.actorOpt.zero_grad()
                self.criticOpt.zero_grad()
                loss.backward()

                # ── Gradient clipping ────────────────────────────────────────
                # Project gradient onto the L2 ball of radius 0.5:
                #   g ← g · min(1, 0.5 / ||g||)
                # Prevents catastrophic parameter updates from gradient spikes.
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

                self.actorOpt.step()
                self.criticOpt.step()

                totalPLoss += policyLoss.item()
                totalVLoss += valueLoss.item()
                totalEnt += entropy.mean().item()
                nUpdates += 1

        self.actor.eval()
        self.critic.eval()

        n = max(nUpdates, 1)
        return {
            "policyLoss": totalPLoss / n,
            "valueLoss": totalVLoss / n,
            "entropy": totalEnt / n,
            "meanAdvantage": float(advantages.mean()),
        }
