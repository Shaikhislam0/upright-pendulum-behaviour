"""Declared PPO variant: independently clip disjoint actor/critic gradients.

The train method follows Stable-Baselines3 2.9.0 PPO.train; the intentional
algorithmic change is separate clipping, with gradient diagnostics added.
Copyright (c) 2019 Antonin Raffin; MIT license in third_party/SB3_LICENSE.txt.
"""

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance


def parameter_groups(policy):
    actor = [*policy.mlp_extractor.policy_net.parameters(), *policy.action_net.parameters(), policy.log_std]
    critic = [*policy.mlp_extractor.value_net.parameters(), *policy.value_net.parameters()]
    actor_ids, critic_ids = {id(p) for p in actor}, {id(p) for p in critic}
    if actor_ids & critic_ids or actor_ids | critic_ids != {id(p) for p in policy.parameters()}:
        raise ValueError("independent clipping requires disjoint actor/critic networks and a parameter-free feature extractor")
    return actor, critic


def clip_independently(policy, maximum):
    actor, critic = parameter_groups(policy)
    actor_norm = float(torch.nn.utils.clip_grad_norm_(actor, maximum, error_if_nonfinite=True))
    critic_norm = float(torch.nn.utils.clip_grad_norm_(critic, maximum, error_if_nonfinite=True))
    return actor_norm, critic_norm


class IndependentGradientPPO(PPO):
    def train(self):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = None if self.clip_range_vf is None else self.clip_range_vf(self._current_progress_remaining)
        entropy_losses, pg_losses, value_losses, clip_fractions = [], [], [], []
        gradient_norms = []
        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for data in self.rollout_buffer.get(self.batch_size):
                actions = data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()
                values, log_prob, entropy = self.policy.evaluate_actions(data.observations, actions)
                values = values.flatten()
                advantages = data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                ratio = torch.exp(log_prob - data.old_log_prob)
                policy_loss = -torch.min(advantages * ratio,
                    advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)).mean()
                pg_losses.append(policy_loss.item())
                clip_fractions.append(torch.mean((torch.abs(ratio - 1) > clip_range).float()).item())
                values_pred = values if clip_range_vf is None else data.old_values + torch.clamp(
                    values - data.old_values, -clip_range_vf, clip_range_vf)
                value_loss = torch.nn.functional.mse_loss(data.returns, values_pred)
                value_losses.append(value_loss.item())
                entropy_loss = -torch.mean(-log_prob) if entropy is None else -torch.mean(entropy)
                entropy_losses.append(entropy_loss.item())
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                with torch.no_grad():
                    log_ratio = log_prob - data.old_log_prob
                    approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl)
                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    continue_training = False
                    break
                self.policy.optimizer.zero_grad()
                loss.backward()
                gradient_norms.append(clip_independently(self.policy, self.max_grad_norm))
                self.policy.optimizer.step()
            self._n_updates += 1
            if not continue_training:
                break
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        if gradient_norms:
            gradients = np.asarray(gradient_norms)
            self.logger.record("train/actor_grad_norm_before_clip", float(gradients[:, 0].mean()))
            self.logger.record("train/critic_grad_norm_before_clip", float(gradients[:, 1].mean()))
            self.logger.record("train/actor_gradient_retained", float(np.minimum(
                1., self.max_grad_norm / (gradients[:, 0] + 1e-6)).mean()))


def load_training_model(checkpoint, *, env, device="cpu", force_reset=False):
    import json
    import zipfile
    with zipfile.ZipFile(checkpoint) as archive:
        kind = json.loads(archive.read("data")).get("pendulum_gradient_clipping", "global")
    if kind not in ("global", "independent"):
        raise ValueError("unknown checkpoint gradient-clipping algorithm")
    algorithm = PPO if kind == "global" else IndependentGradientPPO
    return algorithm.load(checkpoint, env=env, device=device, force_reset=force_reset)
