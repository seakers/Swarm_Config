# rl/ppo.py
"""
PPO update logic (clipped objective) for the centralized policy.

Kept algorithm-agnostic to the specific network: it takes a policy exposing
.evaluate(obs, actions, mask) -> (log_prob, entropy, value) and an optimizer.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn


@dataclass
class PPOConfig:
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    minibatch_size: int = 256
    target_kl: float = 0.03    # early stop if exceeded (None to disable)


def ppo_update(policy, optimizer, batch: dict, cfg: PPOConfig) -> dict:
    """Run PPO epochs over a batch. Returns a dict of logged stats."""
    obs = batch["obs"]
    masks = batch["masks"]
    actions = batch["actions"]
    old_log_probs = batch["log_probs"]
    advantages = batch["advantages"]
    returns = batch["returns"]

    n = obs.shape[0]
    idx = np.arange(n)

    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "kl": [],
             "clip_frac": []}

    for _ in range(cfg.n_epochs):
        np.random.shuffle(idx)
        for start in range(0, n, cfg.minibatch_size):
            mb = idx[start: start + cfg.minibatch_size]
            mb_obs = obs[mb]
            mb_masks = masks[mb]
            mb_actions = actions[mb]
            mb_old_lp = old_log_probs[mb]
            mb_adv = advantages[mb]
            mb_ret = returns[mb]

            log_prob, entropy, value = policy.evaluate(
                mb_obs, mb_actions, mb_masks)

            ratio = torch.exp(log_prob - mb_old_lp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - cfg.clip_ratio,
                                1 + cfg.clip_ratio) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = 0.5 * (mb_ret - value).pow(2).mean()
            entropy_loss = -entropy.mean()

            loss = (policy_loss
                    + cfg.value_coef * value_loss
                    + cfg.entropy_coef * entropy_loss)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (mb_old_lp - log_prob).mean().item()
                clip_frac = ((ratio - 1.0).abs() > cfg.clip_ratio).float().mean().item()
            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy"].append(entropy.mean().item())
            stats["kl"].append(approx_kl)
            stats["clip_frac"].append(clip_frac)

        # Early stop the epoch loop if KL is too large.
        if cfg.target_kl is not None and np.mean(stats["kl"][-1:]) > cfg.target_kl:
            break

    return {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}