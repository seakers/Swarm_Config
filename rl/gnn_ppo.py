# rl/gnn_ppo.py
"""
PPO update for the GNN policy. Operates on collated-graph minibatches from
GNNRolloutBuffer. Same clipped objective as rl/ppo.py, but the policy call
signature is graph-aware.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from rl.ppo import PPOConfig


def gnn_ppo_update(policy, optimizer, buffer, cfg: PPOConfig,
                   device: str = "cpu") -> dict:
    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "kl": [],
             "clip_frac": []}

    t = lambda x, dt: torch.as_tensor(x, dtype=dt, device=device)

    for _ in range(cfg.n_epochs):
        epoch_kls = []
        for mb in buffer.minibatches(cfg.minibatch_size, normalize_adv=True):
            log_prob, entropy, value = policy.evaluate(
                t(mb["node_features"], torch.float32),
                t(mb["edge_index"], torch.int64),
                t(mb["edge_features"], torch.float32),
                t(mb["batch"], torch.int64),
                num_graphs=mb["num_graphs"],
                actions=t(mb["actions"], torch.int64),
                mask=t(mb["masks"], torch.float32),
            )
            old_lp = t(mb["old_log_probs"], torch.float32)
            adv = t(mb["advantages"], torch.float32)
            ret = t(mb["returns"], torch.float32)

            ratio = torch.exp(log_prob - old_lp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - cfg.clip_ratio,
                                1 + cfg.clip_ratio) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (ret - value).pow(2).mean()
            entropy_loss = -entropy.mean()

            loss = (policy_loss + cfg.value_coef * value_loss
                    + cfg.entropy_coef * entropy_loss)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (old_lp - log_prob).mean().item()
                clip_frac = ((ratio - 1.0).abs()
                             > cfg.clip_ratio).float().mean().item()
            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy"].append(entropy.mean().item())
            stats["kl"].append(approx_kl)
            stats["clip_frac"].append(clip_frac)
            epoch_kls.append(approx_kl)

        if cfg.target_kl is not None and np.mean(epoch_kls) > cfg.target_kl:
            break

    return {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}