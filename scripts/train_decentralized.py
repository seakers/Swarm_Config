# scripts/train_decentralized.py
"""
Train a decentralized GNN actor via CTDE (centralized critic during training,
local actor at execution).

The actor sees only local observations and uses FEW message-passing layers
(bounded communication radius). The critic sees the global graph with MORE
layers. At execution the critic is discarded.

Usage:
    python -m scripts.train_decentralized --n 4 6 --missions power \
        --actor-layers 2 --critic-layers 4 --total-steps 400000

    # Study communication radius: vary --actor-layers (1,2,3) and compare.
    python -m scripts.train_decentralized --actor-layers 1 \
        --checkpoint-name gnn_dec_L1.pt
"""
from __future__ import annotations
import argparse
import torch

from rl.ctde_trainer import CTDETrainer, CTDETrainConfig
from rl.ppo import PPOConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", nargs="+", type=int, default=[4, 6])
    ap.add_argument("--missions", nargs="+", default=["power"])
    ap.add_argument("--shape", type=str, default="line", choices=["line", "L"])
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--sun-direction", nargs=3, type=float,
                    default=[0.0, 0.0, 1.0])
    ap.add_argument("--reward-mode", type=str, default="improvement",
                    choices=["improvement", "absolute"])
    ap.add_argument("--total-steps", type=int, default=400_000)
    ap.add_argument("--rollout-len", type=int, default=2048)
    ap.add_argument("--n-envs", type=int, default=32,
                    help="number of parallel environments")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--actor-layers", type=int, default=2,
                    help="MP layers for local actor (communication radius)")
    ap.add_argument("--critic-layers", type=int, default=4,
                    help="MP layers for centralized critic")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    ap.add_argument("--checkpoint-name", type=str,
                    default="gnn_decentralized.pt")
    ap.add_argument("--clip-ratio", type=float, default=0.2)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--n-epochs", type=int, default=4)
    ap.add_argument("--minibatch-size", type=int, default=256)
    args = ap.parse_args()

    cfg = CTDETrainConfig(
        n_modules_choices=args.n, mission_choices=args.missions,
        shape=args.shape, max_steps=args.max_steps,
        sun_direction=tuple(args.sun_direction), reward_mode=args.reward_mode,
        total_steps=args.total_steps, rollout_len=args.rollout_len, n_envs=args.n_envs,
        gamma=args.gamma, gae_lambda=args.gae_lambda, lr=args.lr,
        hidden_dim=args.hidden_dim, actor_layers=args.actor_layers,
        critic_layers=args.critic_layers, seed=args.seed, device=args.device,
        checkpoint_dir=args.checkpoint_dir, checkpoint_name=args.checkpoint_name,
    )
    ppo_cfg = PPOConfig(
        clip_ratio=args.clip_ratio, entropy_coef=args.entropy_coef,
        value_coef=args.value_coef, n_epochs=args.n_epochs,
        minibatch_size=args.minibatch_size,
    )

    print("=" * 60)
    print("DECENTRALIZED GNN CTDE TRAINING")
    print("=" * 60)
    print(f"N choices={args.n} missions={args.missions}")
    print(f"actor_layers={args.actor_layers} (local) "
          f"critic_layers={args.critic_layers} (global)")
    print(f"total_steps={args.total_steps} seed={args.seed}")
    print("=" * 60)

    trainer = CTDETrainer(cfg, ppo_cfg)
    trainer.train()


if __name__ == "__main__":
    main()