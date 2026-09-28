# scripts/train_gnn.py
"""
Train a (centralized) GNN PPO policy.

Because the GNN handles any N and mission is a per-node feature, one checkpoint
can be trained across multiple module counts and missions.

Usage:
    # Single task, fixed N
    python -m scripts.train_gnn --n 6 --missions power

    # One policy across N and tasks (for generalization experiments)
    python -m scripts.train_gnn --n 4 6 8 --missions power thermal comms \
        --total-steps 500000 --n-layers 4

The number of message-passing layers controls information locality:
  large --n-layers => centralized (whole-graph) reasoning.
"""
from __future__ import annotations
import argparse
import torch

from rl.gnn_trainer import GNNPPOTrainer, GNNTrainConfig
from rl.ppo import PPOConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", nargs="+", type=int, default=[4, 6],
                    help="module counts to train across (sampled per episode)")
    ap.add_argument("--missions", nargs="+", default=["power"],
                    help="missions to train across (sampled per episode)")
    ap.add_argument("--shape", type=str, default="line", choices=["line", "L"])
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--sun-direction", nargs=3, type=float,
                    default=[0.0, 0.0, 1.0])
    ap.add_argument("--reward-mode", type=str, default="improvement",
                    choices=["improvement", "absolute"])
    ap.add_argument("--total-steps", type=int, default=300_000)
    ap.add_argument("--rollout-len", type=int, default=2048)
    ap.add_argument("--n-envs", type=int, default=32,
                    help="number of parallel environments")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=3,
                    help="message-passing layers (large=centralized reasoning)")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    ap.add_argument("--checkpoint-name", type=str, default="gnn_centralized.pt")
    # PPO
    ap.add_argument("--clip-ratio", type=float, default=0.2)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--n-epochs", type=int, default=4)
    ap.add_argument("--minibatch-size", type=int, default=256)
    args = ap.parse_args()

    cfg = GNNTrainConfig(
        n_modules_choices=args.n, mission_choices=args.missions,
        shape=args.shape, max_steps=args.max_steps,
        sun_direction=tuple(args.sun_direction), reward_mode=args.reward_mode,
        total_steps=args.total_steps, rollout_len=args.rollout_len,
        n_envs=args.n_envs, gamma=args.gamma, gae_lambda=args.gae_lambda, lr=args.lr,
        hidden_dim=args.hidden_dim, n_layers=args.n_layers,
        seed=args.seed, device=args.device,
        checkpoint_dir=args.checkpoint_dir, checkpoint_name=args.checkpoint_name,
    )
    ppo_cfg = PPOConfig(
        clip_ratio=args.clip_ratio, entropy_coef=args.entropy_coef,
        value_coef=args.value_coef, n_epochs=args.n_epochs,
        minibatch_size=args.minibatch_size,
    )

    print("=" * 60)
    print("CENTRALIZED GNN PPO TRAINING")
    print("=" * 60)
    print(f"N choices={args.n} missions={args.missions} "
          f"n_layers={args.n_layers} seed={args.seed}")
    print(f"total_steps={args.total_steps} device={args.device}")
    print("=" * 60)

    trainer = GNNPPOTrainer(cfg, ppo_cfg)
    trainer.train()


if __name__ == "__main__":
    main()