"""
train.py
========
Top-level training entry point.

Usage
-----
    # Quick smoke test (8 cubes, 50k steps)
    python train.py --num_cubes 8 --total_timesteps 50000 --run_name smoke_test

    # Full training run
    python train.py --num_cubes 27 --total_timesteps 1000000 --curriculum

    # Evaluate a saved checkpoint
    python train.py --eval_only --checkpoint checkpoints/swarm_ppo_best.pt
"""

import argparse
import logging

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description='Train constellation swarm agent')

    # Environment
    p.add_argument('--num_cubes',    type=int,   default=8)
    p.add_argument('--max_steps',    type=int,   default=500)
    p.add_argument('--num_groups',   type=int,   default=2)
    p.add_argument('--target_baseline', type=float, default=5000.0)

    # Training
    p.add_argument('--total_timesteps', type=int,   default=50_000)
    p.add_argument('--rollout_steps',   type=int,   default=2048)
    p.add_argument('--num_epochs',      type=int,   default=10)
    p.add_argument('--batch_size',      type=int,   default=64)
    p.add_argument('--lr',              type=float, default=3e-4)
    p.add_argument('--clip_range',      type=float, default=0.2)
    p.add_argument('--entropy_coef',    type=float, default=0.01)
    p.add_argument('--curriculum',      action='store_true')

    # Architecture
    p.add_argument('--hidden_dim',      type=int,  default=128)
    p.add_argument('--num_gnn_layers',  type=int,  default=3)
    p.add_argument('--num_heads',       type=int,  default=4)

    # Logging / saving
    p.add_argument('--run_name',       type=str,  default='swarm_ppo')
    p.add_argument('--checkpoint_dir', type=str,  default='checkpoints')
    p.add_argument('--log_dir',        type=str,  default='logs')
    p.add_argument('--log_interval',   type=int,  default=10)
    p.add_argument('--save_interval',  type=int,  default=50)

    # Evaluation
    p.add_argument('--eval_only',   action='store_true')
    p.add_argument('--checkpoint',  type=str,  default=None)
    p.add_argument('--num_eval_eps',type=int,  default=20)

    # Device
    p.add_argument('--device', type=str, default='auto',
                   help='"auto", "cpu", or "cuda"')

    return p.parse_args()


def build_env(args):
    """Construct the environment from CLI args."""
    from env.constellation_env import ConstellationEnv
    from tasks.constellation_tasks import FormConstellationTask

    task = FormConstellationTask(
        target_num_groups = args.num_groups,
        target_baseline   = args.target_baseline,
    )
    env = ConstellationEnv(
        num_cubes = args.num_cubes,
        task      = task,
        max_steps = args.max_steps,
    )
    return env


def main():
    args = parse_args()

    import torch
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    logger.info(f"Device: {device}")

    env = build_env(args)

    if args.eval_only:
        # ── Evaluation mode ───────────────────────────────────────────────
        from rl.agent    import ConstellationAgent
        from rl.evaluate import evaluate_agent, run_ablation

        agent = ConstellationAgent(env, device=device)
        if args.checkpoint:
            agent.load(args.checkpoint)
            logger.info(f"Loaded checkpoint: {args.checkpoint}")

        stats = evaluate_agent(agent, env, num_episodes=args.num_eval_eps,
                               deterministic=True)
        for k, v in stats.items():
            logger.info(f"  {k}: {v:.4f}")

        run_ablation(agent, env, num_episodes=args.num_eval_eps)

    else:
        # ── Training mode ─────────────────────────────────────────────────
        from rl.trainer import PPOTrainer, TrainingConfig

        cfg = TrainingConfig(
            total_timesteps = args.total_timesteps,
            rollout_steps   = args.rollout_steps,
            num_epochs      = args.num_epochs,
            batch_size      = args.batch_size,
            learning_rate   = args.lr,
            clip_range      = args.clip_range,
            entropy_coef    = args.entropy_coef,
            curriculum      = args.curriculum,
            hidden_dim      = args.hidden_dim,
            num_gnn_layers  = args.num_gnn_layers,
            num_heads       = args.num_heads,
            run_name        = args.run_name,
            checkpoint_dir  = args.checkpoint_dir,
            log_dir         = args.log_dir,
            log_interval    = args.log_interval,
            save_interval   = args.save_interval,
            device          = device,
        )

        trainer = PPOTrainer(env, cfg)

        if args.checkpoint:
            trainer.agent.load(args.checkpoint)
            logger.info(f"Resuming from checkpoint: {args.checkpoint}")

        trainer.train()

        # Quick post-training evaluation
        from rl.evaluate import evaluate_agent, plot_training_curves
        stats = evaluate_agent(trainer.agent, env,
                               num_episodes=args.num_eval_eps,
                               deterministic=True)
        logger.info("Post-training evaluation:")
        for k, v in stats.items():
            logger.info(f"  {k}: {v:.4f}")

        import os
        ep_csv  = os.path.join(args.log_dir, f"{args.run_name}_episodes.csv")
        upd_csv = os.path.join(args.log_dir, f"{args.run_name}_updates.csv")
        fig_path= os.path.join(args.log_dir, f"{args.run_name}_curves.png")
        plot_training_curves(ep_csv, upd_csv, save_path=fig_path)


if __name__ == '__main__':
    import logging
    logging.basicConfig(
        level   = logging.INFO,
        format  = '%(asctime)s | %(levelname)s | %(message)s',
        datefmt = '%H:%M:%S',
    )
    main()