from rl.train_fast import train, TrainingConfig
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO agent")
    parser.add_argument("--curriculum_enabled", type=bool, default=True)
    parser.add_argument("--total_timesteps", type=int, default=50_000_000)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--rollout_steps", type=int, default=4096)
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    
    # New: model size arguments
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_gnn_layers", type=int, default=5)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    
    # New: parallelism arguments
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--use_amp", action="store_true", default=False)
    
    return parser.parse_args()


def main():
    args = parse_args()

    config = TrainingConfig(
        curriculum_enabled=args.curriculum_enabled,
        total_timesteps=args.total_timesteps,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        rollout_steps=args.rollout_steps,
        max_episode_steps=args.max_episode_steps,
        # New fields
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        num_attention_heads=args.num_attention_heads,
        learning_rate=args.learning_rate,
        num_envs=args.num_envs,
        num_workers=args.num_workers,
        use_amp=args.use_amp,
    )
    agent = train(config)
    return agent


if __name__ == "__main__":
    main()
