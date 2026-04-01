from rl.train import train, TrainingConfig

def main():
    params = {
        "num_cubes": 27, # 64
        "total_timesteps": 1_000, # 1_000_000
        "task_type": "form_constellation",
        "save_dir": "./checkpoints",
        "log_dir": "./logs",
        "rollout_steps": 2048,
        "max_episode_steps": 1000,
    }
    config = TrainingConfig(**params)
    
    agent = train(config)
    
    return agent


if __name__ == "__main__":
    main()