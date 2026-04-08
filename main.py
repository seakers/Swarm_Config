from rl.train import train, TrainingConfig

def main():
    params = {
        "curriculum_enabled": True,
        # "num_cubes": 8, # 64 # not relevant when using curriculum
        "total_timesteps": 100, # 1_000_000
        # "task_type": "form_constellation", # "form_constellation" # not relevant when using curriculum
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