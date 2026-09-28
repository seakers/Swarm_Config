from environment.make_env import make_env
from environment.graph import is_connected
from controllers.random_controller import RandomController
from evaluation.metrics import run_episode



env = make_env(n_modules=8, mission='power',
                max_steps=40, seed=0)

obs, info = env.reset()

legal = env.legal_actions()
