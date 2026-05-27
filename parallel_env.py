"""
parallel_env.py
===============
Parallel environment workers for massively faster rollout collection.
Uses multiprocessing to step multiple environments simultaneously.
"""

import multiprocessing as mp
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import traceback
import queue


def to_cpu(obj):
    """Recursively move tensors/containers to CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    elif isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_cpu(v) for v in obj]
    elif hasattr(obj, 'to'): # Handles PyG Batch/Data objects
        return obj.to('cpu')
    return obj


def _env_worker(
    worker_id: int,
    pipe: mp.connection,
    config_dict: dict,
):
    """
    Worker process that runs a single environment instance.
    Communicates with the main process via a Pipe.
    """
    import numpy as np
    import torch
    
    # Import here to avoid pickling issues
    from rl.train_fast import TrainingConfig
    from rl.env_wrapper import ConstellationTrainingEnv
    from tasks.constellation_tasks import FormConstellationTask
    from tasks.curriculum_tasks import TaskCurriculum, CurriculumSampler
    
    try:
        # Unpack only what we need — no TrainingConfig reconstruction at all
        num_cubes = config_dict['num_cubes']
        max_episode_steps = config_dict['max_episode_steps']
        time_step = config_dict['time_step']
        curriculum_enabled = config_dict['curriculum_enabled']
        num_cubes_range = config_dict['num_cubes_range']

        if curriculum_enabled:
            from tasks.curriculum_tasks import TaskCurriculum, CurriculumSampler
            curriculum = TaskCurriculum(num_cubes_range=num_cubes_range)
            sampler = CurriculumSampler(curriculum)
            task_key, task, num_cubes = sampler.sample()
        else:
            sampler = None
            task = FormConstellationTask(target_num_groups=2, target_baseline=5000.0)
            task_key = 'form_constellation'

        env = ConstellationTrainingEnv(
            num_cubes=num_cubes,
            task=task,
            max_steps=max_episode_steps,
            time_step=time_step,
        )
        env._current_task_key = task_key

        obs = env.reset()
        pipe.send(("ready", obs, worker_id))
        
        while True:
            cmd, data = pipe.recv()
            
            if cmd == "step":
                action_type, sub_action, masks = data
                result = env.step(action_type, sub_action, masks)
                # result = (graph_data, mode_idx, env_features, action_masks, reward, done, info)
                
                graph_data, mode_idx, env_features, action_masks, reward, done, info = result
                
                if done:
                    # Handle curriculum update
                    if sampler is not None:
                        task_key = getattr(env, '_current_task_key', 'unknown')
                        sampler.record_outcome(
                            task_key,
                            success=info.get('task_complete', False),
                            progress=info.get('task_progress', 0.0),
                        )
                        new_task_key, new_task, num_cubes = sampler.sample()
                        env.task = new_task
                        env.num_cubes = num_cubes
                        env._current_task_key = new_task_key
                    
                    # Auto-reset
                    new_obs = env.reset()
                    info['terminal_observation'] = (graph_data, mode_idx, env_features, action_masks)
                    obs_cpu = to_cpu((graph_data, mode_idx, env_features, action_masks))
                    pipe.send(("step_result", obs_cpu, reward, done, info, worker_id))
                else:
                    pipe.send(("step_result", 
                              (graph_data, mode_idx, env_features, action_masks),
                              reward, False, info, worker_id))
            
            elif cmd == "get_env_state":
                # Return current env state for action computation
                pipe.send(("env_state", {
                    'constellation': env.constellation,
                    'controller': env.controller,
                    'movement': env.movement,
                    'mission_mode': env.mission_mode,
                    'sun_direction': env.sun_direction,
                    'earth_direction': env.earth_direction,
                    'target_direction': env.target_direction,
                    'sun_distance_au': env.sun_distance_au,
                }, worker_id))
            
            elif cmd == "reset":
                obs = env.reset()
                pipe.send(("reset_result", obs, worker_id))
            
            elif cmd == "close":
                pipe.send(("closed", None, worker_id))
                break
                
    except Exception as e:
        pipe.send(("error", traceback.format_exc(), worker_id))


class ParallelEnvManager:
    """
    Manages multiple environment workers for parallel rollout collection.
    
    Key design: environments step in parallel while the GPU computes actions.
    """
    
    def __init__(self, num_envs: int, config: 'TrainingConfig'):
        self.num_envs = num_envs
        self.config = config
        
        # Convert config to dict for pickling
        self.config_dict = {
            'num_cubes': config.num_cubes,
            'max_episode_steps': config.max_episode_steps,
            'time_step': config.time_step,
            'curriculum_enabled': config.curriculum_enabled,
            'num_cubes_range': config.num_cubes_range,
            'task_type': config.task_type,
        }
        self.workers: List[mp.Process] = []
        self.pipes: List[mp.Connection] = []
        
        self._start_workers()
    
    def _start_workers(self):
        """Start all worker processes."""
        import torch.multiprocessing as tmp
        tmp.set_sharing_strategy('file_system')
        mp.set_start_method('spawn', force=True)
        
        for i in range(self.num_envs):
            parent_pipe, child_pipe = mp.Pipe()
            
            worker = mp.Process(
                target=_env_worker,
                args=(i, child_pipe, self.config_dict),
                daemon=True,
            )
            worker.start()
            
            self.pipes.append(parent_pipe)
            self.workers.append(worker)
        
        # Wait for all workers to be ready
        initial_obs = []
        for pipe in self.pipes:
            msg_type, obs, worker_id = pipe.recv()
            assert msg_type == "ready", f"Worker {worker_id} failed: {obs}"
            initial_obs.append(obs)
        
        self.current_obs = initial_obs
        print(f"  All {self.num_envs} environment workers ready.")
    
    def step_async(self, actions: List[Tuple[int, int, Dict]]):
        """Send step commands to all workers asynchronously."""
        for i, (action_type, sub_action, masks) in enumerate(actions):
            self.pipes[i].send(("step", (action_type, sub_action, masks)))
    
    def step_wait(self) -> List[Tuple]:
        """Wait for all step results."""
        results = []
        for pipe in self.pipes:
            msg = pipe.recv()
            msg_type = msg[0]
            
            if msg_type == "error":
                # This will show you the ACTUAL traceback from the worker
                _, traceback_str, worker_id = msg
                print(f"\nCRITICAL: Worker {worker_id} crashed!")
                print(traceback_str)
                self.close()
                exit(1) # Or handle gracefully
                
            # If it's a normal result, it will be ("step_result", obs, reward, done, info, worker_id)
            _, obs, reward, done, info, worker_id = msg
            results.append((obs, reward, done, info))
        return results
    
    def step(self, actions: List[Tuple[int, int, Dict]]) -> List[Tuple]:
        """Step all environments synchronously."""
        self.step_async(actions)
        return self.step_wait()
    
    def close(self):
        """Shutdown all workers."""
        for pipe in self.pipes:
            pipe.send(("close", None))
        for worker in self.workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
