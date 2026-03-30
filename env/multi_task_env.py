import numpy as np
from typing import Tuple, Optional, Dict

from env.env import SwarmReconfigurationEnv
from tasks.tasks import Task, FormCubeTask, FormPlaneTask, FormLineTask, MinimizeSurfaceTask, MaximizeSpreadTask


class MultiTaskSwarmEnv(SwarmReconfigurationEnv):
    """
    Environment that randomly samples tasks from a task distribution.
    
    Useful for training a general-purpose reconfiguration agent.
    """
    
    def __init__(self, 
                 task_distribution: Optional[Dict[Task, float]] = None,
                 **kwargs):
        """
        Args:
            task_distribution: Dict mapping tasks to their sampling probabilities.
                              If None, uses uniform distribution over default tasks.
            **kwargs: Passed to SwarmReconfigurationEnv
        """
        # Initialize with a dummy task first
        super().__init__(task=FormCubeTask(), **kwargs)
        
        if task_distribution is None:
            # Default task distribution
            self.tasks = [
                FormCubeTask(target_size=4),
                FormPlaneTask(normal=(0, 0, 1), width=8, height=8),
                FormPlaneTask(normal=(1, 0, 0), width=8, height=8),
                FormPlaneTask(normal=(0, 1, 0), width=8, height=8),
                FormLineTask(axis=(1, 0, 0), length=64),
                FormLineTask(axis=(0, 1, 0), length=64),
                FormLineTask(axis=(0, 0, 1), length=64),
                MinimizeSurfaceTask(),
                MaximizeSpreadTask(target_baseline=15.0),
            ]
            self.task_probs = [1.0 / len(self.tasks)] * len(self.tasks)
        else:
            self.tasks = list(task_distribution.keys())
            probs = list(task_distribution.values())
            total = sum(probs)
            self.task_probs = [p / total for p in probs]
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        # Sample a task
        if seed is not None:
            np.random.seed(seed)
        
        task_idx = np.random.choice(len(self.tasks), p=self.task_probs)
        self.task = self.tasks[task_idx]
        
        obs, info = super().reset(seed=seed, options=options)
        
        info['sampled_task_idx'] = task_idx
        info['sampled_task_type'] = self.task.get_task_info().get('task_type', 'unknown')
        
        return obs, info