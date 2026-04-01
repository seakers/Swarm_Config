"""
utils.py
========
Utility functions for RL training.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute explained variance.
    
    ev = 1 - Var[y_true - y_pred] / Var[y_true]
    """
    var_y = np.var(y_true)
    if var_y == 0:
        return 0.0
    return 1 - np.var(y_true - y_pred) / var_y


def linear_schedule(initial_value: float, final_value: float = 0.0):
    """
    Linear learning rate schedule.
    
    Returns a function that computes the current learning rate
    based on progress (0 to 1).
    """
    def func(progress: float) -> float:
        return final_value + (initial_value - final_value) * (1 - progress)
    return func


def save_training_config(config: dict, path: str) -> None:
    """Save training configuration to JSON."""
    with open(path, 'w') as f:
        json.dump(config, f, indent=2, default=str)


def load_training_config(path: str) -> dict:
    """Load training configuration from JSON."""
    with open(path, 'r') as f:
        return json.load(f)


class RunningMeanStd:
    """
    Running mean and standard deviation tracker.
    
    Useful for observation normalization.
    """
    
    def __init__(self, shape: Tuple[int, ...] = (), epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x: np.ndarray) -> None:
        """Update statistics with new batch of data."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        
        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize input using running statistics."""
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


class Logger:
    """Simple logger for training metrics."""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.metrics: Dict[str, List[float]] = {}
        self.steps: Dict[str, List[int]] = {}
    
    def log(self, name: str, value: float, step: int) -> None:
        """Log a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
            self.steps[name] = []
        
        self.metrics[name].append(value)
        self.steps[name].append(step)
    
    def save(self) -> None:
        """Save all metrics to file."""
        data = {
            'metrics': self.metrics,
            'steps': self.steps,
        }
        
        path = os.path.join(self.log_dir, 'training_log.json')
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def get_recent(self, name: str, n: int = 100) -> List[float]:
        """Get most recent n values for a metric."""
        if name not in self.metrics:
            return []
        return self.metrics[name][-n:]