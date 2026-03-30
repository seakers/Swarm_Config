"""
evaluate.py
===========
Evaluation and visualisation utilities for the trained ConstellationAgent.

Functions
---------
evaluate_agent      – run N episodes and return statistics
render_episode      – run one episode with rendering / frame capture
plot_training_curves– plot reward and loss from CSV logs
run_ablation        – compare agent vs random baseline

Usage
-----
    from rl.evaluate import evaluate_agent, plot_training_curves

    stats = evaluate_agent(agent, env, num_episodes=20, deterministic=True)
    print(stats)

    plot_training_curves('logs/swarm_ppo_episodes.csv')
"""

from __future__ import annotations

import os
import csv
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_agent(agent,
                   env,
                   num_episodes  : int  = 20,
                   deterministic : bool = True,
                   render        : bool = False,
                   seed          : Optional[int] = None) -> Dict[str, float]:
    """
    Run num_episodes evaluation episodes.

    Returns
    -------
    dict with mean / std / min / max of:
      episode_reward, episode_length, task_progress,
      num_groups, delta_v_used, task_complete_rate
    """
    rewards, lengths, progresses = [], [], []
    num_groups_list, task_complete = [], []

    agent.eval()

    for ep in range(num_episodes):
        ep_seed = None if seed is None else seed + ep
        obs, info = env.reset(seed=ep_seed)
        ep_reward = 0.0
        ep_length = 0
        done      = False

        while not done:
            action_dict = agent.act(obs, info, deterministic=deterministic)
            action_type = action_dict['action_type']
            sub_action  = action_dict['sub_action']

            env_action = np.array([action_type, sub_action], dtype=np.int64)
            obs, reward, terminated, truncated, info = env.step(env_action)

            ep_reward += reward
            ep_length += 1
            done = terminated or truncated

            if render:
                env.render()

        rewards.append(ep_reward)
        lengths.append(ep_length)
        progresses.append(info.get('task_progress', 0.0))
        num_groups_list.append(info.get('num_groups', 1))
        task_complete.append(float(info.get('task_complete', False)))

    stats = {}
    for name, vals in [('reward',    rewards),
                        ('length',    lengths),
                        ('progress',  progresses),
                        ('num_groups',num_groups_list)]:
        arr = np.array(vals, dtype=np.float32)
        stats[f'{name}_mean'] = float(arr.mean())
        stats[f'{name}_std']  = float(arr.std())
        stats[f'{name}_min']  = float(arr.min())
        stats[f'{name}_max']  = float(arr.max())

    stats['task_complete_rate'] = float(np.mean(task_complete))

    logger.info(
        f"Eval ({num_episodes} eps) | "
        f"reward={stats['reward_mean']:.3f}±{stats['reward_std']:.3f} | "
        f"progress={stats['progress_mean']:.3f} | "
        f"complete={stats['task_complete_rate']:.2%}"
    )
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Random baseline
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_random_baseline(env,
                              num_episodes: int = 20,
                              seed: Optional[int] = None) -> Dict[str, float]:
    """
    Evaluate a random policy (uniform over valid actions) for comparison.
    """
    rewards, progresses, completes = [], [], []

    for ep in range(num_episodes):
        ep_seed = None if seed is None else seed + ep
        obs, info = env.reset(seed=ep_seed)
        ep_reward = 0.0
        done      = False

        while not done:
            cache = getattr(env, '_valid_actions_cache', {})
            # Randomly pick action type with valid sub-actions
            valid_types = [t for t in range(5) if len(cache.get(t, [])) > 0]
            at  = int(np.random.choice(valid_types)) if valid_types else 4
            n   = len(cache.get(at, [None]))
            sub = int(np.random.randint(0, max(1, n)))

            obs, reward, terminated, truncated, info = \
                env.step(np.array([at, sub], dtype=np.int64))
            ep_reward += reward
            done = terminated or truncated

        rewards.append(ep_reward)
        progresses.append(info.get('task_progress', 0.0))
        completes.append(float(info.get('task_complete', False)))

    return {
        'reward_mean'       : float(np.mean(rewards)),
        'reward_std'        : float(np.std(rewards)),
        'progress_mean'     : float(np.mean(progresses)),
        'task_complete_rate': float(np.mean(completes)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Episode renderer / frame capture
# ─────────────────────────────────────────────────────────────────────────────

def render_episode(agent,
                   env,
                   deterministic: bool = True,
                   save_frames  : bool = False,
                   frame_dir    : str  = 'frames',
                   seed         : Optional[int] = None) -> List[np.ndarray]:
    """
    Run one episode with rendering.

    Returns list of RGB frames if save_frames=True, else empty list.
    """
    import matplotlib.pyplot as plt

    if save_frames:
        os.makedirs(frame_dir, exist_ok=True)

    frames = []
    obs, info = env.reset(seed=seed)
    done = False
    step = 0

    while not done:
        action_dict = agent.act(obs, info, deterministic=deterministic)
        action = np.array([action_dict['action_type'],
                           action_dict['sub_action']], dtype=np.int64)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

        if save_frames:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
                plt.imsave(os.path.join(frame_dir, f'frame_{step:04d}.png'), frame)

    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Training curve plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(episode_csv: str,
                         update_csv : Optional[str] = None,
                         window     : int = 20,
                         save_path  : Optional[str] = None):
    """
    Plot episode rewards and (optionally) loss curves from CSV logs.

    Parameters
    ----------
    episode_csv : path to episodes CSV (output of PPOTrainer)
    update_csv  : path to updates CSV (optional)
    window      : smoothing window for rolling mean
    save_path   : if given, save figure instead of showing
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        logger.warning("matplotlib not available – skipping plot")
        return

    # ── Load data ────────────────────────────────────────────────────────
    steps, rewards, progresses, stages = [], [], [], []
    with open(episode_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['step']))
            rewards.append(float(row['reward']))
            progresses.append(float(row['task_progress']))
            stages.append(int(row['stage']))

    steps     = np.array(steps)
    rewards   = np.array(rewards)
    progresses= np.array(progresses)

    def rolling_mean(x, w):
        return np.convolve(x, np.ones(w) / w, mode='valid')

    # ── Figure ───────────────────────────────────────────────────────────
    n_plots = 3 if update_csv else 2
    fig = plt.figure(figsize=(14, 4 * n_plots))
    fig.patch.set_facecolor('#0a0a1a')
    gs  = gridspec.GridSpec(n_plots, 1, hspace=0.4)

    kw_ax  = dict(facecolor='#12122a')
    kw_raw = dict(alpha=0.25, linewidth=0.8)
    kw_sm  = dict(linewidth=2.0)

    colors = {'reward': '#00d4ff', 'progress': '#7fff7f', 'loss': '#ff7f50'}

    # ── Reward ───────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0], **kw_ax)
    ax1.plot(steps, rewards,          color=colors['reward'], label='reward (raw)', **kw_raw)
    if len(rewards) > window:
        sm_steps = steps[window - 1:]
        ax1.plot(sm_steps, rolling_mean(rewards, window),
                 color=colors['reward'], label=f'reward (mean{window})', **kw_sm)
    _add_stage_bands(ax1, steps, stages)
    ax1.set_xlabel('Environment steps', color='white')
    ax1.set_ylabel('Episode reward',     color='white')
    ax1.set_title('Training Reward', color='white', fontsize=12)
    ax1.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
    _style_ax(ax1)

    # ── Task progress ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], **kw_ax)
    ax2.plot(steps, progresses, color=colors['progress'], label='task progress', **kw_raw)
    if len(progresses) > window:
        ax2.plot(sm_steps, rolling_mean(progresses, window),
                 color=colors['progress'], label=f'progress (mean{window})', **kw_sm)
    _add_stage_bands(ax2, steps, stages)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Environment steps', color='white')
    ax2.set_ylabel('Task progress',      color='white')
    ax2.set_title('Task Progress', color='white', fontsize=12)
    ax2.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
    _style_ax(ax2)

    # ── Losses ───────────────────────────────────────────────────────────
    if update_csv and os.path.exists(update_csv):
        u_steps, pg_l, vf_l, ent = [], [], [], []
        with open(update_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                u_steps.append(int(row['step']))
                pg_l.append(float(row['pg_loss']))
                vf_l.append(float(row['vf_loss']))
                ent.append(float(row['entropy']))

        ax3 = fig.add_subplot(gs[2], **kw_ax)
        ax3.plot(u_steps, pg_l, color='#ff6b6b', label='policy loss',  **kw_sm)
        ax3.plot(u_steps, vf_l, color='#ffd93d', label='value loss',   **kw_sm)
        ax3.plot(u_steps, ent,  color='#6bcb77', label='entropy',       **kw_sm)
        ax3.set_xlabel('Environment steps', color='white')
        ax3.set_ylabel('Loss',               color='white')
        ax3.set_title('PPO Losses', color='white', fontsize=12)
        ax3.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
        _style_ax(ax3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        logger.info(f"Saved training curves → {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _style_ax(ax):
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333360')
    ax.grid(True, color='#1e1e4a', linewidth=0.5)


def _add_stage_bands(ax, steps, stages):
    """Shade background by curriculum stage."""
    import matplotlib.patches as mpatches
    stage_colors = ['#1a2a1a', '#1a1a2a', '#2a1a1a', '#2a2a1a']
    if not stages:
        return
    prev_stage = stages[0]
    band_start = steps[0]
    for i, (s, st) in enumerate(zip(steps, stages)):
        if st != prev_stage or i == len(steps) - 1:
            c = stage_colors[prev_stage % len(stage_colors)]
            ax.axvspan(band_start, s, alpha=0.3, color=c, linewidth=0)
            band_start  = s
            prev_stage  = st


# ─────────────────────────────────────────────────────────────────────────────
# Quick ablation helper
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(agent, env, num_episodes: int = 20, seed: int = 42) -> None:
    """Compare trained agent vs random baseline and log results."""
    logger.info("=== Ablation: Trained Agent ===")
    agent_stats = evaluate_agent(
        agent, env, num_episodes=num_episodes,
        deterministic=True, seed=seed)

    logger.info("=== Ablation: Random Baseline ===")
    rand_stats = evaluate_random_baseline(env, num_episodes=num_episodes, seed=seed)

    headers = ['metric', 'agent', 'random', 'improvement']
    rows    = []
    for key in ('reward_mean', 'progress_mean', 'task_complete_rate'):
        ag  = agent_stats.get(key, float('nan'))
        rnd = rand_stats.get(key, float('nan'))
        imp = ag - rnd
        rows.append([key, f'{ag:.4f}', f'{rnd:.4f}', f'{imp:+.4f}'])

    col_w = max(len(h) for h in headers + [r[0] for r in rows]) + 2
    header_line = '  '.join(h.ljust(col_w) for h in headers)
    logger.info('\n' + header_line)
    for row in rows:
        logger.info('  '.join(c.ljust(col_w) for c in row))