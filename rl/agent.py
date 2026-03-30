"""
agent.py
========
Full GNN-based agent that combines:
  - ConstellationObservationEncoder  (gnn_encoder.py)
  - HierarchicalPolicy               (policy_heads.py)

Exposes a clean API used by the trainer and evaluator:
  - act(obs, info) -> action dict
  - evaluate(batch) -> log_probs, entropy, values

Architecture recap
------------------
obs ──► ObservationBuilder ──► PyG graph + masks
                                        │
                                        ▼
                          GoalEncoder (mission mode + env features)
                                        │
                                        ▼
                          SwarmGNN (GAT layers + FiLM conditioning)
                                        │
                    ┌───────────────────┴────────────────────┐
                    ▼                                        ▼
           HierarchicalPolicy                          ValueHead
     (ActionType ► sub-head)                         V(s) scalar
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from typing import Dict, Optional, Tuple

from rl.gnn_encoder import ConstellationObservationEncoder
from rl.policy_heads import HierarchicalPolicy
from rl.obs_builder  import ObservationBuilder

import numpy as np


class ConstellationAgent(nn.Module):
    """
    Full agent for constellation swarm control.

    Parameters
    ----------
    env           : a ConstellationEnv / HierarchicalConstellationEnv instance
                    (used for dimension inference and obs building)
    hidden_dim    : embedding width throughout
    num_gnn_layers: number of GATv2 message-passing layers
    num_heads     : attention heads per GAT layer
    num_modes     : number of mission modes (for GoalEncoder)
    device        : torch device
    """

    def __init__(self,
                 env,
                 hidden_dim    : int = 128,
                 num_gnn_layers: int = 3,
                 num_heads     : int = 4,
                 num_modes     : int = 4,
                 device        : str = 'cpu'):
        super().__init__()

        self.device     = torch.device(device)
        self.hidden_dim = hidden_dim

        # ── Observation builder (not a nn.Module) ────────────────────────
        self.obs_builder = ObservationBuilder(
            env             = env,
            max_cubes       = env.num_cubes,
            max_groups      = getattr(env, 'max_groups',
                                      getattr(getattr(env, 'separation_reqs', None), 'max_groups', 1)),
            max_moves_per_cube     = 24,
            max_separation_actions = 100,
            device          = device,
        )

        # ── GNN encoder ──────────────────────────────────────────────────
        self.encoder = ConstellationObservationEncoder(
            cube_input_dim      = ObservationBuilder.CUBE_FEAT_DIM,
            group_input_dim     = ObservationBuilder.GROUP_FEAT_DIM,
            hidden_dim          = hidden_dim,
            num_gnn_layers      = num_gnn_layers,
            num_attention_heads = num_heads,
            num_mission_modes   = num_modes,
            env_feature_dim     = ObservationBuilder.GLOBAL_FEAT_DIM,
        ).to(self.device)

        # ── Policy + value heads ──────────────────────────────────────────
        max_groups = (env.max_groups
                      if hasattr(env, 'max_groups')
                      else env.separation_reqs.max_groups)

        self.policy = HierarchicalPolicy(
            hidden_dim             = hidden_dim,
            max_cubes              = env.num_cubes,
            max_groups             = max_groups,
            max_separation_actions = 100,
            max_docking_actions    = max_groups ** 2,
            max_moves_per_cube     = 24,
            num_action_types       = 5,
        ).to(self.device)

    # ──────────────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def act(self,
            obs       : np.ndarray,
            info      : Dict,
            deterministic: bool = False) -> Dict:
        """
        Choose an action given the current observation.

        Returns
        -------
        dict with keys: action_type, sub_action, log_prob, value
        """
        graph_data, masks = self.obs_builder.build(obs, info)
        graph_data = graph_data.to(self.device)

        mode_idx, env_feats = self._extract_goal(graph_data)

        global_feat, node_feat = self.encoder(graph_data, mode_idx, env_feats)

        cube_mask  = (graph_data.node_type == 0)
        group_mask = (graph_data.node_type == 1)

        return self.policy.sample_action(
            global_features  = global_feat,
            node_features    = node_feat,
            cube_mask_nodes  = cube_mask,
            group_mask_nodes = group_mask,
            batch            = graph_data.batch,
            masks            = masks,
            deterministic    = deterministic,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Training evaluation
    # ──────────────────────────────────────────────────────────────────────

    def evaluate_batch(self,
                       graph_batch : Data,
                       masks_batch : Dict,
                       mode_idx    : torch.Tensor,
                       env_feats   : torch.Tensor,
                       action_types: torch.Tensor,
                       sub_actions : torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Re-evaluate a batch of stored transitions for PPO loss.

        Parameters
        ----------
        graph_batch  : batched PyG Data (use Batch.from_data_list)
        masks_batch  : dict of batched mask tensors
        mode_idx     : [B] int tensor of mission mode indices
        env_feats    : [B, env_feat_dim] tensor
        action_types : [B] int tensor
        sub_actions  : [B] int tensor

        Returns
        -------
        dict with log_probs [B], entropy [B], values [B]
        """
        global_feat, node_feat = self.encoder(graph_batch, mode_idx, env_feats)

        cube_mask  = (graph_batch.node_type == 0)
        group_mask = (graph_batch.node_type == 1)

        return self.policy.evaluate_actions(
            global_features  = global_feat,
            node_features    = node_feat,
            cube_mask_nodes  = cube_mask,
            group_mask_nodes = group_mask,
            batch            = graph_batch.batch,
            masks            = masks_batch,
            stored_action_types = action_types,
            stored_sub_actions  = sub_actions,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _extract_goal(self,
                      graph_data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pull mission-mode index and environmental features from the graph's
        global node (node_type == 2).

        Returns
        -------
        mode_idx  : [1] long tensor
        env_feats : [1, env_feat_dim] float tensor
        """
        global_mask = (graph_data.node_type == 2)
        global_node = graph_data.x[global_mask]  # [1, feat_width]

        # Slot 6 of global features encodes normalised task type (0–1)
        # We map it back to an integer mode index [0, num_modes-1]
        num_modes = self.encoder.goal_encoder.mode_embedding.num_embeddings
        mode_float = global_node[0, 6] * 4.0        # was divided by 4 in builder
        mode_idx   = mode_float.long().clamp(0, num_modes - 1).unsqueeze(0)

        # Use the full global node features as env features
        env_dim   = self.encoder.goal_encoder.mlp[0].in_features - 16
        env_feats = global_node[:, :env_dim]

        return mode_idx, env_feats

    def parameters(self, recurse: bool = True):
        return list(self.encoder.parameters(recurse)) + \
               list(self.policy.parameters(recurse))

    def train(self, mode: bool = True):
        self.encoder.train(mode)
        self.policy.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def save(self, path: str):
        torch.save({
            'encoder': self.encoder.state_dict(),
            'policy' : self.policy.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt['encoder'])
        self.policy.load_state_dict(ckpt['policy'])