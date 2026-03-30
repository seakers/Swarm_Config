"""
policy_heads.py
===============
Hierarchical policy heads for the hybrid action space.

Action types:
0 - Cube hinge move
1 - Separation
2 - Docking
3 - Maneuver
4 - No-op
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, Tuple, Optional, List
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Action-type selector
# ─────────────────────────────────────────────────────────────────────────────

class ActionTypeHead(nn.Module):
    """
    Selects which type of action to take.

    Outputs logits over action types: [cube_move, separation, docking, maneuver, noop]
    """

    def __init__(self, input_dim: int = 128, num_action_types: int = 5):
        super().__init__()
        self.num_action_types = num_action_types
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_action_types),
        )

    def forward(self,
                global_features: torch.Tensor,
                action_type_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_features : [batch, input_dim]
            action_type_mask: [batch, num_action_types] – True where valid

        Returns:
            logits: [batch, num_action_types]
        """
        logits = self.net(global_features)
        logits = logits.masked_fill(~action_type_mask, -1e9)
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# Cube-move head  (node-level scoring)
# ─────────────────────────────────────────────────────────────────────────────

class CubeMoveHead(nn.Module):
    """
    Selects which cube move to execute.

    Each cube node is scored for every one of its 24 possible hinge moves
    (12 edges × 2 directions).  Invalid moves are masked out before sampling.
    """

    def __init__(self,
                 node_dim: int = 128,
                 global_dim: int = 128,
                 max_moves_per_cube: int = 24):
        super().__init__()
        self.max_moves_per_cube = max_moves_per_cube

        self.node_scorer = nn.Sequential(
            nn.Linear(node_dim + global_dim, 128),
            nn.ReLU(),
            nn.Linear(128, max_moves_per_cube),
        )

    def forward(self,
                node_features: torch.Tensor,
                global_features: torch.Tensor,
                cube_mask: torch.Tensor,
                batch: torch.Tensor,
                move_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features : [num_nodes, node_dim]
            global_features: [batch_size, global_dim]
            cube_mask      : [num_nodes] bool – True for cube nodes
            batch          : [num_nodes] int  – batch index per node
            move_mask      : [batch_size, max_cubes, max_moves_per_cube] bool

        Returns:
            logits: [batch_size, max_cubes * max_moves_per_cube]
        """
        batch_size = global_features.size(0)
        max_cubes  = move_mask.size(1)

        # Cube nodes only
        cube_nodes  = node_features[cube_mask]          # [C, node_dim]
        cube_batch  = batch[cube_mask]                  # [C]
        global_exp  = global_features[cube_batch]       # [C, global_dim]

        scores = self.node_scorer(torch.cat([cube_nodes, global_exp], dim=-1))
        # scores: [C, max_moves_per_cube]

        # Scatter into padded [batch, max_cubes, max_moves_per_cube] tensor
        logits = torch.full(
            (batch_size, max_cubes, self.max_moves_per_cube),
            -1e9,
            device=node_features.device,
        )

        # Count how many cube nodes belong to each batch item to assign cube idx
        cube_idx = torch.zeros_like(cube_batch)
        batch_cube_counter = torch.zeros(batch_size, dtype=torch.long, device=cube_batch.device)
        for n in range(cube_nodes.size(0)):
            b = cube_batch[n].item()
            cube_idx[n] = batch_cube_counter[b]
            batch_cube_counter[b] += 1

        # Clamp in case we exceed max_cubes (shouldn't happen in practice)
        valid_c = cube_idx < max_cubes
        logits[cube_batch[valid_c], cube_idx[valid_c]] = scores[valid_c]

        # Apply move mask
        logits = logits.masked_fill(~move_mask, -1e9)

        # Flatten to [batch, max_cubes * max_moves_per_cube]
        return logits.view(batch_size, -1)


# ─────────────────────────────────────────────────────────────────────────────
# Separation head
# ─────────────────────────────────────────────────────────────────────────────

class SeparationHead(nn.Module):
    """
    Scores candidate separation actions.

    The constellation controller enumerates valid separation options.  We
    represent each option as a *set* of cube IDs to split off.  We embed that
    set via mean-pooling of the corresponding cube node features and then score
    the option with an MLP.
    """

    def __init__(self,
                 node_dim: int = 128,
                 global_dim: int = 128,
                 max_separation_actions: int = 100,
                 hidden_dim: int = 128):
        super().__init__()
        self.max_separation_actions = max_separation_actions

        # Scores a (global_state, separation_set_embedding) pair
        self.scorer = nn.Sequential(
            nn.Linear(global_dim + node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self,
                node_features: torch.Tensor,
                global_features: torch.Tensor,
                cube_mask: torch.Tensor,
                batch: torch.Tensor,
                sep_cube_sets: List[List[List[int]]],
                sep_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features  : [num_nodes, node_dim]
            global_features: [batch_size, global_dim]
            cube_mask      : [num_nodes] bool
            batch          : [num_nodes] int
            sep_cube_sets  : list[batch] of list[action] of list[cube_id]
                             – which cube IDs form each candidate separation
            sep_mask       : [batch_size, max_separation_actions] bool

        Returns:
            logits: [batch_size, max_separation_actions]
        """
        batch_size = global_features.size(0)
        device     = node_features.device

        # Build a lookup: (batch_idx, cube_id) -> node_feature row
        cube_nodes = node_features[cube_mask]   # [C, node_dim]
        cube_batch = batch[cube_mask]           # [C]

        # Map (batch_b, cube_id) -> row in cube_nodes
        # We'll build a dict for simplicity; in a real hot-loop consider tensors
        node_lookup: Dict[Tuple[int, int], int] = {}
        batch_cube_counter = torch.zeros(batch_size, dtype=torch.long)
        for row in range(cube_nodes.size(0)):
            b = cube_batch[row].item()
            c = int(batch_cube_counter[b].item())
            node_lookup[(b, c)] = row
            batch_cube_counter[b] += 1

        logits = torch.full(
            (batch_size, self.max_separation_actions),
            -1e9,
            device=device,
        )

        for b in range(batch_size):
            actions_for_b = sep_cube_sets[b] if b < len(sep_cube_sets) else []
            for a_idx, cube_ids in enumerate(actions_for_b):
                if a_idx >= self.max_separation_actions:
                    break

                # Embed the separating subset via mean-pool of cube features
                rows = [node_lookup[(b, cid)] for cid in cube_ids
                        if (b, cid) in node_lookup]
                if rows:
                    sep_emb = cube_nodes[rows].mean(dim=0)   # [node_dim]
                else:
                    sep_emb = torch.zeros(cube_nodes.size(-1), device=device)

                inp = torch.cat([global_features[b], sep_emb], dim=-1)  # [global+node]
                logits[b, a_idx] = self.scorer(inp.unsqueeze(0)).squeeze()

        logits = logits.masked_fill(~sep_mask, -1e9)
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# Docking head
# ─────────────────────────────────────────────────────────────────────────────

class DockingHead(nn.Module):
    """
    Scores candidate docking actions (pairs of groups).
    """

    def __init__(self,
                 node_dim: int = 128,
                 global_dim: int = 128,
                 max_groups: int = 8,
                 hidden_dim: int = 64):
        super().__init__()
        self.max_groups = max_groups
        self.max_docking_actions = max_groups * max_groups

        self.group_scorer = nn.Sequential(
            nn.Linear(global_dim + 2 * node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self,
                node_features: torch.Tensor,
                global_features: torch.Tensor,
                group_mask_nodes: torch.Tensor,
                batch: torch.Tensor,
                dock_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features    : [num_nodes, node_dim]
            global_features  : [batch_size, global_dim]
            group_mask_nodes : [num_nodes] bool – True for group super-nodes
            batch            : [num_nodes] int
            dock_mask        : [batch_size, max_groups, max_groups] bool

        Returns:
            logits: [batch_size, max_groups * max_groups]
        """
        batch_size = global_features.size(0)
        device     = node_features.device

        group_nodes = node_features[group_mask_nodes]    # [G_total, node_dim]
        group_batch = batch[group_mask_nodes]            # [G_total]

        node_dim = group_nodes.size(-1)

        logits = torch.full(
            (batch_size, self.max_groups, self.max_groups),
            -1e9,
            device=device,
        )

        # Collect group node features per batch element
        groups_per_batch: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]
        for row in range(group_nodes.size(0)):
            b = group_batch[row].item()
            if b < batch_size:
                groups_per_batch[b].append(group_nodes[row])

        for b in range(batch_size):
            g_feats = groups_per_batch[b]
            for i in range(min(len(g_feats), self.max_groups)):
                for j in range(min(len(g_feats), self.max_groups)):
                    if i == j:
                        continue
                    inp = torch.cat([global_features[b],
                                     g_feats[i], g_feats[j]], dim=-1)
                    logits[b, i, j] = self.group_scorer(inp.unsqueeze(0)).squeeze()

        logits = logits.masked_fill(~dock_mask, -1e9)
        return logits.view(batch_size, -1)


# ─────────────────────────────────────────────────────────────────────────────
# Maneuver head
# ─────────────────────────────────────────────────────────────────────────────

class ManeuverHead(nn.Module):
    """
    Scores maneuver direction for each group.

    7 directions: +x, -x, +y, -y, +z, -z, zero-thrust
    """

    NUM_DIRECTIONS = 7

    def __init__(self,
                 node_dim: int = 128,
                 global_dim: int = 128,
                 max_groups: int = 8,
                 hidden_dim: int = 64):
        super().__init__()
        self.max_groups    = max_groups
        self.num_dir       = self.NUM_DIRECTIONS

        self.direction_head = nn.Sequential(
            nn.Linear(node_dim + global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_dir),
        )

    def forward(self,
                node_features: torch.Tensor,
                global_features: torch.Tensor,
                group_mask_nodes: torch.Tensor,
                batch: torch.Tensor,
                maneuver_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            maneuver_mask: [batch_size, max_groups, num_directions] bool

        Returns:
            logits: [batch_size, max_groups * num_directions]
        """
        batch_size = global_features.size(0)
        device     = node_features.device

        group_nodes = node_features[group_mask_nodes]
        group_batch = batch[group_mask_nodes]

        logits = torch.full(
            (batch_size, self.max_groups, self.num_dir),
            -1e9,
            device=device,
        )

        batch_g_counter = torch.zeros(batch_size, dtype=torch.long)
        for row in range(group_nodes.size(0)):
            b = group_batch[row].item()
            g_idx = int(batch_g_counter[b].item())
            if g_idx < self.max_groups:
                inp = torch.cat([group_nodes[row], global_features[b]], dim=-1)
                logits[b, g_idx] = self.direction_head(inp.unsqueeze(0)).squeeze()
            batch_g_counter[b] += 1

        logits = logits.masked_fill(~maneuver_mask, -1e9)
        return logits.view(batch_size, -1)


# ─────────────────────────────────────────────────────────────────────────────
# Value head (critic)
# ─────────────────────────────────────────────────────────────────────────────

class ValueHead(nn.Module):
    """Simple MLP value head."""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_features: torch.Tensor) -> torch.Tensor:
        return self.net(global_features).squeeze(-1)  # [batch]


# ─────────────────────────────────────────────────────────────────────────────
# Combined hierarchical policy
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalPolicy(nn.Module):
    """
    Full hierarchical policy that:
      1. Chooses action type (via ActionTypeHead)
      2. Chooses specific action within that type

    Wraps all sub-heads and exposes a unified sample / log_prob / entropy API
    compatible with PPO training.
    """

    ACTION_CUBE     = 0
    ACTION_SEP      = 1
    ACTION_DOCK     = 2
    ACTION_MANEUVER = 3
    ACTION_NOOP     = 4

    def __init__(self,
                 hidden_dim: int = 128,
                 max_cubes: int = 64,
                 max_groups: int = 8,
                 max_separation_actions: int = 100,
                 max_docking_actions: int = 64,
                 max_moves_per_cube: int = 24,
                 num_action_types: int = 5):
        super().__init__()

        self.hidden_dim             = hidden_dim
        self.max_cubes              = max_cubes
        self.max_groups             = max_groups
        self.max_separation_actions = max_separation_actions
        self.max_docking_actions    = max_docking_actions
        self.max_moves_per_cube     = max_moves_per_cube
        self.num_action_types       = num_action_types

        self.action_type_head = ActionTypeHead(hidden_dim, num_action_types)
        self.cube_move_head   = CubeMoveHead(hidden_dim, hidden_dim, max_moves_per_cube)
        self.separation_head  = SeparationHead(hidden_dim, hidden_dim, max_separation_actions)
        self.docking_head     = DockingHead(hidden_dim, hidden_dim, max_groups)
        self.maneuver_head    = ManeuverHead(hidden_dim, hidden_dim, max_groups)
        self.value_head       = ValueHead(hidden_dim)

    # ------------------------------------------------------------------
    # Forward: returns logits for all sub-heads (training use)
    # ------------------------------------------------------------------

    def forward(self,
                global_features: torch.Tensor,
                node_features: torch.Tensor,
                cube_mask_nodes: torch.Tensor,
                group_mask_nodes: torch.Tensor,
                batch: torch.Tensor,
                masks: Dict) -> Dict[str, torch.Tensor]:
        """
        Args:
            global_features   : [B, hidden_dim]
            node_features     : [num_nodes, hidden_dim]
            cube_mask_nodes   : [num_nodes] bool
            group_mask_nodes  : [num_nodes] bool
            batch             : [num_nodes] int
            masks (dict) with:
              'action_type'  : [B, 5] bool
              'cube_move'    : [B, max_cubes, max_moves_per_cube] bool
              'separation'   : [B, max_separation_actions] bool
              'docking'      : [B, max_groups, max_groups] bool
              'maneuver'     : [B, max_groups, 7] bool
              'sep_cube_sets': list[B][action][cube_ids]

        Returns:
            dict with logit tensors + value estimate
        """
        at_logits = self.action_type_head(global_features, masks['action_type'])

        cm_logits = self.cube_move_head(
            node_features, global_features,
            cube_mask_nodes, batch, masks['cube_move'],
        )

        sep_logits = self.separation_head(
            node_features, global_features,
            cube_mask_nodes, batch,
            masks.get('sep_cube_sets', [[] for _ in range(global_features.size(0))]),
            masks['separation'],
        )

        dock_logits = self.docking_head(
            node_features, global_features,
            group_mask_nodes, batch, masks['docking'],
        )

        man_logits = self.maneuver_head(
            node_features, global_features,
            group_mask_nodes, batch, masks['maneuver'],
        )

        value = self.value_head(global_features)

        return {
            'action_type_logits': at_logits,
            'cube_move_logits'  : cm_logits,
            'separation_logits' : sep_logits,
            'docking_logits'    : dock_logits,
            'maneuver_logits'   : man_logits,
            'value'             : value,
        }

    # ------------------------------------------------------------------
    # Sampling (inference / rollout)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_action(self,
                      global_features: torch.Tensor,
                      node_features: torch.Tensor,
                      cube_mask_nodes: torch.Tensor,
                      group_mask_nodes: torch.Tensor,
                      batch: torch.Tensor,
                      masks: Dict,
                      deterministic: bool = False) -> Dict:
        """
        Sample a complete hierarchical action.

        Returns a dict with:
          action_type  : int
          sub_action   : int  (index within the chosen sub-action space)
          log_prob     : float
          value        : float
        """
        out = self.forward(
            global_features, node_features,
            cube_mask_nodes, group_mask_nodes, batch, masks,
        )

        B = global_features.size(0)
        assert B == 1, "sample_action expects single-env input (batch=1)"

        # ── Step 1: action type ──
        at_dist = Categorical(logits=out['action_type_logits'])
        if deterministic:
            action_type = out['action_type_logits'].argmax(dim=-1)
        else:
            action_type = at_dist.sample()

        at_log_prob = at_dist.log_prob(action_type)
        at_int = action_type.item()

        # ── Step 2: sub-action ──
        if at_int == self.ACTION_CUBE:
            logits = out['cube_move_logits']
        elif at_int == self.ACTION_SEP:
            logits = out['separation_logits']
        elif at_int == self.ACTION_DOCK:
            logits = out['docking_logits']
        elif at_int == self.ACTION_MANEUVER:
            logits = out['maneuver_logits']
        else:  # NOOP
            return {
                'action_type': at_int,
                'sub_action' : 0,
                'log_prob'   : at_log_prob.item(),
                'value'      : out['value'].item(),
            }

        sub_dist = Categorical(logits=logits)
        if deterministic:
            sub_action = logits.argmax(dim=-1)
        else:
            sub_action = sub_dist.sample()

        sub_log_prob = sub_dist.log_prob(sub_action)
        total_log_prob = at_log_prob + sub_log_prob

        return {
            'action_type': at_int,
            'sub_action' : sub_action.item(),
            'log_prob'   : total_log_prob.item(),
            'value'      : out['value'].item(),
        }

    # ------------------------------------------------------------------
    # Log-prob / entropy for training
    # ------------------------------------------------------------------

    def evaluate_actions(self,
                         global_features: torch.Tensor,
                         node_features: torch.Tensor,
                         cube_mask_nodes: torch.Tensor,
                         group_mask_nodes: torch.Tensor,
                         batch: torch.Tensor,
                         masks: Dict,
                         stored_action_types: torch.Tensor,
                         stored_sub_actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Re-evaluate stored actions for PPO loss computation.

        Returns:
            log_probs : [B]
            entropy   : [B]
            values    : [B]
        """
        out = self.forward(
            global_features, node_features,
            cube_mask_nodes, group_mask_nodes, batch, masks,
        )

        B = global_features.size(0)
        device = global_features.device

        at_dist     = Categorical(logits=out['action_type_logits'])
        at_log_prob = at_dist.log_prob(stored_action_types)
        at_entropy  = at_dist.entropy()

        # Gather the correct sub-action logits based on each sample's action type
        sub_log_probs = torch.zeros(B, device=device)
        sub_entropies = torch.zeros(B, device=device)

        logit_map = {
            self.ACTION_CUBE    : out['cube_move_logits'],
            self.ACTION_SEP     : out['separation_logits'],
            self.ACTION_DOCK    : out['docking_logits'],
            self.ACTION_MANEUVER: out['maneuver_logits'],
        }

        for at in range(self.ACTION_NOOP):  # Skip NOOP (no sub-action)
            idx = (stored_action_types == at).nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                continue
            logits  = logit_map[at][idx]
            sub_act = stored_sub_actions[idx]
            dist    = Categorical(logits=logits)
            sub_log_probs[idx] = dist.log_prob(sub_act)
            sub_entropies[idx] = dist.entropy()

        total_log_probs = at_log_prob + sub_log_probs
        total_entropy   = at_entropy  + sub_entropies

        return {
            'log_probs': total_log_probs,
            'entropy'  : total_entropy,
            'values'   : out['value'],
        }