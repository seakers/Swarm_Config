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

from rl.gnn_encoder import ConstellationObservationEncoder

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
    
    def forward(self, global_features: torch.Tensor, 
                action_type_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_features: [batch, input_dim]
            action_type_mask: [batch, num_action_types] bool mask
        
        Returns:
            logits: [batch, num_action_types]
        """
        logits = self.net(global_features)
        
        # Apply mask (set masked actions to large negative value)
        logits = logits.masked_fill(~action_type_mask, -1e9)
        
        return logits


class CubeMoveHead(nn.Module):
    """
    Selects which cube move to execute.
    
    Uses node features to score each cube, then scores each valid move.
    """
    
    def __init__(self, 
                 node_dim: int = 128,
                 global_dim: int = 128,
                 max_moves_per_cube: int = 24):  # 12 edges * 2 directions
        super().__init__()
        
        self.max_moves_per_cube = max_moves_per_cube
        
        # Combine node and global features
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
            node_features: [num_nodes, node_dim]
            global_features: [batch_size, global_dim]
            cube_mask: [num_nodes] bool - which nodes are cubes
            batch: [num_nodes] batch assignment
            move_mask: [batch_size, max_cubes, max_moves_per_cube] bool
        
        Returns:
            logits: [batch_size, max_cubes * max_moves_per_cube]
        """
        batch_size = global_features.size(0)
        
        # Get cube nodes only
        cube_nodes = node_features[cube_mask]  # [num_cube_nodes, node_dim]
        # Handle case where batch is None (single graph, not batched)
        if batch is None:
            cube_batch = torch.zeros(cube_mask.sum(), dtype=torch.long, device=node_features.device)
        else:
            cube_batch = batch[cube_mask]  # [num_cube_nodes]
        
        # Expand global features to match cube nodes
        global_expanded = global_features[cube_batch]  # [num_cube_nodes, global_dim]
        
        # Concatenate and score
        combined = torch.cat([cube_nodes, global_expanded], dim=-1)
        scores = self.node_scorer(combined)  # [num_cube_nodes, max_moves_per_cube]
        
        # Reshape to batch form - need to handle variable num cubes per batch
        # For simplicity, we'll pad to max_cubes
        max_cubes = move_mask.size(1)
        logits = torch.full(
            (batch_size, max_cubes, self.max_moves_per_cube),
            -1e9,
            device=node_features.device
        )

        # Scatter scores into the padded tensor based on batch assignment
        for i, (node_idx, batch_idx) in enumerate(zip(torch.where(cube_mask)[0], cube_batch)):
            # Find which cube index within this batch
            # This requires knowing the cube ordering within each batch
            cube_idx_in_batch = (cube_batch[:i+1] == batch_idx).sum() - 1
            if cube_idx_in_batch < max_cubes:
                logits[batch_idx, cube_idx_in_batch] = scores[i]
        
        # Flatten to [batch_size, max_cubes * max_moves_per_cube]
        logits_flat = logits.view(batch_size, -1)
        
        # Apply move mask
        move_mask_flat = move_mask.view(batch_size, -1)
        logits_flat = logits_flat.masked_fill(~move_mask_flat, -1e9)
        
        return logits_flat


class SeparationHead(nn.Module):
    """
    Selects which separation action to execute.
    
    Uses attention over cube nodes to score candidate separations.
    """
    
    def __init__(self, 
                 node_dim: int = 128,
                 global_dim: int = 128,
                 max_separations: int = 100):
        super().__init__()
        
        self.max_separations = max_separations
        
        # Score each candidate separation
        # Input: global features + aggregated features of separation set
        self.separation_scorer = nn.Sequential(
            nn.Linear(global_dim + node_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        
        # Attention to aggregate cube features for a separation set
        self.cube_attention = nn.MultiheadAttention(
            embed_dim=node_dim,
            num_heads=4,
            batch_first=True,
        )
        
    def forward(self,
                node_features: torch.Tensor,
                global_features: torch.Tensor,
                separation_cube_masks: torch.Tensor,
                separation_valid_mask: torch.Tensor,
                cube_mask: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: [num_nodes, node_dim]
            global_features: [batch_size, global_dim]
            separation_cube_masks: [batch_size, max_separations, max_cubes] bool
                                   Which cubes are in each separation set
            separation_valid_mask: [batch_size, max_separations] bool
                                   Which separations are valid
            cube_mask: [num_nodes] bool - which nodes are cubes
            batch: [num_nodes] batch assignment
            
        Returns:
            logits: [batch_size, max_separations]
        """
        batch_size = global_features.size(0)
        device = node_features.device
        
        logits = torch.full(
            (batch_size, self.max_separations),
            -1e9,
            device=device
        )
        
        # Get cube nodes
        cube_nodes = node_features[cube_mask]  # [num_cube_nodes, node_dim]
        # Handle case where batch is None (single graph, not batched)
        if batch is None:
            cube_batch = torch.zeros(cube_mask.sum(), dtype=torch.long, device=node_features.device)
        else:
            cube_batch = batch[cube_mask]  # [num_cube_nodes]
        
        # For each batch, for each valid separation, aggregate cube features
        for b in range(batch_size):
            # Get cube features for this batch
            batch_cube_mask = (cube_batch == b)
            batch_cube_features = cube_nodes[batch_cube_mask]  # [num_cubes_in_batch, node_dim]
            
            num_cubes_in_batch = batch_cube_features.size(0)
            
            for s in range(self.max_separations):
                if not separation_valid_mask[b, s]:
                    continue
                
                # Get which cubes are in this separation
                sep_cube_mask = separation_cube_masks[b, s, :num_cubes_in_batch]
                
                if sep_cube_mask.sum() == 0:
                    continue
                
                # Aggregate features of cubes in separation set
                sep_cube_features = batch_cube_features[sep_cube_mask]  # [num_sep_cubes, node_dim]
                
                # Mean pooling (could use attention here too)
                sep_aggregate = sep_cube_features.mean(dim=0)  # [node_dim]
                
                # Combine with global features and score
                combined = torch.cat([global_features[b], sep_aggregate], dim=-1)
                score = self.separation_scorer(combined)
                
                logits[b, s] = score.squeeze()
        
        return logits


class DockingHead(nn.Module):
    """
    Selects which docking action to execute.
    
    Scores pairs of groups for docking.
    """
    
    def __init__(self,
                 node_dim: int = 128,
                 global_dim: int = 128,
                 max_groups: int = 8):
        super().__init__()
        
        self.max_groups = max_groups
        self.max_docking_pairs = max_groups * (max_groups - 1) // 2
        
        # Score each group pair
        self.pair_scorer = nn.Sequential(
            nn.Linear(node_dim * 2 + global_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    
    def forward(self,
                node_features: torch.Tensor,
                global_features: torch.Tensor,
                group_mask: torch.Tensor,
                docking_valid_mask: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: [num_nodes, node_dim]
            global_features: [batch_size, global_dim]
            group_mask: [num_nodes] bool - which nodes are groups
            docking_valid_mask: [batch_size, max_docking_pairs] bool
            batch: [num_nodes] batch assignment
            
        Returns:
            logits: [batch_size, max_docking_pairs]
        """
        batch_size = global_features.size(0)
        device = node_features.device
        
        logits = torch.full(
            (batch_size, self.max_docking_pairs),
            -1e9,
            device=device
        )
        
        # Get group nodes
        group_nodes = node_features[group_mask]
        # Handle case where batch is None (single graph, not batched)
        if batch is None:
            group_batch = torch.zeros(group_mask.sum(), dtype=torch.long, device=node_features.device)
        else:
            group_batch = batch[group_mask]  # [num_group_nodes]
        
        for b in range(batch_size):
            batch_group_mask = (group_batch == b)
            batch_group_features = group_nodes[batch_group_mask]
            num_groups = batch_group_features.size(0)
            
            pair_idx = 0
            for i in range(num_groups):
                for j in range(i + 1, num_groups):
                    if pair_idx >= self.max_docking_pairs:
                        break
                    
                    if not docking_valid_mask[b, pair_idx]:
                        pair_idx += 1
                        continue
                    
                    # Combine features of both groups with global
                    combined = torch.cat([
                        batch_group_features[i],
                        batch_group_features[j],
                        global_features[b]
                    ], dim=-1)
                    
                    score = self.pair_scorer(combined)
                    logits[b, pair_idx] = score.squeeze()
                    
                    pair_idx += 1
        
        return logits


class ManeuverHead(nn.Module):
    """
    Selects maneuver action (group and direction).
    
    Outputs discrete maneuver direction per group.
    """
    
    def __init__(self,
                 node_dim: int = 128,
                 global_dim: int = 128,
                 max_groups: int = 8,
                 num_directions: int = 7):  # 6 cardinal + zero
        super().__init__()
        
        self.max_groups = max_groups
        self.num_directions = num_directions
        
        # Per-group direction scorer
        self.direction_scorer = nn.Sequential(
            nn.Linear(node_dim + global_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_directions),
        )
    
    def forward(self,
                node_features: torch.Tensor,
                global_features: torch.Tensor,
                group_mask: torch.Tensor,
                maneuver_valid_mask: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: [num_nodes, node_dim]
            global_features: [batch_size, global_dim]
            group_mask: [num_nodes] bool - which nodes are groups
            maneuver_valid_mask: [batch_size, max_groups * num_directions] bool
            batch: [num_nodes] batch assignment
            
        Returns:
            logits: [batch_size, max_groups * num_directions]
        """
        batch_size = global_features.size(0)
        device = node_features.device
        
        logits = torch.full(
            (batch_size, self.max_groups * self.num_directions),
            -1e9,
            device=device
        )
        
        group_nodes = node_features[group_mask]
        # Handle case where batch is None (single graph, not batched)
        if batch is None:
            group_batch = torch.zeros(group_mask.sum(), dtype=torch.long, device=node_features.device)
        else:
            group_batch = batch[group_mask]  # [num_group_nodes]
        
        for b in range(batch_size):
            batch_group_mask = (group_batch == b)
            batch_group_features = group_nodes[batch_group_mask]
            num_groups = batch_group_features.size(0)
            
            for g in range(min(num_groups, self.max_groups)):
                # Score all directions for this group
                combined = torch.cat([
                    batch_group_features[g],
                    global_features[b]
                ], dim=-1)
                
                direction_scores = self.direction_scorer(combined)  # [num_directions]
                
                # Place in output
                start_idx = g * self.num_directions
                end_idx = start_idx + self.num_directions
                
                logits[b, start_idx:end_idx] = direction_scores
        
        # Apply mask
        logits = logits.masked_fill(~maneuver_valid_mask, -1e9)
        
        return logits


class HierarchicalPolicy(nn.Module):
    """
    Complete hierarchical policy for constellation control.
    
    First selects action type, then selects specific action within that type.
    """
    
    def __init__(self,
                 encoder: 'ConstellationObservationEncoder',
                 hidden_dim: int = 128,
                 max_cubes: int = 64,
                 max_groups: int = 8,
                 max_separations: int = 100,
                 num_maneuver_directions: int = 7):
        super().__init__()
        
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.max_cubes = max_cubes
        self.max_groups = max_groups
        
        # Action type head
        self.action_type_head = ActionTypeHead(hidden_dim, num_action_types=5)
        
        # Sub-action heads
        self.cube_move_head = CubeMoveHead(hidden_dim, hidden_dim, max_moves_per_cube=24)
        self.separation_head = SeparationHead(hidden_dim, hidden_dim, max_separations)
        self.docking_head = DockingHead(hidden_dim, hidden_dim, max_groups)
        self.maneuver_head = ManeuverHead(hidden_dim, hidden_dim, max_groups, num_maneuver_directions)
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        
        # Action dimensions
        self.num_cube_actions = max_cubes * 24
        self.num_separation_actions = max_separations
        self.num_docking_actions = max_groups * (max_groups - 1) // 2
        self.num_maneuver_actions = max_groups * num_maneuver_directions
        
    def forward(self, 
                graph_data,
                mode_idx: torch.Tensor,
                env_features: torch.Tensor,
                action_masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing action logits and value.
        
        Args:
            graph_data: PyG Batch with swarm graph
            mode_idx: [batch] mission mode indices
            env_features: [batch, env_dim] environmental features
            action_masks: Dict with masks for each action type
            
        Returns:
            Dict with 'action_type_logits', 'sub_action_logits', 'value'
        """
        # Encode observation
        global_features, node_features = self.encoder(graph_data, mode_idx, env_features)
        
        batch = graph_data.batch
        node_type = graph_data.node_type
        
        cube_mask = (node_type == 0)
        group_mask = (node_type == 1)
        
        # Action type logits
        action_type_logits = self.action_type_head(
            global_features, 
            action_masks['action_type']
        )
        
        # Sub-action logits for each type
        cube_move_logits = self.cube_move_head(
            node_features, global_features,
            cube_mask, batch,
            action_masks['cube_move']
        )
        
        separation_logits = self.separation_head(
            node_features, global_features,
            action_masks['separation_cube_masks'],
            action_masks['separation_valid'],
            cube_mask, batch
        )
        
        docking_logits = self.docking_head(
            node_features, global_features,
            group_mask,
            action_masks['docking'],
            batch
        )
        
        maneuver_logits = self.maneuver_head(
            node_features, global_features,
            group_mask,
            action_masks['maneuver'],
            batch
        )
        
        # Value estimate
        value = self.value_head(global_features)
        
        return {
            'action_type_logits': action_type_logits,
            'cube_move_logits': cube_move_logits,
            'separation_logits': separation_logits,
            'docking_logits': docking_logits,
            'maneuver_logits': maneuver_logits,
            'value': value,
            'global_features': global_features,
            'node_features': node_features,
        }
    
    def get_action_and_value(self,
                              graph_data,
                              mode_idx: torch.Tensor,
                              env_features: torch.Tensor,
                              action_masks: Dict[str, torch.Tensor],
                              deterministic: bool = False):
        """
        Sample action and compute log probability and value.
        
        Returns:
            action_type, sub_action, log_prob, entropy, value
        """
        outputs = self.forward(graph_data, mode_idx, env_features, action_masks)
        
        # Sample action type
        action_type_dist = Categorical(logits=outputs['action_type_logits'])
        
        if deterministic:
            action_type = outputs['action_type_logits'].argmax(dim=-1)
        else:
            action_type = action_type_dist.sample()
        
        action_type_log_prob = action_type_dist.log_prob(action_type)
        action_type_entropy = action_type_dist.entropy()
        
        # Sample sub-action based on action type
        batch_size = action_type.size(0)
        sub_actions = torch.zeros(batch_size, dtype=torch.long, device=action_type.device)
        sub_action_log_probs = torch.zeros(batch_size, device=action_type.device)
        sub_action_entropies = torch.zeros(batch_size, device=action_type.device)
        
        for b in range(batch_size):
            at = action_type[b].item()
            
            if at == 0:  # Cube move
                logits = outputs['cube_move_logits'][b]
            elif at == 1:  # Separation
                logits = outputs['separation_logits'][b]
            elif at == 2:  # Docking
                logits = outputs['docking_logits'][b]
            elif at == 3:  # Maneuver
                logits = outputs['maneuver_logits'][b]
            else:  # Noop
                sub_actions[b] = 0
                sub_action_log_probs[b] = 0.0
                sub_action_entropies[b] = 0.0
                continue
            
            sub_dist = Categorical(logits=logits)
            
            if deterministic:
                sub_action = logits.argmax()
            else:
                sub_action = sub_dist.sample()
            
            sub_actions[b] = sub_action
            sub_action_log_probs[b] = sub_dist.log_prob(sub_action)
            sub_action_entropies[b] = sub_dist.entropy()
        
        # Total log prob and entropy
        total_log_prob = action_type_log_prob + sub_action_log_probs
        total_entropy = action_type_entropy + sub_action_entropies
        
        return {
            'action_type': action_type,
            'sub_action': sub_actions,
            'log_prob': total_log_prob,
            'entropy': total_entropy,
            'value': outputs['value'].squeeze(-1),
        }