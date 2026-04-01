"""
gnn_encoder.py
==============
Graph Neural Network encoder for variable-size spacecraft swarm observations.

Uses PyTorch Geometric for efficient graph operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from typing import Dict, List, Tuple, Optional
import numpy as np


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer for goal conditioning.
    
    Applies affine transformation based on conditioning signal:
        output = gamma * input + beta
    
    where gamma and beta are learned functions of the conditioning signal.
    """
    
    def __init__(self, feature_dim: int, condition_dim: int):
        super().__init__()
        self.gamma_net = nn.Linear(condition_dim, feature_dim)
        self.beta_net = nn.Linear(condition_dim, feature_dim)
        
        # Initialize to identity transformation
        nn.init.ones_(self.gamma_net.weight.data * 0.01)
        nn.init.zeros_(self.gamma_net.bias.data)
        nn.init.zeros_(self.beta_net.weight.data)
        nn.init.zeros_(self.beta_net.bias.data)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [batch, features] or [num_nodes, features]
            condition: Conditioning signal [batch, condition_dim]
        """
        gamma = self.gamma_net(condition) + 1.0  # Start near identity
        beta = self.beta_net(condition)
        
        if x.dim() == 2 and condition.dim() == 2:
            # Need to broadcast condition to match x
            if x.size(0) != condition.size(0):
                # x is [num_nodes, features], condition is [batch, condition_dim]
                # This happens in GNN - we need batch assignment
                return x * gamma + beta
        
        return x * gamma + beta


class CubeNodeEncoder(nn.Module):
    """
    Encodes per-cube features into node embeddings.
    
    Input features per cube:
    - Relative position (3)
    - Orientation (6 or 9 values)
    - Face alignment scores (4: solar, antenna, camera, radiator)
    - Face exposure mask (6)
    - Delta-v remaining (1)
    - Connectivity info (2: num_connections, is_boundary)
    """
    
    def __init__(self, 
                 input_dim: int = 25,
                 hidden_dim: int = 64,
                 output_dim: int = 64):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class GroupNodeEncoder(nn.Module):
    """
    Encodes per-group features into node embeddings.
    
    Input features per group:
    - Position (3)
    - Velocity (3)
    - Num cubes (1)
    - Delta-v remaining (1)
    - Communication state (variable, encoded as fixed)
    """
    
    def __init__(self,
                 input_dim: int = 12,
                 hidden_dim: int = 64,
                 output_dim: int = 64):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class GoalEncoder(nn.Module):
    """
    Encodes goal/task information into a conditioning vector.
    
    Goal features:
    - Mission mode (one-hot or embedding)
    - Target values (solar efficiency, data rate, etc.)
    - Environmental context (sun/earth directions, distance)
    - Constraints
    """
    
    def __init__(self,
                 num_modes: int = 6,
                 env_dim: int = 12,
                 output_dim: int = 64):
        super().__init__()
        
        self.mode_embedding = nn.Embedding(num_modes, 16)
        
        total_input = 16 + env_dim  # mode embedding + env features
        
        self.mlp = nn.Sequential(
            nn.Linear(total_input, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )
    
    def forward(self, mode_idx: torch.Tensor, env_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mode_idx: [batch] integer mode indices
            env_features: [batch, env_dim] environmental features
        """
        mode_emb = self.mode_embedding(mode_idx)  # [batch, 16]
        combined = torch.cat([mode_emb, env_features], dim=-1)
        return self.mlp(combined)


class SwarmGNN(nn.Module):
    """
    Graph Neural Network for processing swarm observations.
    
    Architecture:
    1. Encode cube nodes and group nodes separately
    2. Apply message passing with attention
    3. Apply FiLM conditioning from goal
    4. Pool to get global representation
    
    Graph structure:
    - Cube nodes (type 0): Physical cube states
    - Group nodes (type 1): Group-level states  
    - Global node (type 2): Overall context
    
    Edges:
    - Cube-to-cube: Physical connections
    - Cube-to-group: Membership
    - Group-to-group: Communication links
    """
    
    def __init__(self,
                 cube_input_dim: int = 25,
                 group_input_dim: int = 12,
                 hidden_dim: int = 128,
                 num_heads: int = 4,
                 num_layers: int = 3,
                 goal_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Node encoders
        self.cube_encoder = CubeNodeEncoder(cube_input_dim, hidden_dim, hidden_dim)
        self.group_encoder = GroupNodeEncoder(group_input_dim, hidden_dim, hidden_dim)
        
        # Node type embedding
        self.node_type_embedding = nn.Embedding(3, hidden_dim)  # cube, group, global
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.film_layers = nn.ModuleList()
        
        for i in range(num_layers):
            # GATv2 for better expressivity
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    add_self_loops=True,
                    concat=True,
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            self.film_layers.append(FiLMLayer(hidden_dim, goal_dim))
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for mean+max pool
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                data: Data,
                goal_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the GNN.
        
        Args:
            data: PyG Data object with:
                - x: Node features [num_nodes, feature_dim]
                - edge_index: Edge connectivity [2, num_edges]
                - node_type: Type of each node [num_nodes]
                - batch: Batch assignment [num_nodes]
            goal_embedding: Goal conditioning [batch_size, goal_dim]
        
        Returns:
            global_features: [batch_size, hidden_dim] - Global representation
            node_features: [num_nodes, hidden_dim] - Per-node features
        """
        x = data.x
        edge_index = data.edge_index
        node_type = data.node_type
        batch = data.batch
        
        # Get batch size
        batch_size = goal_embedding.size(0)
        
        # Encode nodes based on type
        # We'll handle this by processing different node types and combining
        cube_mask = (node_type == 0)
        group_mask = (node_type == 1)
        global_mask = (node_type == 2)
        
        # Initialize node embeddings
        h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        
        # Encode cube nodes
        if cube_mask.any():
            cube_features = x[cube_mask, :25]  # First 25 features are cube features
            h[cube_mask] = self.cube_encoder(cube_features)
        
        # Encode group nodes  
        if group_mask.any():
            group_features = x[group_mask, :12]  # First 12 features are group features
            h[group_mask] = self.group_encoder(group_features)
        
        # Global nodes start as zeros (will aggregate info)
        # Add node type embedding
        h = h + self.node_type_embedding(node_type)
        
        # Expand goal_embedding to match nodes
        # goal_embedding is [batch_size, goal_dim]
        # We need [num_nodes, goal_dim] based on batch assignment
        goal_per_node = goal_embedding[batch]  # [num_nodes, goal_dim]
        
        # Message passing with FiLM conditioning
        for i in range(self.num_layers):
            # Graph attention
            h_new = self.gat_layers[i](h, edge_index)
            
            # Residual + LayerNorm
            h = self.layer_norms[i](h + self.dropout(h_new))
            
            # FiLM conditioning
            h = self.film_layers[i](h, goal_per_node)
            
            h = F.relu(h).squeeze()
        
        # Global pooling (mean + max for richer representation)
        h_mean = global_mean_pool(h, batch)  # [batch_size, hidden_dim]
        h_max = global_max_pool(h, batch)    # [batch_size, hidden_dim]
        
        global_features = self.output_proj(torch.cat([h_mean, h_max], dim=-1))
        
        return global_features, h


class ConstellationObservationEncoder(nn.Module):
    """
    Full observation encoder combining GNN with goal conditioning.
    
    This is the main encoder used by the policy and value networks.
    """
    
    def __init__(self,
                 cube_input_dim: int = 25,
                 group_input_dim: int = 12,
                 hidden_dim: int = 128,
                 num_gnn_layers: int = 3,
                 num_attention_heads: int = 4,
                 num_mission_modes: int = 6,
                 env_feature_dim: int = 12,
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Goal encoder
        self.goal_encoder = GoalEncoder(
            num_modes=num_mission_modes,
            env_dim=env_feature_dim,
            output_dim=64,
        )
        
        # GNN encoder
        self.gnn = SwarmGNN(
            cube_input_dim=cube_input_dim,
            group_input_dim=group_input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads,
            num_layers=num_gnn_layers,
            goal_dim=64,
            dropout=dropout,
        )
    
    def forward(self,
                graph_data: Data,
                mode_idx: torch.Tensor,
                env_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode observation.
        
        Args:
            graph_data: PyG Data/Batch with swarm graph
            mode_idx: [batch] mission mode indices
            env_features: [batch, env_dim] environmental features
        
        Returns:
            global_features: [batch, hidden_dim]
            node_features: [num_nodes, hidden_dim]
        """
        # Encode goal
        goal_embedding = self.goal_encoder(mode_idx, env_features)
        
        # Encode graph with goal conditioning
        global_features, node_features = self.gnn(graph_data, goal_embedding)
        
        return global_features, node_features