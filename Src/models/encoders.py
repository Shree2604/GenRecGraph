"""
Graph Neural Network Encoders for Recommendation Systems

This module implements various GNN architectures for encoding user-item
interaction graphs, including GCN, GAT, and GraphSAGE variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, HeteroConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from typing import Dict, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class GCNEncoder(nn.Module):
    """
    Graph Convolutional Network encoder for bipartite user-item graphs.
    
    Uses multiple GCN layers to learn node embeddings that capture
    both local and higher-order neighborhood information.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: list = [128, 64],
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        """
        Initialize GCN encoder.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            activation: Activation function ('relu', 'elu', 'leaky_relu')
        """
        super(GCNEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        
        # Activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'elu':
            self.activation = F.elu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build GCN layers
        self.convs = nn.ModuleList()
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            self.convs.append(GCNConv(dims[i], dims[i + 1]))
        
        self.dropout_layer = nn.Dropout(dropout)
        
        logger.info(f"Initialized GCN encoder with {len(self.convs)} layers: {dims}")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GCN layers.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, hidden_dims[-1]]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if i < len(self.convs) - 1:  # No activation on last layer
                x = self.activation(x)
                x = self.dropout_layer(x)
        
        return x
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of output embeddings."""
        return self.hidden_dims[-1]


class GATEncoder(nn.Module):
    """
    Graph Attention Network encoder with multi-head attention.
    
    Uses attention mechanisms to dynamically weight neighbor contributions
    during message passing.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: list = [128, 64],
                 num_heads: list = [4, 1],
                 dropout: float = 0.1,
                 attention_dropout: float = 0.1):
        """
        Initialize GAT encoder.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dims: List of hidden layer dimensions
            num_heads: Number of attention heads per layer
            dropout: Dropout probability
            attention_dropout: Attention dropout probability
        """
        super(GATEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_heads = num_heads
        self.dropout = dropout
        
        if len(num_heads) != len(hidden_dims):
            raise ValueError("Number of heads must match number of hidden layers")
        
        # Build GAT layers
        self.convs = nn.ModuleList()
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            heads = num_heads[i]
            out_dim = dims[i + 1]
            
            # For intermediate layers, output dimension is multiplied by heads
            if i < len(dims) - 2:
                conv_out_dim = out_dim // heads
            else:
                conv_out_dim = out_dim
                heads = num_heads[i]
            
            self.convs.append(GATConv(
                dims[i], 
                conv_out_dim,
                heads=heads,
                dropout=attention_dropout,
                concat=(i < len(dims) - 2)  # Concatenate all but last layer
            ))
        
        self.dropout_layer = nn.Dropout(dropout)
        
        logger.info(f"Initialized GAT encoder with {len(self.convs)} layers")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GAT layers.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, hidden_dims[-1]]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if i < len(self.convs) - 1:  # No activation on last layer
                x = F.elu(x)
                x = self.dropout_layer(x)
        
        return x
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of output embeddings."""
        return self.hidden_dims[-1]


class SAGEEncoder(nn.Module):
    """
    GraphSAGE encoder for inductive learning.
    
    Particularly useful for cold-start scenarios where new nodes
    need to be embedded without retraining.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: list = [128, 64],
                 dropout: float = 0.1,
                 aggregator: str = 'mean'):
        """
        Initialize SAGE encoder.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            aggregator: Aggregation function ('mean', 'max', 'lstm')
        """
        super(SAGEEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        
        # Build SAGE layers
        self.convs = nn.ModuleList()
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            self.convs.append(SAGEConv(
                dims[i], 
                dims[i + 1],
                aggr=aggregator
            ))
        
        self.dropout_layer = nn.Dropout(dropout)
        
        logger.info(f"Initialized SAGE encoder with {len(self.convs)} layers")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SAGE layers.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, hidden_dims[-1]]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if i < len(self.convs) - 1:  # No activation on last layer
                x = F.relu(x)
                x = self.dropout_layer(x)
        
        return x
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of output embeddings."""
        return self.hidden_dims[-1]


class HeteroGNNEncoder(nn.Module):
    """
    Heterogeneous GNN encoder for user-item bipartite graphs.
    
    Handles different node types (users and movies) with separate
    message passing and aggregation.
    """
    
    def __init__(self,
                 user_input_dim: int,
                 movie_input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 gnn_type: str = 'gcn'):
        """
        Initialize heterogeneous GNN encoder.
        
        Args:
            user_input_dim: Dimension of user features
            movie_input_dim: Dimension of movie features
            hidden_dim: Hidden dimension for all layers
            num_layers: Number of GNN layers
            gnn_type: Type of GNN ('gcn', 'gat', 'sage')
        """
        super(HeteroGNNEncoder, self).__init__()
        
        self.user_input_dim = user_input_dim
        self.movie_input_dim = movie_input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection layers
        self.user_proj = nn.Linear(user_input_dim, hidden_dim)
        self.movie_proj = nn.Linear(movie_input_dim, hidden_dim)
        
        # Heterogeneous convolution layers
        self.convs = nn.ModuleList()
        
        for _ in range(num_layers):
            if gnn_type == 'gcn':
                conv_dict = {
                    ('user', 'rates', 'movie'): GCNConv(hidden_dim, hidden_dim),
                    ('movie', 'rated_by', 'user'): GCNConv(hidden_dim, hidden_dim)
                }
            elif gnn_type == 'gat':
                conv_dict = {
                    ('user', 'rates', 'movie'): GATConv(hidden_dim, hidden_dim, heads=1),
                    ('movie', 'rated_by', 'user'): GATConv(hidden_dim, hidden_dim, heads=1)
                }
            elif gnn_type == 'sage':
                conv_dict = {
                    ('user', 'rates', 'movie'): SAGEConv(hidden_dim, hidden_dim),
                    ('movie', 'rated_by', 'user'): SAGEConv(hidden_dim, hidden_dim)
                }
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
            
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
        
        logger.info(f"Initialized heterogeneous {gnn_type.upper()} encoder")
    
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through heterogeneous GNN.
        
        Args:
            x_dict: Dictionary of node features by type
            edge_index_dict: Dictionary of edge indices by relation type
            
        Returns:
            Dictionary of node embeddings by type
        """
        # Project input features
        x_dict = {
            'user': self.user_proj(x_dict['user']),
            'movie': self.movie_proj(x_dict['movie'])
        }
        
        # Apply heterogeneous convolutions
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        return x_dict
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of output embeddings."""
        return self.hidden_dim


class BipartiteEncoder(nn.Module):
    """
    Specialized encoder for bipartite user-item graphs.
    
    Separates user and movie embeddings while allowing interaction
    through the bipartite structure.
    """
    
    def __init__(self,
                 input_dim: int,
                 embedding_dim: int = 64,
                 num_users: int = None,
                 num_movies: int = None,
                 gnn_type: str = 'gcn',
                 num_layers: int = 2):
        """
        Initialize bipartite encoder.
        
        Args:
            input_dim: Dimension of input features
            embedding_dim: Dimension of output embeddings
            num_users: Number of users (for splitting embeddings)
            num_movies: Number of movies (for splitting embeddings)
            gnn_type: Type of GNN backbone
            num_layers: Number of GNN layers
        """
        super(BipartiteEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_users = num_users
        self.num_movies = num_movies
        
        # Choose GNN backbone
        if gnn_type == 'gcn':
            self.gnn = GCNEncoder(
                input_dim=input_dim,
                hidden_dims=[embedding_dim] * num_layers
            )
        elif gnn_type == 'gat':
            self.gnn = GATEncoder(
                input_dim=input_dim,
                hidden_dims=[embedding_dim] * num_layers,
                num_heads=[4] * (num_layers - 1) + [1]
            )
        elif gnn_type == 'sage':
            self.gnn = SAGEEncoder(
                input_dim=input_dim,
                hidden_dims=[embedding_dim] * num_layers
            )
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        logger.info(f"Initialized bipartite encoder with {gnn_type.upper()} backbone")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through bipartite encoder.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Dictionary with separate user and movie embeddings
        """
        # Get full node embeddings
        embeddings = self.gnn(x, edge_index)
        
        # Split embeddings by node type
        if self.num_users is not None and self.num_movies is not None:
            user_embeddings = embeddings[:self.num_users]
            movie_embeddings = embeddings[self.num_users:self.num_users + self.num_movies]
            
            return {
                'user': user_embeddings,
                'movie': movie_embeddings,
                'all': embeddings
            }
        else:
            return {'all': embeddings}
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of output embeddings."""
        return self.embedding_dim


def create_encoder(encoder_type: str, 
                  input_dim: int,
                  embedding_dim: int = 64,
                  **kwargs) -> nn.Module:
    """
    Factory function to create different types of encoders.
    
    Args:
        encoder_type: Type of encoder ('gcn', 'gat', 'sage', 'hetero', 'bipartite')
        input_dim: Input feature dimension
        embedding_dim: Output embedding dimension
        **kwargs: Additional arguments for specific encoders
        
    Returns:
        Initialized encoder module
    """
    if encoder_type == 'gcn':
        return GCNEncoder(
            input_dim=input_dim,
            hidden_dims=kwargs.get('hidden_dims', [embedding_dim]),
            dropout=kwargs.get('dropout', 0.1)
        )
    elif encoder_type == 'gat':
        return GATEncoder(
            input_dim=input_dim,
            hidden_dims=kwargs.get('hidden_dims', [embedding_dim]),
            num_heads=kwargs.get('num_heads', [4, 1]),
            dropout=kwargs.get('dropout', 0.1)
        )
    elif encoder_type == 'sage':
        return SAGEEncoder(
            input_dim=input_dim,
            hidden_dims=kwargs.get('hidden_dims', [embedding_dim]),
            dropout=kwargs.get('dropout', 0.1)
        )
    elif encoder_type == 'hetero':
        return HeteroGNNEncoder(
            user_input_dim=kwargs.get('user_input_dim', input_dim),
            movie_input_dim=kwargs.get('movie_input_dim', input_dim),
            hidden_dim=embedding_dim,
            num_layers=kwargs.get('num_layers', 2)
        )
    elif encoder_type == 'bipartite':
        return BipartiteEncoder(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            num_users=kwargs.get('num_users'),
            num_movies=kwargs.get('num_movies'),
            gnn_type=kwargs.get('gnn_type', 'gcn'),
            num_layers=kwargs.get('num_layers', 2)
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
