"""
GenRecGraph: Generative Graph Model for Cold-Start Recommendation

Main model that combines GNN encoders with generative decoders
for addressing cold-start problems in recommendation systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Union
import logging

from .encoders import create_encoder
from .decoders import create_decoder, vae_loss

logger = logging.getLogger(__name__)


class GenRecGraph(nn.Module):
    """
    Main generative graph model for cold-start recommendation.
    
    Combines a GNN encoder for learning node embeddings with a
    generative decoder for predicting and generating user-item interactions.
    """
    
    def __init__(self,
                 encoder_config: Dict,
                 decoder_config: Dict,
                 num_users: int,
                 num_movies: int,
                 input_dim: int,
                 embedding_dim: int = 64):
        """
        Initialize GenRecGraph model.
        
        Args:
            encoder_config: Configuration for encoder
            decoder_config: Configuration for decoder
            num_users: Number of users in the dataset
            num_movies: Number of movies in the dataset
            input_dim: Dimension of input node features
            embedding_dim: Dimension of learned embeddings
        """
        super(GenRecGraph, self).__init__()
        
        self.num_users = num_users
        self.num_movies = num_movies
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Create encoder
        encoder_type = encoder_config.pop('type', 'gcn')
        self.encoder = create_encoder(
            encoder_type=encoder_type,
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            num_users=num_users,
            num_movies=num_movies,
            **encoder_config
        )
        
        # Create decoder
        decoder_type = decoder_config.pop('type', 'inner_product')
        self.decoder = create_decoder(
            decoder_type=decoder_type,
            embedding_dim=embedding_dim,
            **decoder_config
        )
        
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        
        logger.info(f"Initialized GenRecGraph with {encoder_type} encoder "
                   f"and {decoder_type} decoder")
    
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode nodes to embeddings using GNN encoder.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Dictionary with node embeddings
        """
        if self.encoder_type in ['hetero']:
            # For heterogeneous encoders, need to split inputs
            x_dict = {
                'user': x[:self.num_users],
                'movie': x[self.num_users:self.num_users + self.num_movies]
            }
            
            edge_index_dict = {
                ('user', 'rates', 'movie'): edge_index,
                ('movie', 'rated_by', 'user'): torch.flip(edge_index, [0])
            }
            
            embeddings = self.encoder(x_dict, edge_index_dict)
        else:
            embeddings = self.encoder(x, edge_index)
        
        return embeddings
    
    def decode(self, embeddings: Dict[str, torch.Tensor],
               user_indices: torch.Tensor,
               movie_indices: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decode embeddings to edge predictions.
        
        Args:
            embeddings: Node embeddings from encoder
            user_indices: User indices for edges to predict
            movie_indices: Movie indices for edges to predict
            
        Returns:
            Edge predictions or dictionary with VAE outputs
        """
        # Extract user and movie embeddings
        if isinstance(embeddings, dict):
            if 'user' in embeddings and 'movie' in embeddings:
                user_emb = embeddings['user'][user_indices]
                movie_emb = embeddings['movie'][movie_indices]
            else:
                # Bipartite encoder output
                all_emb = embeddings['all']
                user_emb = all_emb[user_indices]
                movie_emb = all_emb[self.num_users + movie_indices]
        else:
            # Homogeneous encoder output
            user_emb = embeddings[user_indices]
            movie_emb = embeddings[self.num_users + movie_indices]
        
        # Decode to edge predictions
        return self.decoder(user_emb, movie_emb)
    
    def forward(self, 
               x: torch.Tensor,
               edge_index: torch.Tensor,
               user_indices: torch.Tensor,
               movie_indices: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the complete model.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            user_indices: User indices for prediction
            movie_indices: Movie indices for prediction
            
        Returns:
            Edge predictions or VAE outputs
        """
        # Encode nodes to embeddings
        embeddings = self.encode(x, edge_index)
        
        # Decode to edge predictions
        predictions = self.decode(embeddings, user_indices, movie_indices)
        
        return predictions
    
    def generate_recommendations(self,
                               x: torch.Tensor,
                               edge_index: torch.Tensor,
                               user_indices: torch.Tensor,
                               num_recommendations: int = 10,
                               exclude_seen: bool = True) -> Dict[int, List[int]]:
        """
        Generate recommendations for specified users.
        
        Args:
            x: Node features
            edge_index: Edge indices
            user_indices: Users to generate recommendations for
            num_recommendations: Number of recommendations per user
            exclude_seen: Whether to exclude already seen movies
            
        Returns:
            Dictionary mapping user indices to recommended movie lists
        """
        self.eval()
        
        with torch.no_grad():
            # Get embeddings
            embeddings = self.encode(x, edge_index)
            
            recommendations = {}
            
            for user_idx in user_indices:
                # Get all movie indices
                all_movie_indices = torch.arange(self.num_movies, device=x.device)
                user_batch = torch.full_like(all_movie_indices, user_idx)
                
                # Predict scores for all movies
                if self.decoder_type == 'vae':
                    outputs = self.decode(embeddings, user_batch, all_movie_indices)
                    scores = outputs['reconstruction'].squeeze()
                else:
                    scores = self.decode(embeddings, user_batch, all_movie_indices).squeeze()
                
                # Exclude seen movies if requested
                if exclude_seen:
                    # Find movies this user has interacted with
                    user_edges = edge_index[:, edge_index[0] == user_idx]
                    seen_movies = user_edges[1] - self.num_users  # Adjust for bipartite indexing
                    seen_movies = seen_movies[seen_movies >= 0]  # Only movie nodes
                    seen_movies = seen_movies[seen_movies < self.num_movies]  # Valid range
                    
                    if len(seen_movies) > 0:
                        scores[seen_movies] = -float('inf')
                
                # Get top recommendations
                _, top_indices = torch.topk(scores, min(num_recommendations, len(scores)))
                recommendations[user_idx.item()] = top_indices.cpu().tolist()
        
        return recommendations
    
    def generate_cold_start_edges(self,
                                x: torch.Tensor,
                                edge_index: torch.Tensor,
                                cold_start_users: List[int],
                                cold_start_movies: List[int],
                                num_edges_per_user: int = 5) -> List[Tuple[int, int, float]]:
        """
        Generate edges for cold-start users and movies.
        
        Args:
            x: Node features
            edge_index: Existing edge indices
            cold_start_users: List of cold-start user indices
            cold_start_movies: List of cold-start movie indices
            num_edges_per_user: Number of edges to generate per user
            
        Returns:
            List of (user_idx, movie_idx, score) tuples
        """
        self.eval()
        
        generated_edges = []
        
        with torch.no_grad():
            embeddings = self.encode(x, edge_index)
            
            for user_idx in cold_start_users:
                # Consider all movies (both cold-start and existing)
                movie_candidates = torch.arange(self.num_movies, device=x.device)
                user_batch = torch.full_like(movie_candidates, user_idx)
                
                # Generate scores
                if self.decoder_type == 'vae':
                    outputs = self.decode(embeddings, user_batch, movie_candidates)
                    scores = outputs['reconstruction'].squeeze()
                else:
                    scores = self.decode(embeddings, user_batch, movie_candidates).squeeze()
                
                # Get top scoring edges
                _, top_indices = torch.topk(scores, min(num_edges_per_user, len(scores)))
                
                for movie_idx in top_indices:
                    score = scores[movie_idx].item()
                    generated_edges.append((user_idx, movie_idx.item(), score))
        
        return generated_edges
    
    def compute_loss(self,
                    predictions: Union[torch.Tensor, Dict[str, torch.Tensor]],
                    targets: torch.Tensor,
                    loss_type: str = 'bce') -> Dict[str, torch.Tensor]:
        """
        Compute loss for training.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            loss_type: Type of loss ('bce', 'mse', 'vae')
            
        Returns:
            Dictionary with loss components
        """
        if self.decoder_type == 'vae' and isinstance(predictions, dict):
            # VAE loss
            return vae_loss(
                predictions['reconstruction'],
                targets,
                predictions['mean'],
                predictions['logvar'],
                beta=1.0
            )
        else:
            # Standard loss
            if isinstance(predictions, dict):
                predictions = predictions['reconstruction']
            
            if loss_type == 'bce':
                loss = F.binary_cross_entropy(predictions.squeeze(), targets.float())
            elif loss_type == 'mse':
                loss = F.mse_loss(predictions.squeeze(), targets.float())
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
            
            return {'total_loss': loss}


class ColdStartGenRecGraph(GenRecGraph):
    """
    Specialized version of GenRecGraph for cold-start scenarios.
    
    Includes additional mechanisms for handling new users and items
    that weren't seen during training.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Additional components for cold-start handling
        self.user_feature_proj = nn.Linear(self.input_dim, self.embedding_dim)
        self.movie_feature_proj = nn.Linear(self.input_dim, self.embedding_dim)
        
    def handle_cold_start_users(self,
                              user_features: torch.Tensor,
                              existing_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Generate embeddings for cold-start users using their features.
        
        Args:
            user_features: Features for cold-start users
            existing_embeddings: Embeddings from the main model
            
        Returns:
            Cold-start user embeddings
        """
        # Project features to embedding space
        cold_start_emb = self.user_feature_proj(user_features)
        
        # Optional: blend with similar existing users
        if 'user' in existing_embeddings:
            # Find most similar existing users
            similarities = torch.mm(cold_start_emb, existing_embeddings['user'].t())
            weights = F.softmax(similarities, dim=1)
            
            # Weighted combination
            blended_emb = torch.mm(weights, existing_embeddings['user'])
            cold_start_emb = 0.7 * cold_start_emb + 0.3 * blended_emb
        
        return cold_start_emb
    
    def handle_cold_start_movies(self,
                               movie_features: torch.Tensor,
                               existing_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Generate embeddings for cold-start movies using their features.
        
        Args:
            movie_features: Features for cold-start movies
            existing_embeddings: Embeddings from the main model
            
        Returns:
            Cold-start movie embeddings
        """
        # Project features to embedding space
        cold_start_emb = self.movie_feature_proj(movie_features)
        
        # Optional: blend with similar existing movies
        if 'movie' in existing_embeddings:
            # Find most similar existing movies
            similarities = torch.mm(cold_start_emb, existing_embeddings['movie'].t())
            weights = F.softmax(similarities, dim=1)
            
            # Weighted combination
            blended_emb = torch.mm(weights, existing_embeddings['movie'])
            cold_start_emb = 0.7 * cold_start_emb + 0.3 * blended_emb
        
        return cold_start_emb


def create_genrecgraph_model(config: Dict) -> GenRecGraph:
    """
    Factory function to create GenRecGraph model from configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized GenRecGraph model
    """
    model_config = config['model']
    data_config = config['data']
    
    if model_config.get('cold_start_mode', False):
        model_class = ColdStartGenRecGraph
    else:
        model_class = GenRecGraph
    
    return model_class(
        encoder_config=model_config['encoder'],
        decoder_config=model_config['decoder'],
        num_users=data_config['num_users'],
        num_movies=data_config['num_movies'],
        input_dim=data_config['input_dim'],
        embedding_dim=model_config.get('embedding_dim', 64)
    )
