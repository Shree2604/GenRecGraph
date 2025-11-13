"""
Recommendation Engine for GenRecGraph

This module provides a recommendation engine that uses the trained GenRecGraph model
to generate personalized recommendations while handling cold-start scenarios.
"""
import os
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from torch_geometric.data import Data
import torch.nn.functional as F

class RecommendationEngine:
    def __init__(self, config: dict):
        """
        Initialize the recommendation engine.
        
        Args:
            config: Configuration dictionary containing model paths and parameters
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.graph = None
        self.user_id_map = None  # Maps external user IDs to internal indices
        self.movie_id_map = None  # Maps external movie IDs to internal indices
        self.reverse_movie_map = None  # Maps internal indices back to external movie IDs
        
        # Load models and data
        self._load_models()
        self._load_graph_data()
    
    def _load_models(self):
        """Load the pre-trained encoder and decoder models."""
        from ..models.encoders import create_encoder
        from ..models.decoders import create_decoder
        
        # Create and load encoder
        self.encoder = create_encoder(
            encoder_type=self.config['encoder']['type'],
            input_dim=self.config['encoder']['input_dim'],
            hidden_dims=self.config['encoder']['hidden_dims'],
            output_dim=self.config['encoder']['output_dim'],
            dropout=self.config['encoder'].get('dropout', 0.1)
        ).to(self.device)
        
        # Load encoder weights
        encoder_weights = torch.load(self.config['model_paths']['encoder'], map_location=self.device)
        if 'model_state_dict' in encoder_weights:
            self.encoder.load_state_dict(encoder_weights['model_state_dict'])
        else:
            self.encoder.load_state_dict(encoder_weights)
        self.encoder.eval()
        
        # Create and load decoder
        self.decoder = create_decoder(
            decoder_type=self.config['decoder']['type'],
            embedding_dim=self.config['encoder']['output_dim'],
            hidden_dims=self.config['decoder'].get('hidden_dims', [128, 64]),
            dropout=self.config['decoder'].get('dropout', 0.1)
        ).to(self.device)
        
        # Load decoder weights
        decoder_weights = torch.load(self.config['model_paths']['decoder'], map_location=self.device)
        if isinstance(decoder_weights, dict) and 'model_state_dict' in decoder_weights:
            self.decoder.load_state_dict(decoder_weights['model_state_dict'])
        else:
            self.decoder.load_state_dict(decoder_weights)
        self.decoder.eval()
    
    def _load_graph_data(self):
        """Load the graph data and create necessary mappings."""
        from ..datapipe import MovieLensLoader, BipartiteGraphBuilder
        
        # Load your actual data here
        data_path = Path(self.config['data']['path'])
        loader = MovieLensLoader(data_path)
        builder = BipartiteGraphBuilder()
        
        # Load and prepare the data
        all_data = loader.load_all_data()
        filtered_ratings = loader.filter_data(min_user_interactions=20, min_movie_interactions=5)
        encoded_ratings, encoding_info = loader.encode_ids(filtered_ratings)
        
        # Build the bipartite graph
        try:
            self.graph = builder.build_bipartite_graph(
                ratings=encoded_ratings,
                movies=all_data['movies'],
                encoding_info=encoding_info
            )
        except AttributeError as e:
            logger.error(f"Error building graph: {str(e)}")
            logger.error("Available methods in BipartiteGraphBuilder: " + 
                        ", ".join([m for m in dir(builder) if not m.startswith('_')]))
            raise
        
        # Create ID mappings
        self._create_id_mappings()
    
    def _create_id_mappings(self):
        """Create mappings between external and internal IDs."""
        # For users (0 to num_users-1)
        self.user_id_map = {i: i for i in range(self.graph.num_users)}
        
        # For movies (num_users to num_users+num_movies-1)
        self.movie_id_map = {i: i + self.graph.num_users 
                           for i in range(self.graph.num_movies)}
        
        # Create reverse mapping for movies
        self.reverse_movie_map = {v - self.graph.num_users: k 
                                for k, v in self.movie_id_map.items()}
    
    def get_user_embeddings(self, user_ids: List[Union[int, str]]) -> torch.Tensor:
        """Get embeddings for the specified users."""
        # Convert external user IDs to internal indices
        internal_ids = [self.user_id_map[uid] for uid in user_ids if uid in self.user_id_map]
        
        with torch.no_grad():
            embeddings = self.encoder(
                self.graph.x.to(self.device),
                self.graph.edge_index.to(self.device)
            )
            return embeddings[internal_ids]
    
    def get_top_k_recommendations(
        self, 
        user_id: Union[int, str],
        k: int = 10,
        filter_rated: bool = True,
        diversity_weight: float = 0.1
    ) -> List[Dict[str, float]]:
        """
        Get top-k movie recommendations for a user with diversity.
        
        Args:
            user_id: External user ID
            k: Number of recommendations to return
            filter_rated: Whether to filter out already rated movies
            diversity_weight: Weight for diversity in ranking (0-1)
            
        Returns:
            List of dictionaries containing movie IDs and scores
        """
        if user_id not in self.user_id_map:
            return self._handle_cold_start_user(user_id, k)
        
        # Get user embedding
        user_emb = self.get_user_embeddings([user_id])[0]  # [emb_dim]
        
        # Get all movie embeddings
        with torch.no_grad():
            movie_embs = self.encoder(
                self.graph.x.to(self.device),
                self.graph.edge_index.to(self.device)
            )[len(self.user_id_map):len(self.user_id_map) + len(self.movie_id_map)]
        
        # Calculate scores for all movies
        user_embs = user_emb.unsqueeze(0).repeat(len(movie_embs), 1)  # [num_movies, emb_dim]
        with torch.no_grad():
            scores = torch.sigmoid(self.decoder(user_embs, movie_embs)).squeeze()
        
        # Convert to numpy for easier manipulation
        scores = scores.cpu().numpy()
        movie_indices = np.arange(len(scores))
        
        # Filter out already rated movies if needed
        if filter_rated and hasattr(self.graph, 'train_edge_label_index'):
            user_idx = self.user_id_map[user_id]
            rated_mask = self.graph.train_edge_label_index[0] == user_idx
            rated_movie_indices = self.graph.train_edge_label_index[1, rated_mask] - len(self.user_id_map)
            mask = np.ones(len(scores), dtype=bool)
            mask[rated_movie_indices] = False
            scores = scores[mask]
            movie_indices = movie_indices[mask]
        
        # Apply diversity if requested
        if diversity_weight > 0 and len(movie_indices) > 0:
            scores = self._apply_diversity(
                user_emb.unsqueeze(0),
                movie_embs[movie_indices],
                scores,
                diversity_weight
            )
        
        # Get top-k recommendations
        top_indices = np.argsort(scores)[-k:][::-1]
        top_movie_indices = movie_indices[top_indices]
        top_scores = scores[top_indices]
        
        # Convert to external movie IDs and format results
        recommendations = []
        for idx, score in zip(top_movie_indices, top_scores):
            movie_id = self.reverse_movie_map.get(idx + len(self.user_id_map), idx)
            recommendations.append({
                'movie_id': movie_id,
                'score': float(score),
                'title': self._get_movie_title(movie_id)
            })
        
        return recommendations
    
    def _apply_diversity(
        self,
        user_emb: torch.Tensor,
        movie_embs: torch.Tensor,
        scores: np.ndarray,
        diversity_weight: float
    ) -> np.ndarray:
        """Apply diversity to the recommendations using MMR."""
        # Maximal Marginal Relevance (MMR) ranking
        selected = []
        remaining = list(range(len(scores)))
        
        # Convert to numpy for easier manipulation
        movie_embs_np = movie_embs.cpu().numpy()
        user_emb_np = user_emb.squeeze(0).cpu().numpy()
        
        # Normalize scores to [0, 1]
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        while remaining and len(selected) < min(100, len(scores)):  # Limit to top 100 for efficiency
            # Compute diversity scores
            if selected:
                # Calculate similarity to already selected items
                sim_matrix = np.dot(movie_embs_np[remaining], movie_embs_np[selected].T)  # [remaining, selected]
                max_sim = np.max(sim_matrix, axis=1)  # [remaining]
            else:
                max_sim = np.zeros(len(remaining))
            
            # Combine relevance and diversity
            mmr_scores = (1 - diversity_weight) * norm_scores[remaining] - diversity_weight * max_sim
            
            # Select item with highest MMR score
            best_idx = np.argmax(mmr_scores)
            selected.append(remaining.pop(best_idx))
        
        # Reorder scores based on MMR ranking
        return scores[selected + remaining]
    
    def _handle_cold_start_user(self, user_id: Union[int, str], k: int) -> List[Dict[str, float]]:
        """Handle cold-start users with no interaction history."""
        # Option 1: Return popular items
        if hasattr(self, 'popular_movies'):
            return self.popular_movies[:k]
        
        # Option 2: Use content-based filtering if available
        if hasattr(self, 'content_embeddings'):
            # Implement content-based recommendation
            pass
        
        # Fallback: Return random items
        movie_ids = list(self.movie_id_map.keys())
        selected = np.random.choice(movie_ids, size=min(k, len(movie_ids)), replace=False)
        return [{'movie_id': mid, 'score': 0.0, 'title': self._get_movie_title(mid)} for mid in selected]
    
    def _get_movie_title(self, movie_id: Union[int, str]) -> str:
        """Get movie title by ID (implement based on your data)."""
        # Implement this based on your data structure
        return str(movie_id)
    
    def predict_ratings(
        self, 
        user_ids: List[Union[int, str]], 
        movie_ids: List[Union[int, str]]
    ) -> List[Dict[str, float]]:
        """
        Predict ratings for given user-movie pairs.
        
        Args:
            user_ids: List of user IDs
            movie_ids: List of movie IDs (same length as user_ids)
            
        Returns:
            List of dictionaries with user_id, movie_id, and predicted_rating
        """
        # Convert to internal indices
        user_indices = [self.user_id_map.get(uid, -1) for uid in user_ids]
        movie_indices = [self.movie_id_map.get(mid, -1) for mid in movie_ids]
        
        # Filter out invalid pairs
        valid = [(i, uid, mid) for i, (uid, mid) in enumerate(zip(user_indices, movie_indices)) 
                if uid != -1 and mid != -1]
        
        if not valid:
            return []
        
        # Get embeddings for valid pairs
        idx, valid_u, valid_m = zip(*valid)
        with torch.no_grad():
            embeddings = self.encoder(
                self.graph.x.to(self.device),
                self.graph.edge_index.to(self.device)
            )
            user_embs = embeddings[list(valid_u)]
            movie_embs = embeddings[list(valid_m)]
            
            # Get predictions
            preds = torch.sigmoid(self.decoder(user_embs, movie_embs)).squeeze().cpu().numpy()
        
        # Create results
        results = []
        for i, (uid, mid) in enumerate(zip(user_ids, movie_ids)):
            if i in idx:
                pred_idx = idx.index(i)
                results.append({
                    'user_id': uid,
                    'movie_id': mid,
                    'predicted_rating': float(preds[pred_idx])
                })
            else:
                results.append({
                    'user_id': uid,
                    'movie_id': mid,
                    'predicted_rating': None,
                    'error': 'User or movie not found'
                })
        
        return results
