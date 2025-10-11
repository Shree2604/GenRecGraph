"""
Graph Construction for MovieLens Recommendation System

This module provides utilities for constructing bipartite user-item graphs
from MovieLens data for use with PyTorch Geometric.
"""

import torch
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List
import logging
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import ToUndirected
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import networkx as nx

logger = logging.getLogger(__name__)


class BipartiteGraphBuilder:
    """
    Builder for creating bipartite user-item graphs from MovieLens data.
    
    Supports both homogeneous and heterogeneous graph representations
    with rich node and edge features.
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize the graph builder.
        
        Args:
            device: Device to place tensors on ('cpu' or 'cuda')
        """
        self.device = device
        self.scaler = StandardScaler()
        self.genre_encoder = MultiLabelBinarizer()
        
    def build_bipartite_graph(self,
                            ratings: pd.DataFrame,
                            movies: pd.DataFrame,
                            encoding_info: Dict,
                            include_features: bool = True) -> Data:
        """
        Build a bipartite graph from user-item interactions.

        Args:
            ratings: DataFrame with user_idx, movie_idx, rating columns
            movies: DataFrame with movie information
            encoding_info: Dictionary with encoding information
            include_features: Whether to include node features

        Returns:
            PyTorch Geometric Data object
        """
        logger.info("Building bipartite user-item graph...")

        # Validate inputs
        required_columns = ['user_idx', 'movie_idx', 'rating']
        for col in required_columns:
            if col not in ratings.columns:
                raise ValueError(f"Required column '{col}' not found in ratings DataFrame")

        num_users = encoding_info.get('num_users', len(ratings['user_idx'].unique()))
        num_movies = encoding_info.get('num_movies', len(ratings['movie_idx'].unique()))

        # Validate data consistency
        if len(ratings) == 0:
            raise ValueError("Empty ratings DataFrame provided")

        # Create edge indices for bipartite graph
        # Users: 0 to num_users-1
        # Movies: num_users to num_users+num_movies-1
        user_indices = ratings['user_idx'].values
        movie_indices = ratings['movie_idx'].values + num_users  # Offset movie indices

        # Create edge index efficiently
        logger.info(f"Creating edge index for {len(ratings)} interactions...")
        # Use numpy array concatenation for better performance
        edge_index_np = np.column_stack([user_indices, movie_indices]).astype(np.int64)

        # Convert to torch tensor more efficiently (avoid the warning)
        edge_index = torch.from_numpy(edge_index_np.T).long()

        # Create edge attributes (ratings) with validation
        edge_attr = torch.FloatTensor(ratings['rating'].values).unsqueeze(1)

        # Validate rating range
        if edge_attr.min() < 0.5 or edge_attr.max() > 5.0:
            logger.warning(f"Rating values outside expected range [0.5, 5.0]: "
                          f"min={edge_attr.min():.2f}, max={edge_attr.max():.2f}")

        # Create node features
        logger.info("Creating node features...")
        node_features = self._create_node_features(num_users, num_movies, movies, ratings)

        # Create PyTorch Geometric Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr
        )

        # Set user and movie counts for easy access
        data.num_users = num_users
        data.num_movies = encoding_info.get('num_movies', num_movies)

        # Validate graph structure
        expected_nodes = num_users + encoding_info.get('num_movies', num_movies)
        if data.num_nodes != expected_nodes:
            logger.warning(f"Node count mismatch: expected {expected_nodes}, got {data.num_nodes}")

        logger.info(f"Created bipartite graph with {data.num_nodes} nodes "
                   f"({data.num_users} users + {data.num_movies} movies) "
                   f"and {data.num_edges} edges")

        return data
    
    def build_heterogeneous_graph(self,
                                ratings: pd.DataFrame,
                                movies: pd.DataFrame,
                                encoding_info: Dict,
                                include_features: bool = True) -> HeteroData:
        """
        Build a heterogeneous graph with separate user and movie node types.
        
        Args:
            ratings: DataFrame with user_idx, movie_idx, rating columns
            movies: DataFrame with movie information
            encoding_info: Dictionary with encoding information
            include_features: Whether to include node features
            
        Returns:
            PyTorch Geometric HeteroData object
        """
        logger.info("Building heterogeneous user-item graph...")
        
        num_users = encoding_info['num_users']
        num_movies = encoding_info['num_movies']
        
        # Create heterogeneous data object
        data = HeteroData()
        
        # Add node features
        if include_features:
            user_features, movie_features = self._create_hetero_node_features(
                num_users, num_movies, movies, ratings
            )
        else:
            user_features = torch.eye(num_users, device=self.device)
            movie_features = torch.eye(num_movies, device=self.device)
        
        data['user'].x = user_features
        data['movie'].x = movie_features
        
        # Add edges
        user_indices = torch.tensor(ratings['user_idx'].values, dtype=torch.long)
        movie_indices = torch.tensor(ratings['movie_idx'].values, dtype=torch.long)
        
        # User-movie interactions
        data['user', 'rates', 'movie'].edge_index = torch.stack([
            user_indices, movie_indices
        ], dim=0).to(self.device)
        
        # Movie-user interactions (reverse)
        data['movie', 'rated_by', 'user'].edge_index = torch.stack([
            movie_indices, user_indices
        ], dim=0).to(self.device)
        
        # Edge attributes (ratings)
        edge_attr = torch.tensor(ratings['rating'].values, dtype=torch.float, device=self.device)
        data['user', 'rates', 'movie'].edge_attr = edge_attr.unsqueeze(1)
        data['movie', 'rated_by', 'user'].edge_attr = edge_attr.unsqueeze(1)
        
        logger.info(f"Created heterogeneous graph with {num_users} users, "
                   f"{num_movies} movies, and {len(ratings)} interactions")
        
        return data
    
    def _create_node_features(self, 
                            num_users: int,
                            num_movies: int,
                            movies: pd.DataFrame,
                            ratings: pd.DataFrame) -> torch.Tensor:
        """Create node features for homogeneous bipartite graph."""
        logger.info("Creating node features...")
        
        # User features (statistical features from ratings)
        user_features = self._create_user_features(ratings, num_users)
        
        # Movie features (content-based features)
        movie_features = self._create_movie_features(movies, num_movies)
        
        # Ensure same dimensionality
        user_dim = user_features.shape[1]
        movie_dim = movie_features.shape[1]
        
        if user_dim != movie_dim:
            max_dim = max(user_dim, movie_dim)
            
            if user_dim < max_dim:
                padding = torch.zeros(num_users, max_dim - user_dim, device=self.device)
                user_features = torch.cat([user_features, padding], dim=1)
            
            if movie_dim < max_dim:
                padding = torch.zeros(num_movies, max_dim - movie_dim, device=self.device)
                movie_features = torch.cat([movie_features, padding], dim=1)
        
        # Concatenate user and movie features
        node_features = torch.cat([user_features, movie_features], dim=0)
        
        logger.info(f"Created node features with shape {node_features.shape}")
        
        return node_features
    
    def _create_hetero_node_features(self,
                                   num_users: int,
                                   num_movies: int,
                                   movies: pd.DataFrame,
                                   ratings: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create separate node features for heterogeneous graph."""
        user_features = self._create_user_features(ratings, num_users)
        movie_features = self._create_movie_features(movies, num_movies)
        
        return user_features, movie_features
    
    def _create_user_features(self, ratings: pd.DataFrame, num_users: int) -> torch.Tensor:
        """Create user features from rating statistics."""
        # Calculate user statistics
        user_stats = ratings.groupby('user_idx')['rating'].agg([
            'mean', 'std', 'count', 'min', 'max'
        ]).fillna(0)
        
        # Ensure all users are represented
        user_features = np.zeros((num_users, 5))
        
        for user_idx in user_stats.index:
            user_features[user_idx] = user_stats.loc[user_idx].values
        
        # Normalize features
        user_features = self.scaler.fit_transform(user_features)
        
        return torch.tensor(user_features, dtype=torch.float, device=self.device)
    
    def _create_movie_features(self, movies: pd.DataFrame, num_movies: int) -> torch.Tensor:
        """Create movie features from content information."""
        # Prepare genre features
        if 'genres_list' in movies.columns:
            # Get all unique genres
            all_genres = set()
            for genres in movies['genres_list'].dropna():
                if isinstance(genres, list):
                    all_genres.update(genres)
            
            # Create genre binary encoding
            genre_matrix = []
            for _, movie in movies.iterrows():
                genres = movie['genres_list'] if isinstance(movie['genres_list'], list) else []
                genre_vector = [1 if genre in genres else 0 for genre in sorted(all_genres)]
                genre_matrix.append(genre_vector)
            
            genre_features = np.array(genre_matrix)
        else:
            # Fallback: parse genres from string
            movies_copy = movies.copy()
            movies_copy['genres_list'] = movies_copy['genres'].str.split('|')
            genre_features = self.genre_encoder.fit_transform(movies_copy['genres_list'])
        
        # Add year feature if available
        if 'title' in movies.columns:
            # Extract year from title (format: "Title (YEAR)")
            years = movies['title'].str.extract(r'\\((\\d{4})\\)$')[0]
            years = pd.to_numeric(years, errors='coerce').fillna(1995)  # Default year
            year_features = ((years - 1995) / (2020 - 1995)).values.reshape(-1, 1)  # Normalize
            
            # Combine genre and year features
            movie_features = np.hstack([genre_features, year_features])
        else:
            movie_features = genre_features
        
        # Ensure correct number of movies
        if len(movie_features) != num_movies:
            # Pad or truncate to match expected number
            if len(movie_features) < num_movies:
                padding = np.zeros((num_movies - len(movie_features), movie_features.shape[1]))
                movie_features = np.vstack([movie_features, padding])
            else:
                movie_features = movie_features[:num_movies]
        
        return torch.tensor(movie_features, dtype=torch.float, device=self.device)
    
    def add_negative_edges(self, 
                          data: Data,
                          num_negative: Optional[int] = None,
                          negative_ratio: float = 1.0) -> Data:
        """
        Add negative edges for training (non-existing user-item pairs).
        
        Args:
            data: PyTorch Geometric data object
            num_negative: Exact number of negative edges to add
            negative_ratio: Ratio of negative to positive edges
            
        Returns:
            Data object with negative edges added
        """
        logger.info("Adding negative edges for training...")
        
        num_users = data.num_users
        num_movies = data.num_movies
        
        # Get existing edges (positive edges)
        edge_index = data.edge_index.cpu().numpy()
        existing_edges = set()
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            if src < num_users and dst >= num_users:  # User to movie edge
                existing_edges.add((src, dst - num_users))
        
        # Determine number of negative edges
        if num_negative is None:
            num_negative = int(len(existing_edges) * negative_ratio)
        
        # Sample negative edges
        negative_edges = []
        max_attempts = num_negative * 10  # Prevent infinite loop
        attempts = 0
        
        while len(negative_edges) < num_negative and attempts < max_attempts:
            user_idx = np.random.randint(0, num_users)
            movie_idx = np.random.randint(0, num_movies)
            
            if (user_idx, movie_idx) not in existing_edges:
                negative_edges.append((user_idx, movie_idx + num_users))
                existing_edges.add((user_idx, movie_idx))  # Avoid duplicates
            
            attempts += 1
        
        if len(negative_edges) < num_negative:
            logger.warning(f"Could only generate {len(negative_edges)} negative edges "
                          f"out of {num_negative} requested")
        
        # Add negative edges to data
        if negative_edges:
            neg_user_indices = [edge[0] for edge in negative_edges]
            neg_movie_indices = [edge[1] for edge in negative_edges]
            
            # Create bidirectional negative edges
            neg_edge_index = torch.tensor([
                neg_user_indices + neg_movie_indices,
                neg_movie_indices + neg_user_indices
            ], dtype=torch.long, device=self.device)
            
            # Combine positive and negative edges
            combined_edge_index = torch.cat([data.edge_index, neg_edge_index], dim=1)
            
            # Create edge labels (1 for positive, 0 for negative)
            pos_labels = torch.ones(data.edge_index.shape[1], device=self.device)
            neg_labels = torch.zeros(neg_edge_index.shape[1], device=self.device)
            edge_labels = torch.cat([pos_labels, neg_labels])
            
            # Update data object
            data.edge_index = combined_edge_index
            data.edge_label = edge_labels
            
            # Update edge attributes
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                neg_edge_attr = torch.zeros(neg_edge_index.shape[1], 1, device=self.device)
                data.edge_attr = torch.cat([data.edge_attr, neg_edge_attr], dim=0)
        
        logger.info(f"Added {len(negative_edges)} negative edges")
        
        return data
    
    def create_subgraph(self, 
                       data: Data,
                       user_indices: List[int],
                       movie_indices: List[int]) -> Data:
        """
        Create a subgraph containing only specified users and movies.
        
        Args:
            data: Original graph data
            user_indices: List of user indices to include
            movie_indices: List of movie indices to include
            
        Returns:
            Subgraph data object
        """
        # Adjust movie indices for bipartite representation
        adjusted_movie_indices = [idx + data.num_users for idx in movie_indices]
        node_indices = user_indices + adjusted_movie_indices
        
        # Create node mapping
        node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(node_indices)}
        
        # Filter edges
        edge_mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
        
        for i in range(data.edge_index.shape[1]):
            src, dst = data.edge_index[0, i].item(), data.edge_index[1, i].item()
            if src in node_mapping and dst in node_mapping:
                edge_mask[i] = True
        
        # Create subgraph
        sub_edge_index = data.edge_index[:, edge_mask]
        
        # Remap node indices
        for i in range(sub_edge_index.shape[1]):
            sub_edge_index[0, i] = node_mapping[sub_edge_index[0, i].item()]
            sub_edge_index[1, i] = node_mapping[sub_edge_index[1, i].item()]
        
        # Create subgraph data
        subgraph_data = Data(
            x=data.x[node_indices],
            edge_index=sub_edge_index,
            num_users=len(user_indices),
            num_movies=len(movie_indices)
        )
        
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            subgraph_data.edge_attr = data.edge_attr[edge_mask]
        
        return subgraph_data


def create_graph_from_ratings(ratings: pd.DataFrame,
                            movies: pd.DataFrame,
                            encoding_info: Dict,
                            graph_type: str = 'bipartite',
                            include_features: bool = True,
                            device: str = 'cpu') -> Data:
    """
    Convenience function to create a graph from MovieLens ratings.
    
    Args:
        ratings: Ratings dataframe
        movies: Movies dataframe
        encoding_info: Encoding information
        graph_type: Type of graph ('bipartite' or 'heterogeneous')
        include_features: Whether to include node features
        device: Device to place tensors on
        
    Returns:
        PyTorch Geometric data object
    """
    builder = BipartiteGraphBuilder(device=device)
    
    if graph_type == 'bipartite':
        return builder.build_bipartite_graph(
            ratings, movies, encoding_info, include_features
        )
    elif graph_type == 'heterogeneous':
        return builder.build_heterogeneous_graph(
            ratings, movies, encoding_info, include_features
        )
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
