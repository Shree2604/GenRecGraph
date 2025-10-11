"""
MovieLens-25M Dataset Loader

This module provides utilities for loading and preprocessing the MovieLens-25M dataset
for graph neural network-based recommendation systems.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import logging
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovieLensLoader:
    """
    Loader for MovieLens-25M dataset with preprocessing capabilities.
    
    Supports loading ratings, movies, tags, and genome data with various
    filtering and preprocessing options for cold-start recommendation research.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the MovieLens loader.
        
        Args:
            data_path: Path to the directory containing MovieLens CSV files
        """
        self.data_path = Path(data_path)
        self.ratings = None
        self.movies = None
        self.tags = None
        self.genome_scores = None
        self.genome_tags = None
        self.links = None
        
        # Encoders for converting IDs to continuous indices
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all MovieLens dataset files.
        
        Returns:
            Dictionary containing all loaded dataframes
        """
        logger.info("Loading MovieLens-25M dataset...")
        
        # Load core files
        self.ratings = self._load_ratings()
        self.movies = self._load_movies()
        self.tags = self._load_tags()
        self.links = self._load_links()
        
        # Load genome data (optional, large files)
        try:
            self.genome_scores = self._load_genome_scores()
            self.genome_tags = self._load_genome_tags()
        except Exception as e:
            logger.warning(f"Could not load genome data: {e}")
        
        logger.info("Dataset loading completed!")
        
        return {
            'ratings': self.ratings,
            'movies': self.movies,
            'tags': self.tags,
            'links': self.links,
            'genome_scores': self.genome_scores,
            'genome_tags': self.genome_tags
        }
    
    def _load_ratings(self) -> pd.DataFrame:
        """Load ratings.csv file."""
        ratings_path = self.data_path / 'ratings.csv'
        logger.info(f"Loading ratings from {ratings_path}")
        
        ratings = pd.read_csv(ratings_path)
        ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
        
        logger.info(f"Loaded {len(ratings):,} ratings from {ratings['userId'].nunique():,} users "
                   f"for {ratings['movieId'].nunique():,} movies")
        
        return ratings
    
    def _load_movies(self) -> pd.DataFrame:
        """Load movies.csv file."""
        movies_path = self.data_path / 'movies.csv'
        logger.info(f"Loading movies from {movies_path}")
        
        movies = pd.read_csv(movies_path)
        
        # Parse genres into list format
        movies['genres_list'] = movies['genres'].str.split('|')
        
        logger.info(f"Loaded {len(movies):,} movies")
        
        return movies
    
    def _load_tags(self) -> pd.DataFrame:
        """Load tags.csv file."""
        tags_path = self.data_path / 'tags.csv'
        logger.info(f"Loading tags from {tags_path}")
        
        tags = pd.read_csv(tags_path)
        tags['timestamp'] = pd.to_datetime(tags['timestamp'], unit='s')
        
        logger.info(f"Loaded {len(tags):,} tag applications")
        
        return tags
    
    def _load_links(self) -> pd.DataFrame:
        """Load links.csv file."""
        links_path = self.data_path / 'links.csv'
        logger.info(f"Loading links from {links_path}")
        
        links = pd.read_csv(links_path)
        
        logger.info(f"Loaded {len(links):,} movie links")
        
        return links
    
    def _load_genome_scores(self) -> pd.DataFrame:
        """Load genome-scores.csv file (large file, optional)."""
        genome_path = self.data_path / 'genome-scores.csv'
        logger.info(f"Loading genome scores from {genome_path}")
        
        # Load in chunks to handle large file
        chunk_size = 100000
        chunks = []
        
        for chunk in pd.read_csv(genome_path, chunksize=chunk_size):
            chunks.append(chunk)
        
        genome_scores = pd.concat(chunks, ignore_index=True)
        
        logger.info(f"Loaded {len(genome_scores):,} genome scores")
        
        return genome_scores
    
    def _load_genome_tags(self) -> pd.DataFrame:
        """Load genome-tags.csv file."""
        genome_tags_path = self.data_path / 'genome-tags.csv'
        logger.info(f"Loading genome tags from {genome_tags_path}")
        
        genome_tags = pd.read_csv(genome_tags_path)
        
        logger.info(f"Loaded {len(genome_tags):,} genome tags")
        
        return genome_tags
    
    def filter_data(self, 
                   min_user_interactions: int = 20,
                   min_movie_interactions: int = 5,
                   rating_threshold: float = 3.0) -> pd.DataFrame:
        """
        Filter the ratings data for quality and sparsity.
        
        Args:
            min_user_interactions: Minimum number of ratings per user
            min_movie_interactions: Minimum number of ratings per movie
            rating_threshold: Minimum rating to consider as positive interaction
            
        Returns:
            Filtered ratings dataframe
        """
        if self.ratings is None:
            raise ValueError("Ratings data not loaded. Call load_all_data() first.")
        
        logger.info("Filtering dataset...")
        original_size = len(self.ratings)
        
        # Filter by rating threshold
        filtered_ratings = self.ratings[self.ratings['rating'] >= rating_threshold].copy()
        
        # Iteratively filter users and movies
        prev_size = 0
        current_size = len(filtered_ratings)
        
        while prev_size != current_size:
            prev_size = current_size
            
            # Filter users with insufficient interactions
            user_counts = filtered_ratings['userId'].value_counts()
            valid_users = user_counts[user_counts >= min_user_interactions].index
            filtered_ratings = filtered_ratings[filtered_ratings['userId'].isin(valid_users)]
            
            # Filter movies with insufficient interactions
            movie_counts = filtered_ratings['movieId'].value_counts()
            valid_movies = movie_counts[movie_counts >= min_movie_interactions].index
            filtered_ratings = filtered_ratings[filtered_ratings['movieId'].isin(valid_movies)]
            
            current_size = len(filtered_ratings)
        
        logger.info(f"Filtered from {original_size:,} to {current_size:,} ratings "
                   f"({current_size/original_size:.2%} retained)")
        logger.info(f"Final dataset: {filtered_ratings['userId'].nunique():,} users, "
                   f"{filtered_ratings['movieId'].nunique():,} movies")
        
        return filtered_ratings
    
    def encode_ids(self, ratings: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Encode user and movie IDs to continuous indices for graph construction.
        
        Args:
            ratings: Ratings dataframe to encode
            
        Returns:
            Tuple of (encoded_ratings, encoding_info)
        """
        logger.info("Encoding user and movie IDs...")
        
        encoded_ratings = ratings.copy()
        
        # Encode user IDs
        encoded_ratings['user_idx'] = self.user_encoder.fit_transform(ratings['userId'])
        
        # Encode movie IDs
        encoded_ratings['movie_idx'] = self.movie_encoder.fit_transform(ratings['movieId'])
        
        encoding_info = {
            'num_users': len(self.user_encoder.classes_),
            'num_movies': len(self.movie_encoder.classes_),
            'user_id_to_idx': dict(zip(self.user_encoder.classes_, 
                                     range(len(self.user_encoder.classes_)))),
            'movie_id_to_idx': dict(zip(self.movie_encoder.classes_, 
                                      range(len(self.movie_encoder.classes_))))
        }
        
        logger.info(f"Encoded {encoding_info['num_users']:,} users and "
                   f"{encoding_info['num_movies']:,} movies")
        
        return encoded_ratings, encoding_info
    
    def create_cold_start_split(self, 
                              ratings: pd.DataFrame,
                              cold_start_ratio: float = 0.1,
                              test_ratio: float = 0.2) -> Dict[str, pd.DataFrame]:
        """
        Create train/validation/test splits with cold-start users and items.
        
        Args:
            ratings: Encoded ratings dataframe
            cold_start_ratio: Ratio of users/items to treat as cold-start
            test_ratio: Ratio of interactions for testing
            
        Returns:
            Dictionary with train/val/test/cold_start splits
        """
        logger.info("Creating cold-start data splits...")
        
        # Sort by timestamp for temporal splitting
        ratings_sorted = ratings.sort_values('timestamp').reset_index(drop=True)
        
        # Split by time
        n_total = len(ratings_sorted)
        n_test = int(n_total * test_ratio)
        n_val = int(n_total * 0.1)  # 10% for validation
        
        test_data = ratings_sorted[-n_test:].copy()
        val_data = ratings_sorted[-(n_test + n_val):-n_test].copy()
        train_data = ratings_sorted[:-(n_test + n_val)].copy()
        
        # Identify cold-start users and items
        train_users = set(train_data['user_idx'].unique())
        train_movies = set(train_data['movie_idx'].unique())
        
        test_users = set(test_data['user_idx'].unique())
        test_movies = set(test_data['movie_idx'].unique())
        
        # Cold-start users: appear in test but not in train
        cold_start_users = test_users - train_users
        
        # Cold-start movies: appear in test but not in train
        cold_start_movies = test_movies - train_movies
        
        # Create cold-start test set
        cold_start_test = test_data[
            (test_data['user_idx'].isin(cold_start_users)) |
            (test_data['movie_idx'].isin(cold_start_movies))
        ].copy()
        
        splits = {
            'train': train_data,
            'val': val_data,
            'test': test_data,
            'cold_start_test': cold_start_test
        }
        
        logger.info("Data split summary:")
        for split_name, split_data in splits.items():
            logger.info(f"  {split_name}: {len(split_data):,} interactions, "
                       f"{split_data['user_idx'].nunique():,} users, "
                       f"{split_data['movie_idx'].nunique():,} movies")
        
        logger.info(f"Cold-start entities: {len(cold_start_users):,} users, "
                   f"{len(cold_start_movies):,} movies")
        
        return splits


def load_movielens_data(data_path: str, 
                       filter_params: Optional[Dict] = None) -> Tuple[Dict, Dict]:
    """
    Convenience function to load and preprocess MovieLens data.
    
    Args:
        data_path: Path to MovieLens data directory
        filter_params: Parameters for data filtering
        
    Returns:
        Tuple of (data_splits, metadata)
    """
    if filter_params is None:
        filter_params = {
            'min_user_interactions': 20,
            'min_movie_interactions': 5,
            'rating_threshold': 3.0
        }
    
    # Initialize loader and load data
    loader = MovieLensLoader(data_path)
    all_data = loader.load_all_data()
    
    # Filter and encode data
    filtered_ratings = loader.filter_data(**filter_params)
    encoded_ratings, encoding_info = loader.encode_ids(filtered_ratings)
    
    # Create splits
    data_splits = loader.create_cold_start_split(encoded_ratings)
    
    # Prepare metadata
    metadata = {
        'encoding_info': encoding_info,
        'filter_params': filter_params,
        'movies': all_data['movies'],
        'tags': all_data['tags'],
        'links': all_data['links']
    }
    
    return data_splits, metadata
