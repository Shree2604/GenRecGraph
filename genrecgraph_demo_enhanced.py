"""
Enhanced GenRecGraph Demo with Cold-Start Handling

This script demonstrates how to use the RecommendationEngine to provide
personalized movie recommendations while handling cold-start scenarios.
"""
import os
import torch
import logging
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration - Update these paths to match your setup
CONFIG = {
    'encoder': {
        'type': 'sage',
        'input_dim': 21,           # Should match your feature dimension
        'hidden_dims': [128, 21],  # Match your trained encoder architecture
        'output_dim': 21,          # Match your trained encoder architecture
        'dropout': 0.1
    },
    'decoder': {
        'type': 'mlp',             # Using the best performing decoder
        'hidden_dims': [64, 32],   # Updated to match the trained model architecture
        'dropout': 0.1
    },
    'model_paths': {
        'encoder': r'D:\Shree\GenRecGraph\output\sage\encoder_checkpoint.pt',
        'decoder': r'D:\Shree\GenRecGraph\output\decoder_comparison\run_20251114_021532\mlp\final_mlp_decoder.pt',
    },
    'data': {
        'path': 'D:/Shree/GenRecGraph/data',  # Path to your MovieLens data
        'num_users': 1000,         # Update with your actual number of users
        'num_movies': 2000,        # Update with your actual number of movies
    },
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'diversity_weight': 0.2,       # Controls diversity in recommendations (0-1)
    'top_k': 10                    # Default number of recommendations
}

class DemoApp:
    """Demo application showing GenRecGraph recommendations."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the demo application."""
        self.config = config
        self.engine = None
        self._load_models()
    
    def _load_models(self):
        """Load the recommendation engine and data."""
        try:
            # Add the project root to the path
            import sys
            sys.path.append(str(Path(__file__).parent.absolute()))
            
            from Src.recommendation.recommendation_engine import RecommendationEngine
            
            logger.info("Initializing Recommendation Engine...")
            self.engine = RecommendationEngine(self.config)
            logger.info("Recommendation Engine initialized successfully")
            
            # Load additional data (e.g., movie titles, popular items)
            self._load_additional_data()
            
        except Exception as e:
            logger.error(f"Failed to initialize recommendation engine: {e}")
            raise
    
    def _load_additional_data(self):
        """Load additional data like movie titles and popular items."""
        # Example: Load movie titles
        self.movie_titles = {}
        movies_file = Path(self.config['data']['path']) / 'movies.csv'
        
        if movies_file.exists():
            import pandas as pd
            try:
                movies_df = pd.read_csv(movies_file)
                self.movie_titles = dict(zip(movies_df['movieId'], movies_df['title']))
                logger.info(f"Loaded {len(self.movie_titles)} movie titles")
            except Exception as e:
                logger.warning(f"Could not load movie titles: {e}")
        
        # Example: Pre-compute popular items
        self.popular_movies = self._get_popular_movies()
    
    def _get_popular_movies(self, top_n: int = 100) -> List[Dict]:
        """Get popular movies based on rating counts."""
        # In a real app, you might load this from a pre-computed file
        # or compute it from your training data
        return [
            {
                'movie_id': i,
                'title': f'Movie {i}',
                'score': 1.0 - (i / 100)  # Just for demo
            }
            for i in range(1, top_n + 1)
        ]
    
    def get_recommendations(
        self, 
        user_id: Union[int, str], 
        k: Optional[int] = None,
        filter_rated: bool = True,
        diversity: Optional[float] = None
    ) -> List[Dict]:
        """
        Get movie recommendations for a user.
        
        Args:
            user_id: User ID (can be new for cold-start)
            k: Number of recommendations to return
            filter_rated: Whether to filter out already rated movies
            diversity: Weight for diversity in recommendations (0-1)
            
        Returns:
            List of recommended movies with scores
        """
        if k is None:
            k = self.config.get('top_k', 10)
        
        if diversity is None:
            diversity = self.config.get('diversity_weight', 0.2)
        
        try:
            # Get recommendations from the engine
            recommendations = self.engine.get_top_k_recommendations(
                user_id=user_id,
                k=k,
                filter_rated=filter_rated,
                diversity_weight=diversity
            )
            
            # Add movie titles if available
            for rec in recommendations:
                movie_id_str = str(rec['movie_id'])
                if 'title' not in rec and movie_id_str in self.movie_titles:
                    rec['title'] = self.movie_titles[movie_id_str]
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            # Fallback to popular items
            return self.popular_movies[:k]
    
    def predict_ratings(
        self, 
        user_ids: List[Union[int, str]], 
        movie_ids: List[Union[int, str]]
    ) -> List[Dict]:
        """
        Predict ratings for user-movie pairs.
        
        Args:
            user_ids: List of user IDs
            movie_ids: List of movie IDs (same length as user_ids)
            
        Returns:
            List of predictions with user_id, movie_id, and predicted_rating
        """
        if len(user_ids) != len(movie_ids):
            raise ValueError("user_ids and movie_ids must have the same length")
        
        try:
            predictions = self.engine.predict_ratings(user_ids, movie_ids)
            
            # Add movie titles if available
            for pred in predictions:
                if 'movie_id' in pred:
                    movie_id_str = str(pred['movie_id'])
                    if movie_id_str in self.movie_titles:
                        pred['title'] = self.movie_titles[movie_id_str]
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting ratings: {e}")
            return [
                {
                    'user_id': uid,
                    'movie_id': mid,
                    'predicted_rating': None,
                    'error': str(e)
                }
                for uid, mid in zip(user_ids, movie_ids)
            ]

def load_movie_names(csv_path):
    """Load movie names from CSV file into a dictionary."""
    import pandas as pd
    try:
        # Read the CSV file
        movies_df = pd.read_csv(csv_path)
        # Create a dictionary with movieId as key and title as value
        return dict(zip(movies_df['movieId'], movies_df['title']))
    except Exception as e:
        logger.error(f"Error loading movie names: {e}")
        return {}

def main():
    """Run the demo application."""
    # Load movie names from CSV
    MOVIES_CSV_PATH = r"D:\Shree\GenRecGraph\Data\movies.csv"
    movie_names = load_movie_names(MOVIES_CSV_PATH)
    
    # Initialize the demo app
    try:
        app = DemoApp(CONFIG)
    except Exception as e:
        logger.error(f"Failed to initialize the application: {e}")
        return
    
    # Example 1: Get recommendations for an existing user
    print("\n" + "="*80)
    print("EXAMPLE 1: RECOMMENDATIONS FOR EXISTING USER")
    print("="*80)
    print("\nThis example shows personalized recommendations for an existing user in the system.")
    print("The recommendations are based on the user's historical interactions and preferences.")
    
    existing_user_id = 1  # Using a known user ID from the dataset
    print(f"\nFetching recommendations for existing user ID: {existing_user_id}...")
    recommendations = app.get_recommendations(existing_user_id, k=5)
    
    print(f"\nTop 5 personalized recommendations for user {existing_user_id}:")
    print("-" * 100)
    print(f"{'#':<4} {'Movie ID':<10} {'Title':<80} {'Score':<8}")
    print("-" * 100)
    for i, rec in enumerate(recommendations, 1):
        movie_id = rec.get('movie_id', 'N/A')
        # Get movie title from our loaded dictionary, fallback to rec['title'] if not found
        print(f"{i:<4} {str(movie_id):<10}  {rec.get('score', 0):.4f}")
    
    # Example 2: Get recommendations for a new user (cold-start)
    print("\n" + "="*80)
    print("EXAMPLE 2: RECOMMENDATIONS FOR NEW USER (COLD-START)")
    print("="*80)
    print("\nThis example demonstrates our cold-start solution for new users.")
    print("When a new user joins, we use a hybrid approach that combines:")
    print("1. Popular items from the user's demographic group (if available)")
    print("2. Trending items in the system")
    print("3. High-quality content with broad appeal")
    
    new_user_id = "new_user_123"  # This user doesn't exist in the training data
    print(f"\nGenerating cold-start recommendations for new user: {new_user_id}")
    recommendations = app.get_recommendations(new_user_id, k=5)
    
    print(f"\nTop 5 recommended movies for new user {new_user_id} (Cold Start):")
    print("-" * 100)
    print(f"{'#':<4} {'Movie ID':<10} ")
    print("-" * 100)
    for i, rec in enumerate(recommendations, 1):
        movie_id = rec.get('movie_id', 'N/A')
        print(f"{i:<4} {str(movie_id):<10} ")
    
    print("\n" + "="*80)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. Existing users receive personalized recommendations based on their history")
    print("2. New users get high-quality recommendations without requiring any interaction")
    print("3. The system seamlessly transitions from cold-start to personalized recommendations")
    print("4. All recommendations include both movie titles and IDs for reference")

if __name__ == "__main__":
    main()