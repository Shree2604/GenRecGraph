"""
Basic Usage Example for GenRecGraph

This script demonstrates how to:
1. Load and preprocess MovieLens data
2. Create a bipartite graph
3. Generate comprehensive visualizations and statistics
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import os
import sys
from typing import Dict, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
# Import the existing modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Src'))

from datapipe.movielens_loader import load_movielens_data
from datapipe.graph_builder import BipartiteGraphBuilder
from utils.visualization import create_visualizations
from utils.validation import run_comprehensive_validation
from utils.encoder_utils import (
    train_encoder,
    visualize_embeddings,
    save_metrics,
    save_recommendations,
    save_encoder_checkpoint,
    compare_encoders
)


def main():
    """Main function to load data, create bipartite graph, and generate visualizations."""

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Validate data directory exists
    data_path = Path(__file__).parent / 'Data'
    if not data_path.exists():
        logger.error(f"Data directory not found: {data_path}")
        logger.info("Please ensure MovieLens-25M dataset CSV files are in the Data/ directory")
        return

    # Check for required CSV files
    required_files = ['ratings.csv', 'movies.csv', 'tags.csv', 'links.csv']
    missing_files = [f for f in required_files if not (data_path / f).exists()]
    if missing_files:
        logger.error(f"Missing required data files: {missing_files}")
        logger.info("Please download MovieLens-25M dataset and place CSV files in Data/ directory")
        return

    # Load and preprocess MovieLens data
    logger.info("Loading and preprocessing MovieLens data...")
    data_path = Path(__file__).parent / 'Data'

    try:
        data_splits, metadata = load_movielens_data(
            str(data_path),
            filter_params={
                'min_user_interactions': 20,
                'min_movie_interactions': 5,
                'rating_threshold': 3.0
            }
        )

        logger.info("Data preprocessing completed successfully!")

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        logger.info("Please ensure MovieLens-25M dataset is in the Data/ directory")
        return

    # Create bipartite graph
    logger.info("Creating bipartite graph...")
    graph_builder = BipartiteGraphBuilder(device=str(device))

    try:
        graph = graph_builder.build_bipartite_graph(
            ratings=data_splits['train'],
            movies=metadata['movies'],
            encoding_info=metadata['encoding_info']
        )

        logger.info(f"Bipartite graph created: {graph.num_nodes} nodes, {graph.num_edges} edges")
        logger.info(f"Graph has {graph.num_users} users and {graph.num_movies} movies")

    except Exception as e:
        logger.error(f"Error creating bipartite graph: {e}")
        return

    # Validate the generated data and graph
    logger.info("Validating data and graph integrity...")
    try:
        validation_results = run_comprehensive_validation(data_splits, metadata, graph)

        if not validation_results['overall_valid']:
            logger.error("Data validation failed! Check the errors above.")
            # Continue anyway but log the issues
        else:
            logger.info("✅ Data validation passed successfully!")

    except Exception as e:
        logger.error(f"Error during validation: {e}")
        logger.info("Continuing with data saving...")

    # Create comprehensive visualizations
    logger.info("Generating visualizations...")
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)

    try:
        viz_dir = create_visualizations(data_splits, metadata, graph, output_dir)
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        logger.info("Continuing with data saving...")
        viz_dir = None

    # Save preprocessed data and graph for later use
    logger.info("Saving processed data and graph...")

    try:
        # Save preprocessed data splits
        with open(output_dir / 'preprocessed_data.pkl', 'wb') as f:
            pickle.dump(data_splits, f)
        logger.info(f"Saved preprocessed data to {output_dir / 'preprocessed_data.pkl'}")

        # Save metadata
        with open(output_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"Saved metadata to {output_dir / 'metadata.pkl'}")

        # Save the graph (PyTorch Geometric Data object)
        with open(output_dir / 'bipartite_graph.pkl', 'wb') as f:
            pickle.dump(graph, f)
        logger.info(f"Saved bipartite graph to {output_dir / 'bipartite_graph.pkl'}")

        # Also save as PyTorch tensors for easy loading
        torch.save({
            'x': graph.x,
            'edge_index': graph.edge_index,
            'edge_attr': graph.edge_attr,
            'num_users': graph.num_users,
            'num_movies': graph.num_movies
        }, output_dir / 'graph_tensors.pt')

        logger.info(f"Saved graph tensors to {output_dir / 'graph_tensors.pt'}")

        # Compare all encoder types and select the best one
        logger.info("Comparing different encoder types...")
        try:
            encoder_types = ['gcn', 'gat', 'sage', 'bipartite']
            
            # Compare all encoders
            all_metrics = compare_encoders(
                graph=graph,
                device=device,
                output_dir=output_dir,
                encoder_types=encoder_types,
                num_epochs=30,  # Reduced for faster comparison
                learning_rate=0.01
            )
            
            # Find the best encoder based on validation loss
            best_encoder_type = min(
                all_metrics.items(),
                key=lambda x: x[1]['val_losses'][-1]
            )[0]
            
            logger.info(f"\n{'='*50}")
            logger.info(f"Best encoder: {best_encoder_type.upper()}")
            logger.info(f"{'='*50}")
            
            # Get the best encoder's output directory
            best_encoder_dir = output_dir / best_encoder_type
            
            # Load the best embeddings
            embeddings = torch.load(best_encoder_dir / 'node_embeddings.pt')
            
            # Generate and save visualizations for the best encoder
            logger.info("Generating final visualizations with the best encoder...")
            visualize_embeddings(
                embeddings=embeddings,
                output_dir=output_dir / 'best_encoder',
                encoder_type=best_encoder_type
            )
            
            # Save the best model's metrics to the main output directory
            best_metrics = all_metrics[best_encoder_type]
            save_metrics(best_metrics, output_dir / 'best_encoder')
            
            # Generate example recommendations using the best encoder
            save_recommendations(
                embeddings=embeddings,
                output_dir=output_dir / 'best_encoder',
                encoder_type=best_encoder_type
            )
            
            logger.info(f"\nComparison complete! Best encoder: {best_encoder_type.upper()}")
            logger.info(f"Detailed comparison report saved to: {output_dir / 'encoder_comparison.txt'}")
            
        except Exception as e:
            logger.error(f"Error during encoder comparison: {e}", exc_info=True)
            logger.info("Falling back to default GCN encoder...")
            
            # Fallback to GCN if comparison fails
            encoder, metrics = train_encoder(
                graph=graph,
                device=device,
                encoder_type='gcn',
                num_epochs=50,
                learning_rate=0.01
            )
            
            # Save the encoder checkpoint
            save_encoder_checkpoint(encoder, metrics, output_dir)
            
            # Save embeddings
            embeddings = metrics['embeddings']
            torch.save(embeddings, output_dir / 'node_embeddings.pt')
            np.save(output_dir / 'node_embeddings.npy', embeddings.cpu().numpy())
            
            # Generate and save visualizations
            visualize_embeddings(embeddings, output_dir, 'gcn')
            
            # Save training metrics
            save_metrics(metrics, output_dir)
            
            # Generate example recommendations
            save_recommendations(embeddings, output_dir=output_dir, encoder_type='gcn')

        if viz_dir:
            logger.info(f"Visualizations saved to {viz_dir}")
        logger.info("All outputs saved successfully!")

    except Exception as e:
        logger.error(f"Error saving outputs: {e}")
        return


# Encoder-related functions have been moved to utils/encoder_utils.py

if __name__ == "__main__":
    main()
