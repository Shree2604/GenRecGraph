"""
GenRecGraph - Main Script for Decoder Comparison

This script loads a preprocessed graph, uses the best encoder (SAGE),
and compares different decoder architectures for recommendation.
"""

import os
import sys
import torch
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.absolute()))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import local modules
from Src.utils.decoder_utils import compare_decoders
from Src.models.encoders import create_encoder

def load_graph_data(data_path: str) -> dict:
    """
    Load and preprocess the raw dataset into a bipartite graph.
    
    Args:
        data_path: Path to the directory containing MovieLens CSV files
        
    Returns:
        Dictionary containing the processed graph data
    """
    from pathlib import Path
    import pandas as pd
    import torch
    from torch_geometric.data import Data
    from Src.datapipe import MovieLensLoader, BipartiteGraphBuilder
    
    logger.info("Loading and preprocessing raw dataset...")
    data_path = Path(data_path)
    
    # Initialize data loader
    loader = MovieLensLoader(data_path)
    
    try:
        # Load and preprocess the data
        logger.info("Loading MovieLens data...")
        all_data = loader.load_all_data()
        
        # Filter data
        logger.info("Filtering data...")
        filtered_ratings = loader.filter_data(
            min_user_interactions=20,
            min_movie_interactions=5,
            rating_threshold=3.0
        )
        
        # Encode IDs
        logger.info("Encoding user and movie IDs...")
        encoded_ratings, encoding_info = loader.encode_ids(filtered_ratings)
        
        # Create train/val/test splits
        logger.info("Creating data splits...")
        data_splits = loader.create_cold_start_split(encoded_ratings)
        
        # Build bipartite graph
        logger.info("Building bipartite graph...")
        graph_builder = BipartiteGraphBuilder(device='cuda' if torch.cuda.is_available() else 'cpu')
        
        graph = graph_builder.build_bipartite_graph(
            ratings=data_splits['train'],
            movies=all_data['movies'],
            encoding_info=encoding_info
        )
        
        # Convert to dictionary for saving
        graph_data = {
            'x': graph.x,
            'edge_index': graph.edge_index,
            'edge_attr': graph.edge_attr if hasattr(graph, 'edge_attr') else None,
            'num_users': graph.num_users,
            'num_movies': graph.num_movies,
            'train_mask': None,  # Add train/val/test masks if available
            'val_mask': None,
            'test_mask': None
        }
        
        logger.info(f"Graph created with {graph.num_nodes} nodes and {graph.num_edges} edges")
        logger.info(f"Number of users: {graph.num_users}, Number of movies: {graph.num_movies}")
        
        return graph_data
        
    except Exception as e:
        logger.error(f"Error loading graph data: {e}")
        raise

def main():
    # Configuration
    config = {
        'data': {
            'path': 'D:/Shree/GenRecGraph/output/graph_tensors.pt',  # Update this path if needed
        },
        'model': {
            'encoder': {
                'path': 'D:/Shree/GenRecGraph/output/sage/encoder_checkpoint.pt',  # Your pre-trained encoder
                'type': 'sage',
                'input_dim': 21,  # Should match your feature dimension
                'hidden_dims': [128, 21],  # Updated to match pre-trained model
                'output_dim': 21,  # Updated to match pre-trained model
                'dropout': 0.1
            },
            'decoder': {
                'types': ['mlp', 'vae', 'autoregressive', 'bilinear'],
                'hidden_dims': [128, 64],
                'dropout': 0.1
            }
        },
        'training': {
            'num_epochs': 50,
            'learning_rate': 0.01,
            'weight_decay': 5e-4,
            'patience': 10,
            'batch_size': 128
        },
        'output': {
            'root_dir': 'D:/Shree/GenRecGraph/output/decoder_comparison',
            'save_models': True
        }
    }

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(config['output']['root_dir']) / f'run_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load graph data
    try:
        graph_data = torch.load(config['data']['path'])
        logger.info("Graph data loaded successfully")
    except Exception as e:
        logger.error(f"Error loading graph data: {e}")
        return

    # Create Data object
    from torch_geometric.data import Data
    graph = Data(
        x=graph_data['x'],
        edge_index=graph_data['edge_index'],
        edge_attr=graph_data.get('edge_attr'),
        num_users=graph_data.get('num_users'),
        num_movies=graph_data.get('num_movies')
    )

    # Initialize and load the best encoder (SAGE)
    logger.info("Initializing SAGE encoder...")
    
    # Create encoder with proper configuration
    encoder_config = config['model']['encoder']
    encoder = create_encoder(
        encoder_type=encoder_config['type'],
        input_dim=encoder_config['input_dim'],
        hidden_dims=encoder_config['hidden_dims'],
        output_dim=encoder_config['output_dim'],
        dropout=encoder_config.get('dropout', 0.0)
    ).to(device)

    # Load the pre-trained SAGE encoder
    encoder_path = Path(config['model']['encoder']['path'])
    if not encoder_path.exists():
        raise FileNotFoundError(f"Pre-trained encoder not found at {encoder_path}. Please check the path.")
    
    try:
        checkpoint = torch.load(encoder_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            encoder.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded pre-trained encoder from {encoder_path}")
        elif 'encoder' in checkpoint:
            # Handle case where checkpoint has 'encoder' key
            encoder.load_state_dict(checkpoint['encoder'])
            logger.info(f"Loaded pre-trained encoder from {encoder_path} (using 'encoder' key)")
        else:
            # Try direct loading as a last resort
            encoder.load_state_dict(checkpoint)
            logger.info(f"Loaded encoder weights (direct loading) from {encoder_path}")
        logger.info("Encoder loaded successfully!")
    except Exception as e:
        raise RuntimeError(f"Failed to load encoder from {encoder_path}. Error: {str(e)}")

    # Set encoder to evaluation mode
    encoder.eval()

    # Compare decoders
    logger.info("\n" + "="*50)
    logger.info("Starting decoder comparison")
    logger.info("="*50)

    decoder_metrics = compare_decoders(
        graph=graph_data,
        encoder=encoder,
        decoder_config=config['model']['decoder'],
        training_config=config['training'],
        device=device,
        output_dir=output_dir,
        embedding_dim=encoder.get_embedding_dim(),
        num_users=graph_data['num_users'],
        num_movies=graph_data['num_movies']
    )

    # Print results
    logger.info("\n" + "="*50)
    logger.info("Decoder Comparison Results")
    logger.info("="*50)
    for decoder_type, metrics in decoder_metrics.items():
        if 'error' in metrics:
            logger.info(f"{decoder_type.upper()}: Error - {metrics['error']}")
        else:
            logger.info(
                f"{decoder_type.upper()}: "
                f"Test AUC: {metrics.get('test_auc', 'N/A'):.4f}, "
                f"Test AP: {metrics.get('test_ap', 'N/A'):.4f}"
            )

    logger.info("\nDecoder comparison completed!")

if __name__ == "__main__":
    main()
