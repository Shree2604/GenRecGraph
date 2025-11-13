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
from Src.utils.encoder_utils import load_encoder

def load_graph_data(data_path: str) -> dict:
    """Load preprocessed graph data."""
    logger.info(f"Loading graph data from {data_path}")
    data = torch.load(data_path)
    return data

def main():
    # Configuration
    config = {
        'data': {
            'path': 'output/graph_tensors.pt',  # Path to preprocessed graph
        },
        'model': {
            'encoder': {
                'type': 'sage',  # Using SAGE as the best encoder
                'input_dim': 21,  # Should match your feature dimension
                'hidden_dims': [64, 64],
                'output_dim': 64,
                'dropout': 0.1
            },
            'decoder': {
                'types': ['inner_product', 'mlp', 'vae', 'autoregressive', 'bilinear'],
                'hidden_dims': [128, 64],
                'dropout': 0.1
            }
        },
        'training': {
            'num_epochs': 100,
            'learning_rate': 0.01,
            'weight_decay': 5e-4,
            'patience': 10,
            'batch_size': 128
        },
        'output': {
            'root_dir': 'output/decoder_comparison',
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

    # Load pre-trained encoder weights if available
    encoder_path = Path('output/encoder_comparison/sage/best_model.pt')
    if encoder_path.exists():
        try:
            checkpoint = torch.load(encoder_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                encoder.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded pre-trained encoder from {encoder_path}")
            else:
                encoder.load_state_dict(checkpoint)
                logger.info(f"Loaded encoder weights (legacy format) from {encoder_path}")
        except Exception as e:
            logger.warning(f"Error loading encoder weights: {e}. Using random initialization.")
    else:
        logger.warning("No pre-trained encoder found. Using random initialization.")

    # Set encoder to evaluation mode
    encoder.eval()

    # Compare decoders
    logger.info("\n" + "="*50)
    logger.info("Starting decoder comparison")
    logger.info("="*50)

    decoder_metrics = compare_decoders(
        encoder=encoder,
        graph=graph,
        device=device,
        decoder_types=config['model']['decoder']['types'],
        output_dir=str(output_dir),
        num_epochs=config['training']['num_epochs'],
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        patience=config['training']['patience'],
        decoder_kwargs={
            'hidden_dims': config['model']['decoder']['hidden_dims'],
            'dropout': config['model']['decoder']['dropout']
        }
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
