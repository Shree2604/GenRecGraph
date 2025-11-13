"""
Encoder utilities for GenRecGraph

This module contains functions for training and evaluating graph encoders,
and generating visualizations and recommendations from learned embeddings.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    'train_encoder',
    'visualize_embeddings',
    'save_metrics',
    'save_recommendations',
    'save_encoder_checkpoint',
    'compare_encoders'
]

def train_encoder(graph, device: torch.device, encoder_type: str = 'gcn', 
                   num_epochs: int = 100, learning_rate: float = 0.01) -> Tuple[torch.nn.Module, Dict]:
    """
    Train the graph encoder model.
    
    Args:
        graph: The input graph
        device: Device to run the model on
        encoder_type: Type of encoder to use ('gcn', 'gat', 'sage', 'hetero', 'bipartite')
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        
    Returns:
        Tuple of (trained_encoder, training_metrics)
    """
    from models.encoders import create_encoder
    
    # Initialize encoder based on type
    input_dim = graph.num_node_features
    hidden_dim = 128
    embedding_dim = 64
    
    encoder = create_encoder(
        encoder_type=encoder_type,
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        hidden_dims=[hidden_dim, embedding_dim],
        dropout=0.1
    ).to(device)
    
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
    # Training loop
    metrics = {
        'train_losses': [],
        'val_losses': [],
        'training_time': 0,
        'encoder_type': encoder_type,
        'num_parameters': sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    }
    
    # Move graph to device
    x = graph.x.to(device)
    edge_index = graph.edge_index.to(device)
    
    # Train/validation split (simple version - in practice, use proper k-fold)
    num_edges = edge_index.size(1)
    indices = torch.randperm(num_edges)
    train_idx = indices[:int(0.8 * num_edges)]
    val_idx = indices[int(0.8 * num_edges):]
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        encoder.train()
        optimizer.zero_grad()
        
        # Forward pass
        if encoder_type == 'hetero':
            # For heterogeneous graph, we need to pass node types
            # Assuming we have a way to separate user and movie nodes
            # This is a simplified version - adjust based on your graph structure
            x_dict = {
                'user': x[graph.x[:, 0] == 1],  # Assuming first feature indicates user
                'movie': x[graph.x[:, 0] == 0]   # and second indicates movie
            }
            # This needs to be adapted to your actual edge_index structure
            embeddings = encoder(x_dict, graph.edge_index_dict)
            # Calculate loss for each node type and average
            loss = sum(criterion(emb, x_dict[ntype]) 
                     for ntype, emb in embeddings.items()) / len(embeddings)
        else:
            # For homogeneous graphs
            embeddings = encoder(x, edge_index)
            loss = criterion(embeddings, x)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Validation
        with torch.no_grad():
            encoder.eval()
            if encoder_type == 'hetero':
                val_embeddings = encoder(x_dict, graph.edge_index_dict)
                val_loss = sum(criterion(emb, x_dict[ntype]) 
                             for ntype, emb in val_embeddings.items()) / len(val_embeddings)
            else:
                val_embeddings = encoder(x, edge_index)
                val_loss = criterion(val_embeddings, x)
        
        metrics['train_losses'].append(loss.item())
        metrics['val_losses'].append(val_loss.item())
        
        if (epoch + 1) % 10 == 0:
            logger.info(f'[{encoder_type.upper()}] Epoch {epoch+1}/{num_epochs}, ' 
                      f'Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    
    metrics['training_time'] = time.time() - start_time
    logger.info(f'[{encoder_type.upper()}] Training completed in {metrics["training_time"]:.2f} seconds')
    
    # Get final embeddings
    with torch.no_grad():
        encoder.eval()
        if encoder_type == 'hetero':
            final_embeddings = encoder(x_dict, graph.edge_index_dict)
            # Combine embeddings from different node types
            # This is a simplified version - adjust based on your needs
            final_embeddings = torch.cat([e for e in final_embeddings.values()], dim=0)
        else:
            final_embeddings = encoder(x, edge_index)
    
    metrics.update({
        'embeddings': final_embeddings,
        'final_train_loss': metrics['train_losses'][-1],
        'final_val_loss': metrics['val_losses'][-1]
    })
    
    return encoder, metrics

def visualize_embeddings(embeddings: Union[torch.Tensor, np.ndarray], 
                         output_dir: Path, 
                         encoder_type: str = 'gcn',
                         n_samples: int = 1000) -> None:
    """
    Generate and save visualizations of the learned embeddings.
    
    Args:
        embeddings: Learned node embeddings (tensor or numpy array)
        output_dir: Directory to save visualizations
        encoder_type: Type of encoder used
        n_samples: Number of samples to use for t-SNE (for performance)
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Convert to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    
    # Sample a subset for t-SNE if the dataset is large
    if len(embeddings) > n_samples:
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        sampled_embeddings = embeddings[indices]
    else:
        sampled_embeddings = embeddings
    
    # t-SNE Visualization
    logger.info(f"Generating t-SNE visualization for {encoder_type}...")
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(sampled_embeddings)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
    plt.title(f't-SNE of Node Embeddings ({encoder_type.upper()})')
    plt.savefig(output_dir / 'tsne_embeddings.png', bbox_inches='tight')
    plt.close()
    
    # Similarity Heatmap
    logger.info("Generating similarity heatmap...")
    n_heatmap = min(50, len(embeddings))  # Use first 50 for heatmap
    similarity = cosine_similarity(embeddings[:n_heatmap])
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity, cmap='viridis', square=True)
    plt.title(f'Node Similarity Heatmap ({encoder_type.upper()})')
    plt.savefig(output_dir / 'similarity_heatmap.png', bbox_inches='tight')
    plt.close()

def save_metrics(metrics: Dict, output_dir: Path) -> None:
    """
    Save training metrics to a text file.
    
    Args:
        metrics: Dictionary containing training metrics
        output_dir: Directory to save the metrics file
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    with open(output_dir / 'training_metrics.txt', 'w') as f:
        f.write("Training Metrics\n")
        f.write("="*50 + "\n\n")
        f.write(f"Encoder Type: {metrics.get('encoder_type', 'unknown')}\n")
        f.write(f"Number of Parameters: {metrics.get('num_parameters', 'N/A')}\n")
        f.write(f"Final Training Loss: {metrics['train_losses'][-1]:.6f}\n")
        f.write(f"Final Validation Loss: {metrics['val_losses'][-1]:.6f}\n")
        f.write(f"Training Time: {metrics['training_time']:.2f} seconds\n\n")
        
        # Add learning curve data
        f.write("Learning Curve (first 5 and last 5 epochs):\n")
        epochs = len(metrics['train_losses'])
        for i in list(range(5)) + list(range(max(5, epochs-5), epochs)):
            f.write(f"Epoch {i+1:3d}: Train Loss = {metrics['train_losses'][i]:.6f}, "
                   f"Val Loss = {metrics['val_losses'][i]:.6f}\n")

def save_recommendations(embeddings: Union[torch.Tensor, np.ndarray], 
                         n_recommendations: int = 5, 
                         output_dir: Optional[Path] = None,
                         encoder_type: str = 'gcn') -> None:
    """
    Generate and save example recommendations based on embedding similarity.
    
    Args:
        embeddings: Learned node embeddings (tensor or numpy array)
        n_recommendations: Number of recommendations to generate
        output_dir: Directory to save recommendations (if None, print to console)
        encoder_type: Type of encoder used
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    
    # Example: Find similar items to the first user
    user_idx = 0
    user_embedding = embeddings[user_idx].reshape(1, -1)
    
    # Calculate cosine similarity with all items
    similarities = cosine_similarity(user_embedding, embeddings)[0]
    
    # Get top-k most similar items (excluding the user itself)
    top_indices = np.argsort(similarities)[-n_recommendations-1:-1][::-1]
    
    # Format recommendations
    recommendations = [f"Top {n_recommendations} recommendations for user {user_idx}:"]
    for i, idx in enumerate(top_indices, 1):
        recommendations.append(f"{i}. Item {idx} (Similarity: {similarities[idx]:.4f})")
    
    # Save or print
    if output_dir is not None:
        output_dir.mkdir(exist_ok=True, parents=True)
        with open(output_dir / f'recommendations_{encoder_type}.txt', 'w') as f:
            f.write("\n".join(recommendations))
    else:
        print("\n".join(recommendations))

def compare_encoders(graph, device: torch.device, output_dir: Path,
                    encoder_types: list = ['gcn', 'gat', 'sage', 'bipartite'],
                    num_epochs: int = 30, learning_rate: float = 0.01) -> Dict[str, Any]:
    """
    Compare multiple encoder types and return their metrics.
    
    Args:
        graph: Input graph
        device: Device to run the models on
        output_dir: Directory to save results
        encoder_types: List of encoder types to compare
        num_epochs: Number of training epochs per encoder
        learning_rate: Learning rate for optimizer
        
    Returns:
        Dictionary containing metrics for all encoders
    """
    from . import train_encoder  # Import here to avoid circular imports
    
    all_metrics = {}
    
    for encoder_type in encoder_types:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {encoder_type.upper()} encoder")
        logger.info(f"{'='*50}")
        
        try:
            # Create encoder-specific output directory
            encoder_output_dir = output_dir / encoder_type
            encoder_output_dir.mkdir(exist_ok=True, parents=True)
            
            # Train the encoder
            encoder, metrics = train_encoder(
                graph=graph,
                device=device,
                encoder_type=encoder_type,
                num_epochs=num_epochs,
                learning_rate=learning_rate
            )
            
            # Save the encoder checkpoint
            save_encoder_checkpoint(encoder, metrics, encoder_output_dir)
            
            # Save embeddings
            embeddings = metrics['embeddings']
            torch.save(embeddings, encoder_output_dir / 'node_embeddings.pt')
            np.save(encoder_output_dir / 'node_embeddings.npy', embeddings.cpu().numpy())
            
            # Generate and save visualizations
            visualize_embeddings(
                embeddings=embeddings,
                output_dir=encoder_output_dir,
                encoder_type=encoder_type
            )
            
            # Save metrics
            save_metrics(metrics, encoder_output_dir)
            
            # Generate example recommendations
            save_recommendations(
                embeddings=embeddings,
                output_dir=encoder_output_dir,
                encoder_type=encoder_type
            )
            
            all_metrics[encoder_type] = metrics
            
        except Exception as e:
            logger.error(f"Error with {encoder_type} encoder: {e}", exc_info=True)
    
    # Generate comparison report
    generate_comparison_report(all_metrics, output_dir)
    
    return all_metrics

def generate_comparison_report(metrics_dict: Dict[str, Any], output_dir: Path) -> None:
    """Generate a comparison report of all encoders."""
    report_path = output_dir / 'encoder_comparison.txt'
    
    with open(report_path, 'w') as f:
        f.write("ENCODER COMPARISON REPORT\n")
        f.write("="*50 + "\n\n")
        
        # Sort encoders by validation loss (ascending)
        sorted_encoders = sorted(
            metrics_dict.items(),
            key=lambda x: x[1]['val_losses'][-1] if 'val_losses' in x[1] else float('inf')
        )
        
        # Write summary table
        f.write("Summary:\n")
        f.write(f"{'Encoder':<15} {'Params':<15} {'Train Loss':<15} {'Val Loss':<15} {'Time (s)':<15}\n")
        f.write("-"*70 + "\n")
        
        for encoder_type, metrics in sorted_encoders:
            f.write(f"{encoder_type:<15} {metrics.get('num_parameters', 'N/A'):<15} "
                   f"{metrics['train_losses'][-1]:<15.6f} {metrics['val_losses'][-1]:<15.6f} "
                   f"{metrics.get('training_time', 'N/A'):<15.2f}\n")
        
        # Add detailed metrics for each encoder
        f.write("\n" + "="*50 + "\n")
        f.write("DETAILED METRICS\n")
        f.write("="*50 + "\n\n")
        
        for encoder_type, metrics in metrics_dict.items():
            f.write(f"{encoder_type.upper()} ENCODER\n")
            f.write("-"*50 + "\n")
            f.write(f"Number of parameters: {metrics.get('num_parameters', 'N/A')}\n")
            f.write(f"Final training loss: {metrics['train_losses'][-1]:.6f}\n")
            f.write(f"Final validation loss: {metrics['val_losses'][-1]:.6f}\n")
            f.write(f"Training time: {metrics.get('training_time', 'N/A'):.2f} seconds\n\n")
    
    logger.info(f"Generated encoder comparison report at {report_path}")


def save_encoder_checkpoint(encoder: torch.nn.Module, 
                             metrics: Dict, 
                             output_dir: Path, 
                             filename: Optional[str] = None) -> None:
    """
    Save encoder checkpoint with training metrics.
    
    Args:
        encoder: Trained encoder model
        metrics: Dictionary containing training metrics
        output_dir: Directory to save the checkpoint
        filename: Name of the checkpoint file
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = output_dir / filename
    
    if filename is None:
        filename = f'encoder_{metrics.get("encoder_type", "unknown")}_checkpoint.pth'
    checkpoint_path = output_dir / filename
    
    torch.save({
        'model_state_dict': encoder.state_dict(),
        'train_losses': metrics['train_losses'],
        'val_losses': metrics['val_losses'],
        'encoder_type': metrics.get('encoder_type', 'unknown'),
        'num_parameters': metrics.get('num_parameters', 0),
        'training_time': metrics.get('training_time', 0)
    }, checkpoint_path)
    
    logger.info(f"Saved encoder checkpoint to {checkpoint_path}")
