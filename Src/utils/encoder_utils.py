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

logger = logging.getLogger(__name__)

def train_encoder(graph, device: torch.device, num_epochs: int = 100, 
                 learning_rate: float = 0.01) -> Tuple[torch.nn.Module, Dict]:
    """
    Train the graph encoder model.
    
    Args:
        graph: The input graph
        device: Device to run the model on
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        
    Returns:
        Tuple of (trained_encoder, training_history)
    """
    from models.encoders import GraphEncoder  # Local import to avoid circular imports
    
    # Initialize encoder
    input_dim = graph.num_node_features
    hidden_dim = 128
    embedding_dim = 64
    
    encoder = GraphEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim
    ).to(device)
    
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    
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
        embeddings = encoder(x, edge_index)
        
        # Simple reconstruction loss (can be replaced with more sophisticated loss)
        loss = criterion(embeddings, x)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Validation
        with torch.no_grad():
            encoder.eval()
            val_embeddings = encoder(x, edge_index)
            val_loss = criterion(val_embeddings, x)
            
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        
        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    
    training_time = time.time() - start_time
    logger.info(f'Training completed in {training_time:.2f} seconds')
    
    # Get final embeddings
    with torch.no_grad():
        encoder.eval()
        final_embeddings = encoder(x, edge_index)
    
    return encoder, {
        'embeddings': final_embeddings,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'training_time': training_time
    }

def visualize_embeddings(embeddings: Union[torch.Tensor, np.ndarray], 
                        output_dir: Path, 
                        n_samples: int = 1000) -> None:
    """
    Generate and save visualizations of the learned embeddings.
    
    Args:
        embeddings: Learned node embeddings (tensor or numpy array)
        output_dir: Directory to save visualizations
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
    logger.info("Generating t-SNE visualization...")
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(sampled_embeddings)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
    plt.title('t-SNE of Node Embeddings')
    plt.savefig(output_dir / 'tsne_embeddings.png', bbox_inches='tight')
    plt.close()
    
    # Similarity Heatmap
    logger.info("Generating similarity heatmap...")
    n_heatmap = min(50, len(embeddings))  # Use first 50 for heatmap
    similarity = cosine_similarity(embeddings[:n_heatmap])
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity, cmap='viridis', square=True)
    plt.title('Node Similarity Heatmap')
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
                        output_dir: Optional[Path] = None) -> None:
    """
    Generate and save example recommendations based on embedding similarity.
    
    Args:
        embeddings: Learned node embeddings (tensor or numpy array)
        n_recommendations: Number of recommendations to generate
        output_dir: Directory to save recommendations (if None, print to console)
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
        with open(output_dir / 'recommendations.txt', 'w') as f:
            f.write("\n".join(recommendations))
    else:
        print("\n".join(recommendations))

def save_encoder_checkpoint(encoder: torch.nn.Module, 
                          metrics: Dict, 
                          output_dir: Path, 
                          filename: str = 'encoder_checkpoint.pth') -> None:
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
    
    torch.save({
        'model_state_dict': encoder.state_dict(),
        'train_losses': metrics['train_losses'],
        'val_losses': metrics['val_losses']
    }, checkpoint_path)
    
    logger.info(f"Saved encoder checkpoint to {checkpoint_path}")
