"""
Decoder Utilities for GenRecGraph

This module provides utilities for training and evaluating different decoder architectures.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import logging
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns

from ..models.decoders import (
    create_decoder,
    GraphVAEDecoder,
    InnerProductDecoder,
    MLPDecoder,
    AutoregressiveDecoder,
    BilinearDecoder
)

logger = logging.getLogger(__name__)

def train_decoder(
    encoder: nn.Module,
    decoder: nn.Module,
    graph: Data,
    device: torch.device,
    decoder_type: str,
    num_epochs: int = 100,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    patience: int = 10,
    output_dir: Optional[str] = None
) -> Dict[str, List[float]]:
    """
    Train a decoder model.
    
    Args:
        encoder: Pretrained encoder model
        decoder: Decoder model to train
        graph: Graph data
        device: Device to train on
        decoder_type: Type of decoder
        num_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        patience: Patience for early stopping
        output_dir: Directory to save checkpoints
        
    Returns:
        Dictionary of training metrics
    """
    encoder.eval()
    decoder.train()
    
    # Handle both PyG Data objects and dictionaries
    if isinstance(graph, dict):
        from torch_geometric.data import Data
        data = Data(
            x=graph['x'],
            edge_index=graph['edge_index'],
            num_nodes=graph.get('num_nodes', graph['x'].size(0))
        )
    else:
        data = graph
    
    # Get node embeddings - keep gradient flow for inner product decoder
    if decoder_type == 'inner_product':
        # For inner product decoder, we need gradients through embeddings
        embeddings = encoder(data.x.to(device), data.edge_index.to(device))
        embeddings.requires_grad_(True)
    else:
        # For other decoders, detach embeddings
        with torch.no_grad():
            embeddings = encoder(data.x.to(device), data.edge_index.to(device))
    
    # Prepare data
    pos_edge_index = data.edge_index
    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=pos_edge_index.size(1)
    )
    
    # Split edges into train/val/test
    pos_train, pos_val, pos_test = split_edges(pos_edge_index, val_ratio=0.1, test_ratio=0.1)
    neg_train, neg_val, neg_test = split_edges(neg_edge_index, val_ratio=0.1, test_ratio=0.1)
    
    # Move to device
    pos_train, pos_val, pos_test = pos_train.to(device), pos_val.to(device), pos_test.to(device)
    neg_train, neg_val, neg_test = neg_train.to(device), neg_val.to(device), neg_test.to(device)
    
    # Optimizer - handle different decoder types
    if decoder_type == 'inner_product':
        # For inner product, optimize the embeddings directly
        optimizer = optim.Adam([embeddings], lr=lr, weight_decay=weight_decay)
    else:
        params = [p for p in decoder.parameters() if p.requires_grad]
        if not params:
            params = [nn.Parameter(torch.tensor(1.0, requires_grad=True))]
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    
    criterion = nn.BCELoss()
    
    # Training loop
    best_val_auc = 0
    best_epoch = 0
    best_model_state = None
    metrics = {
        'train_loss': [],
        'val_auc': [],
        'val_ap': [],
        'test_auc': [],
        'test_ap': []
    }
    
    for epoch in range(num_epochs):
        decoder.train()
        optimizer.zero_grad()
        
        # Get user and movie embeddings for the edges
        user_emb = embeddings[pos_train[0]]
        movie_emb = embeddings[pos_train[1]]
        neg_user_emb = embeddings[neg_train[0]]
        neg_movie_emb = embeddings[neg_train[1]]
        
        # Forward pass - handle different decoder types
        if isinstance(decoder, GraphVAEDecoder):
            # VAE decoder returns a dict - extract the output tensor
            pos_out_dict = decoder(user_emb, movie_emb)
            neg_out_dict = decoder(neg_user_emb, neg_movie_emb)
            
            # Try different possible keys for the reconstruction
            if isinstance(pos_out_dict, dict):
                # Common keys: 'recon', 'reconstruction', 'out', 'output', 'pred'
                for key in ['recon', 'reconstruction', 'out', 'output', 'pred']:
                    if key in pos_out_dict:
                        pos_out = pos_out_dict[key]
                        neg_out = neg_out_dict[key]
                        break
                else:
                    # If no known key found, use the first value
                    pos_out = list(pos_out_dict.values())[0]
                    neg_out = list(neg_out_dict.values())[0]
            else:
                # If it's not a dict, use it directly
                pos_out = pos_out_dict
                neg_out = neg_out_dict
        elif isinstance(decoder, AutoregressiveDecoder):
            # Autoregressive decoder - handle dimension mismatch
            # The decoder's forward expects (user_emb, movie_emb, edge_sequence)
            # Create dummy edge_sequence if needed or reshape embeddings
            try:
                # First try: pass embeddings directly
                pos_out = decoder(user_emb, movie_emb)
                neg_out = decoder(neg_user_emb, neg_movie_emb)
            except RuntimeError as e:
                if "same number of dimensions" in str(e):
                    # Add sequence dimension if needed
                    user_emb_3d = user_emb.unsqueeze(1)  # [batch, 1, emb_dim]
                    movie_emb_3d = movie_emb.unsqueeze(1)
                    neg_user_emb_3d = neg_user_emb.unsqueeze(1)
                    neg_movie_emb_3d = neg_movie_emb.unsqueeze(1)
                    
                    # Create edge sequence [batch, seq_len, emb_dim]
                    pos_seq = torch.cat([user_emb_3d, movie_emb_3d], dim=1)
                    neg_seq = torch.cat([neg_user_emb_3d, neg_movie_emb_3d], dim=1)
                    
                    # Try calling with sequence
                    pos_out = decoder(user_emb, movie_emb, pos_seq)
                    neg_out = decoder(neg_user_emb, neg_movie_emb, neg_seq)
                else:
                    raise
        elif isinstance(decoder, (MLPDecoder, BilinearDecoder)):
            # These decoders already apply sigmoid
            pos_out = decoder(user_emb, movie_emb)
            neg_out = decoder(neg_user_emb, neg_movie_emb)
        else:
            # InnerProduct and other decoders
            pos_out = decoder(user_emb, movie_emb)
            neg_out = decoder(neg_user_emb, neg_movie_emb)
            pos_out = torch.sigmoid(pos_out)
            neg_out = torch.sigmoid(neg_out)
        
        # Loss
        pos_loss = criterion(pos_out, torch.ones_like(pos_out))
        neg_loss = criterion(neg_out, torch.zeros_like(neg_out))
        loss = pos_loss + neg_loss
        
        # Add KL divergence for VAE
        if isinstance(decoder, GraphVAEDecoder):
            # Extract KL loss if available in the dict
            if isinstance(pos_out_dict, dict):
                kl_loss = pos_out_dict.get('kl_loss', 0) + neg_out_dict.get('kl_loss', 0)
                if kl_loss != 0:
                    loss = loss + 0.001 * kl_loss  # Small weight for KL term
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Validation
        val_auc, val_ap = evaluate(decoder, embeddings, pos_val, neg_val, decoder_type)
        
        # Update metrics
        metrics['train_loss'].append(loss.item())
        metrics['val_auc'].append(val_auc)
        metrics['val_ap'].append(val_ap)
        
        # Early stopping and model saving
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_model_state = decoder.state_dict().copy()
            
            # Save best model
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                save_checkpoint(
                    decoder, 
                    os.path.join(output_dir, f'best_{decoder_type}_decoder.pt'),
                    epoch=epoch,
                    val_auc=val_auc,
                    val_ap=val_ap
                )
                
        elif epoch - best_epoch > patience:
            logger.info(f'Early stopping at epoch {epoch}')
            break
            
        # Log progress
        if epoch % 10 == 0:
            logger.info(
                f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                f'Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}'
            )
    
    # Restore best model
    if best_model_state is not None:
        decoder.load_state_dict(best_model_state)
    
    # Final evaluation on test set
    test_auc, test_ap = evaluate(decoder, embeddings, pos_test, neg_test, decoder_type)
    metrics['test_auc'] = test_auc
    metrics['test_ap'] = test_ap
    
    # Save final model
    if output_dir:
        save_checkpoint(
            decoder,
            os.path.join(output_dir, f'final_{decoder_type}_decoder.pt'),
            epoch=epoch,
            val_auc=best_val_auc,
            test_auc=test_auc,
            test_ap=test_ap
        )
        
        # Save metrics
        with open(os.path.join(output_dir, f'{decoder_type}_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
    
    logger.info(f"Best model saved at epoch {best_epoch} with Val AUC: {best_val_auc:.4f}")
    
    return metrics

def evaluate(
    decoder: nn.Module,
    embeddings: torch.Tensor,
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
    decoder_type: str = None
) -> Tuple[float, float]:
    """
    Evaluate decoder performance.
    
    Args:
        decoder: Decoder model
        embeddings: Node embeddings
        pos_edge_index: Positive edges
        neg_edge_index: Negative edges
        decoder_type: Type of decoder (for special handling)
        
    Returns:
        AUC and AP scores
    """
    decoder.eval()
    with torch.no_grad():
        # Get user and movie embeddings for evaluation
        pos_user_emb = embeddings[pos_edge_index[0]]
        pos_movie_emb = embeddings[pos_edge_index[1]]
        neg_user_emb = embeddings[neg_edge_index[0]]
        neg_movie_emb = embeddings[neg_edge_index[1]]
        
        # Handle different decoder types in evaluation
        if isinstance(decoder, GraphVAEDecoder):
            # VAE returns dict or tensor
            pos_out = decoder(pos_user_emb, pos_movie_emb)
            neg_out = decoder(neg_user_emb, neg_movie_emb)
            
            # Extract tensor from dict if needed
            if isinstance(pos_out, dict):
                for key in ['recon', 'reconstruction', 'out', 'output', 'pred']:
                    if key in pos_out:
                        pos_pred = pos_out[key].cpu()
                        neg_pred = neg_out[key].cpu()
                        break
                else:
                    pos_pred = list(pos_out.values())[0].cpu()
                    neg_pred = list(neg_out.values())[0].cpu()
            else:
                pos_pred = pos_out.cpu()
                neg_pred = neg_out.cpu()
        elif isinstance(decoder, AutoregressiveDecoder):
            # Autoregressive expects separate embeddings
            try:
                pos_pred = decoder(pos_user_emb, pos_movie_emb).cpu()
                neg_pred = decoder(neg_user_emb, neg_movie_emb).cpu()
            except RuntimeError as e:
                if "same number of dimensions" in str(e):
                    # Add sequence dimension
                    pos_user_emb_3d = pos_user_emb.unsqueeze(1)
                    pos_movie_emb_3d = pos_movie_emb.unsqueeze(1)
                    neg_user_emb_3d = neg_user_emb.unsqueeze(1)
                    neg_movie_emb_3d = neg_movie_emb.unsqueeze(1)
                    
                    pos_seq = torch.cat([pos_user_emb_3d, pos_movie_emb_3d], dim=1)
                    neg_seq = torch.cat([neg_user_emb_3d, neg_movie_emb_3d], dim=1)
                    
                    pos_pred = decoder(pos_user_emb, pos_movie_emb, pos_seq).cpu()
                    neg_pred = decoder(neg_user_emb, neg_movie_emb, neg_seq).cpu()
                else:
                    raise
        elif isinstance(decoder, (MLPDecoder, BilinearDecoder)):
            # Already applies sigmoid
            pos_pred = decoder(pos_user_emb, pos_movie_emb).cpu()
            neg_pred = decoder(neg_user_emb, neg_movie_emb).cpu()
        else:
            # InnerProduct and others
            pos_pred = torch.sigmoid(decoder(pos_user_emb, pos_movie_emb)).cpu()
            neg_pred = torch.sigmoid(decoder(neg_user_emb, neg_movie_emb)).cpu()
        
        y_true = torch.cat([
            torch.ones(pos_pred.size(0)),
            torch.zeros(neg_pred.size(0))
        ], dim=0).numpy()
        
        y_score = torch.cat([pos_pred, neg_pred], dim=0).numpy()
        
        auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        
    return auc, ap

def compare_decoders(
    encoder: nn.Module,
    graph: Data,
    device: torch.device,
    decoder_types: List[str] = None,
    output_dir: str = 'output/decoders',
    **kwargs
) -> Dict[str, Dict[str, float]]:
    """
    Compare different decoder architectures.
    
    Args:
        encoder: Pretrained encoder model
        graph: Graph data
        device: Device to use
        decoder_types: List of decoder types to compare
        output_dir: Output directory
        **kwargs: Additional arguments for train_decoder
        
    Returns:
        Dictionary of metrics for each decoder
    """
    if decoder_types is None:
        # Exclude inner_product decoder
        decoder_types = ['mlp', 'vae', 'bilinear']
    
    os.makedirs(output_dir, exist_ok=True)
    all_metrics = {}
    best_decoder = None
    best_auc = 0
    
    for decoder_type in decoder_types:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {decoder_type.upper()} decoder")
        logger.info(f"{'='*50}")
        
        # Create decoder-specific output directory
        decoder_output_dir = os.path.join(output_dir, decoder_type)
        os.makedirs(decoder_output_dir, exist_ok=True)
        
        # Create decoder
        decoder = create_decoder(
            decoder_type=decoder_type,
            embedding_dim=encoder.get_embedding_dim(),
            **kwargs.get('decoder_kwargs', {})
        ).to(device)
        
        # Train decoder
        try:
            # Extract training parameters from kwargs
            training_config = kwargs.get('training_config', {})
            metrics = train_decoder(
                encoder=encoder,
                decoder=decoder,
                graph=graph,
                device=device,
                decoder_type=decoder_type,
                output_dir=decoder_output_dir,
                num_epochs=training_config.get('num_epochs', 50),  # Changed default to 50
                lr=training_config.get('learning_rate', 0.01),
                weight_decay=training_config.get('weight_decay', 5e-4),
                patience=training_config.get('patience', 10)
            )
            all_metrics[decoder_type] = metrics
            
            # Track best decoder
            if metrics['test_auc'] > best_auc:
                best_auc = metrics['test_auc']
                best_decoder = decoder_type
            
            logger.info(
                f"{decoder_type.upper()} - "
                f"Test AUC: {metrics['test_auc']:.4f}, "
                f"Test AP: {metrics['test_ap']:.4f}"
            )
            
        except Exception as e:
            logger.error(f"Error with {decoder_type} decoder: {e}")
            import traceback
            logger.error(traceback.format_exc())
            all_metrics[decoder_type] = {'error': str(e)}
    
    # Generate comparison plots
    plot_decoder_comparison(all_metrics, output_dir)
    
    # Save comparison results
    comparison_results = {
        'all_metrics': all_metrics,
        'best_decoder': best_decoder,
        'best_auc': best_auc
    }
    
    with open(os.path.join(output_dir, 'decoder_comparison.json'), 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    # Copy best model to main output directory
    if best_decoder:
        import shutil
        best_model_src = os.path.join(output_dir, best_decoder, f'best_{best_decoder}_decoder.pt')
        best_model_dst = os.path.join(output_dir, 'best_overall_decoder.pt')
        if os.path.exists(best_model_src):
            shutil.copy(best_model_src, best_model_dst)
            logger.info(f"\nBest overall decoder: {best_decoder.upper()} with AUC: {best_auc:.4f}")
            logger.info(f"Best model copied to: {best_model_dst}")
    
    return all_metrics

def save_checkpoint(model: nn.Module, path: str, **kwargs):
    """Save model checkpoint with additional metadata."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        **kwargs
    }
    torch.save(checkpoint, path)
    logger.info(f"Model checkpoint saved to: {path}")

def load_checkpoint(model: nn.Module, path: str, device: str = 'cpu'):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Model loaded from: {path}")
    return model, checkpoint

def negative_sampling(edge_index, num_nodes, num_neg_samples=None):
    """Generate negative samples."""
    if num_neg_samples is None:
        num_neg_samples = edge_index.size(1)
    
    # Create edge set for quick lookup
    edge_set = set()
    for i in range(edge_index.size(1)):
        edge_set.add((edge_index[0, i].item(), edge_index[1, i].item()))
    
    neg_edges = []
    max_attempts = num_neg_samples * 10
    attempts = 0
    
    while len(neg_edges) < num_neg_samples and attempts < max_attempts:
        src = torch.randint(0, num_nodes, (1,)).item()
        dst = torch.randint(0, num_nodes, (1,)).item()
        
        # Ensure it's not a self-loop and not an existing edge
        if src != dst and (src, dst) not in edge_set:
            neg_edges.append([src, dst])
            edge_set.add((src, dst))  # Avoid duplicates
        
        attempts += 1
    
    if len(neg_edges) < num_neg_samples:
        logger.warning(f"Only generated {len(neg_edges)} negative samples out of {num_neg_samples} requested")
    
    neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).t()
    return neg_edge_index

def split_edges(edge_index, val_ratio=0.05, test_ratio=0.1):
    """Split edges into train/val/test sets."""
    num_edges = edge_index.size(1)
    num_val = int(num_edges * val_ratio)
    num_test = int(num_edges * test_ratio)
    num_train = num_edges - num_val - num_test
    
    perm = torch.randperm(num_edges)
    train_idx = perm[:num_train]
    val_idx = perm[num_train:num_train + num_val]
    test_idx = perm[num_train + num_val:]
    
    return edge_index[:, train_idx], edge_index[:, val_idx], edge_index[:, test_idx]


def plot_decoder_comparison(all_metrics: Dict, output_dir: str):
    """
    Generate comparison plots for different decoders.
    
    Args:
        all_metrics: Dictionary containing metrics for all decoders
        output_dir: Directory to save plots
    """
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # Filter out decoders with errors
    valid_metrics = {k: v for k, v in all_metrics.items() if 'error' not in v}
    
    if not valid_metrics:
        logger.warning("No valid metrics to plot")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Decoder Comparison Metrics', fontsize=16, fontweight='bold')
    
    # 1. Training Loss Over Epochs
    ax = axes[0, 0]
    for decoder_name, metrics in valid_metrics.items():
        if 'train_loss' in metrics and metrics['train_loss']:
            epochs = range(len(metrics['train_loss']))
            ax.plot(epochs, metrics['train_loss'], marker='o', label=decoder_name.upper(), linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Validation AUC Over Epochs
    ax = axes[0, 1]
    for decoder_name, metrics in valid_metrics.items():
        if 'val_auc' in metrics and metrics['val_auc']:
            epochs = range(len(metrics['val_auc']))
            ax.plot(epochs, metrics['val_auc'], marker='o', label=decoder_name.upper(), linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation AUC', fontsize=12)
    ax.set_title('Validation AUC Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Validation AP Over Epochs
    ax = axes[0, 2]
    for decoder_name, metrics in valid_metrics.items():
        if 'val_ap' in metrics and metrics['val_ap']:
            epochs = range(len(metrics['val_ap']))
            ax.plot(epochs, metrics['val_ap'], marker='o', label=decoder_name.upper(), linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation AP', fontsize=12)
    ax.set_title('Validation AP Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Test AUC Bar Chart
    ax = axes[1, 0]
    decoder_names = list(valid_metrics.keys())
    test_aucs = [metrics['test_auc'] for metrics in valid_metrics.values()]
    colors = plt.cm.viridis(np.linspace(0, 1, len(decoder_names)))
    bars = ax.bar(decoder_names, test_aucs, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Test AUC', fontsize=12)
    ax.set_title('Test AUC Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([min(test_aucs) - 0.02, 1.0])
    # Add value labels on bars
    for bar, value in zip(bars, test_aucs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5. Test AP Bar Chart
    ax = axes[1, 1]
    test_aps = [metrics['test_ap'] for metrics in valid_metrics.values()]
    bars = ax.bar(decoder_names, test_aps, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Test AP', fontsize=12)
    ax.set_title('Test AP Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([min(test_aps) - 0.02, 1.0])
    # Add value labels on bars
    for bar, value in zip(bars, test_aps):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Combined Test Metrics Comparison
    ax = axes[1, 2]
    x = np.arange(len(decoder_names))
    width = 0.35
    bars1 = ax.bar(x - width/2, test_aucs, width, label='AUC', 
                   color='steelblue', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, test_aps, width, label='AP', 
                   color='coral', edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Test Metrics: AUC vs AP', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(decoder_names)
    ax.legend()
    ax.set_ylim([min(min(test_aucs), min(test_aps)) - 0.02, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save the figure
    plot_path = os.path.join(output_dir, 'decoder_comparison_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Comparison plots saved to: {plot_path}")
    plt.close()
    
    # Create individual plots for each metric (more detailed)
    
    # Training Loss - Individual Plot
    plt.figure(figsize=(10, 6))
    for decoder_name, metrics in valid_metrics.items():
        if 'train_loss' in metrics and metrics['train_loss']:
            epochs = range(len(metrics['train_loss']))
            plt.plot(epochs, metrics['train_loss'], marker='o', label=decoder_name.upper(), 
                    linewidth=2, markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_loss_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Validation AUC - Individual Plot
    plt.figure(figsize=(10, 6))
    for decoder_name, metrics in valid_metrics.items():
        if 'val_auc' in metrics and metrics['val_auc']:
            epochs = range(len(metrics['val_auc']))
            plt.plot(epochs, metrics['val_auc'], marker='o', label=decoder_name.upper(), 
                    linewidth=2, markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation AUC', fontsize=12)
    plt.title('Validation AUC Over Epochs', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'validation_auc_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Validation AP - Individual Plot
    plt.figure(figsize=(10, 6))
    for decoder_name, metrics in valid_metrics.items():
        if 'val_ap' in metrics and metrics['val_ap']:
            epochs = range(len(metrics['val_ap']))
            plt.plot(epochs, metrics['val_ap'], marker='o', label=decoder_name.upper(), 
                    linewidth=2, markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation AP', fontsize=12)
    plt.title('Validation AP Over Epochs', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'validation_ap_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Summary statistics table
    create_summary_table(valid_metrics, output_dir)
    
    logger.info(f"All comparison plots saved to: {output_dir}")


def create_summary_table(valid_metrics: Dict, output_dir: str):
    """
    Create a summary table image with decoder performance metrics.
    
    Args:
        valid_metrics: Dictionary of valid decoder metrics
        output_dir: Directory to save the table
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    decoder_names = list(valid_metrics.keys())
    table_data = []
    table_data.append(['Decoder', 'Test AUC', 'Test AP', 'Best Val AUC', 'Final Train Loss'])
    
    for decoder_name in decoder_names:
        metrics = valid_metrics[decoder_name]
        test_auc = f"{metrics['test_auc']:.4f}"
        test_ap = f"{metrics['test_ap']:.4f}"
        best_val_auc = f"{max(metrics['val_auc']):.4f}" if metrics['val_auc'] else "N/A"
        final_loss = f"{metrics['train_loss'][-1]:.4f}" if metrics['train_loss'] else "N/A"
        table_data.append([decoder_name.upper(), test_auc, test_ap, best_val_auc, final_loss])
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')
    
    # Style data rows with alternating colors
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f0f0f0')
            else:
                cell.set_facecolor('#ffffff')
    
    plt.title('Decoder Performance Summary', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    table_path = os.path.join(output_dir, 'decoder_summary_table.png')
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    logger.info(f"Summary table saved to: {table_path}")
    plt.close()