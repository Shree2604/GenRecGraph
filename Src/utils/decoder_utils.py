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
        # Convert dictionary to PyG Data object
        from torch_geometric.data import Data
        data = Data(
            x=graph['x'],
            edge_index=graph['edge_index'],
            num_nodes=graph.get('num_nodes', graph['x'].size(0))
        )
    else:
        data = graph
    
    # Get node embeddings
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
    
    # Optimizer and loss
    optimizer = optim.Adam(decoder.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    
    # Training loop
    best_val_auc = 0
    best_epoch = 0
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
        
        # Forward pass
        pos_out = decoder(embeddings, pos_train, sigmoid=True)
        neg_out = decoder(embeddings, neg_train, sigmoid=True)
        
        # Loss
        pos_loss = criterion(pos_out, torch.ones_like(pos_out))
        neg_loss = criterion(neg_out, torch.zeros_like(neg_out))
        loss = pos_loss + neg_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Validation
        val_auc, val_ap = evaluate(decoder, embeddings, pos_val, neg_val)
        
        # Update metrics
        metrics['train_loss'].append(loss.item())
        metrics['val_auc'].append(val_auc)
        metrics['val_ap'].append(val_ap)
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            
            # Save best model
            if output_dir:
                save_checkpoint(decoder, os.path.join(output_dir, f'best_{decoder_type}_decoder.pt'))
                
        elif epoch - best_epoch > patience:
            logger.info(f'Early stopping at epoch {epoch}')
            break
            
        # Log progress
        if epoch % 10 == 0:
            logger.info(
                f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                f'Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}'
            )
    
    # Final evaluation on test set
    test_auc, test_ap = evaluate(decoder, embeddings, pos_test, neg_test)
    metrics['test_auc'] = test_auc
    metrics['test_ap'] = test_ap
    
    # Save metrics
    if output_dir:
        with open(os.path.join(output_dir, f'{decoder_type}_metrics.json'), 'w') as f:
            json.dump(metrics, f)
    
    return metrics

def evaluate(
    decoder: nn.Module,
    embeddings: torch.Tensor,
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor
) -> Tuple[float, float]:
    """
    Evaluate decoder performance.
    
    Args:
        decoder: Decoder model
        embeddings: Node embeddings
        pos_edge_index: Positive edges
        neg_edge_index: Negative edges
        
    Returns:
        AUC and AP scores
    """
    decoder.eval()
    with torch.no_grad():
        pos_pred = decoder(embeddings, pos_edge_index, sigmoid=True).cpu()
        neg_pred = decoder(embeddings, neg_edge_index, sigmoid=True).cpu()
        
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
        decoder_types = ['inner_product', 'mlp', 'vae', 'autoregressive', 'bilinear']
    
    os.makedirs(output_dir, exist_ok=True)
    all_metrics = {}
    
    for decoder_type in decoder_types:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {decoder_type.upper()} decoder")
        logger.info(f"{'='*50}")
        
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
                output_dir=output_dir,
                num_epochs=training_config.get('num_epochs', 100),
                lr=training_config.get('learning_rate', 0.01),
                weight_decay=training_config.get('weight_decay', 5e-4),
                patience=training_config.get('patience', 10)
            )
            all_metrics[decoder_type] = metrics
            
            logger.info(
                f"{decoder_type.upper()} - "
                f"Test AUC: {metrics['test_auc']:.4f}, "
                f"Test AP: {metrics['test_ap']:.4f}"
            )
            
        except Exception as e:
            logger.error(f"Error with {decoder_type} decoder: {e}")
            all_metrics[decoder_type] = {'error': str(e)}
    
    # Save comparison results
    with open(os.path.join(output_dir, 'decoder_comparison.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    return all_metrics

def save_checkpoint(model: nn.Module, path: str):
    """Save model checkpoint."""
    torch.save({
        'model_state_dict': model.state_dict(),
    }, path)

def load_checkpoint(model: nn.Module, path: str, device: str = 'cpu'):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def negative_sampling(edge_index, num_nodes, num_neg_samples=None):
    """Generate negative samples."""
    if num_neg_samples is None:
        num_neg_samples = edge_index.size(1)
        
    # Generate random negative edges
    size = (2, num_neg_samples)
    neg_edge_index = torch.randint(0, num_nodes, size, dtype=torch.long)
    
    # Remove false negatives
    # (edges that are actually positive)
    mask = (edge_index[0] == edge_index[1])
    neg_edge_index = neg_edge_index[:, ~mask]
    
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
