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
    num_epochs: int = 50,
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
            # VAE decoder returns a dict with 'recon' key
            pos_out_dict = decoder(user_emb, movie_emb)
            neg_out_dict = decoder(neg_user_emb, neg_movie_emb)
            pos_out = pos_out_dict['recon']
            neg_out = neg_out_dict['recon']
        elif isinstance(decoder, AutoregressiveDecoder):
            # Autoregressive decoder expects 3D input [batch, seq_len, features]
            # Combine user and movie embeddings as a sequence
            pos_seq = torch.stack([user_emb, movie_emb], dim=1)  # [batch, 2, emb_dim]
            neg_seq = torch.stack([neg_user_emb, neg_movie_emb], dim=1)
            pos_out = decoder(pos_seq)
            neg_out = decoder(neg_seq)
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
            kl_loss = pos_out_dict.get('kl_loss', 0) + neg_out_dict.get('kl_loss', 0)
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
    test_auc, test_ap = evaluate(decoder, embeddings, pos_test, neg_test, decoder_type)
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
            # VAE returns dict
            pos_pred = decoder(pos_user_emb, pos_movie_emb)['recon'].cpu()
            neg_pred = decoder(neg_user_emb, neg_movie_emb)['recon'].cpu()
        elif isinstance(decoder, AutoregressiveDecoder):
            # Autoregressive expects 3D input
            pos_seq = torch.stack([pos_user_emb, pos_movie_emb], dim=1)
            neg_seq = torch.stack([neg_user_emb, neg_movie_emb], dim=1)
            pos_pred = decoder(pos_seq).cpu()
            neg_pred = decoder(neg_seq).cpu()
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
        decoder_types = ['mlp', 'vae', 'autoregressive', 'bilinear']
    
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
                num_epochs=training_config.get('num_epochs', 50),
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
            import traceback
            logger.error(traceback.format_exc())
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
    
    # Create edge set for quick lookup
    edge_set = set()
    for i in range(edge_index.size(1)):
        edge_set.add((edge_index[0, i].item(), edge_index[1, i].item()))
    
    neg_edges = []
    while len(neg_edges) < num_neg_samples:
        src = torch.randint(0, num_nodes, (1,)).item()
        dst = torch.randint(0, num_nodes, (1,)).item()
        
        # Ensure it's not a self-loop and not an existing edge
        if src != dst and (src, dst) not in edge_set:
            neg_edges.append([src, dst])
    
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