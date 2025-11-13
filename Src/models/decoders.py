"""
Graph Generative Decoders for Cold-Start Recommendation

This module implements various generative models for predicting and generating
user-item interactions, including GraphVAE and autoregressive decoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli
from typing import Dict, Tuple, Optional, List
import logging
import math

logger = logging.getLogger(__name__)


class GraphVAEDecoder(nn.Module):
    """
    Variational Autoencoder decoder for graph generation.
    
    Learns a probabilistic model of edge existence based on node embeddings
    and can generate new edges by sampling from the learned distribution.
    """
    
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int = 32,
                 latent_dim: int = 16,
                 dropout: float = 0.1):
        """
        Initialize GraphVAE decoder.
        
        Args:
            embedding_dim: Dimension of input node embeddings
            hidden_dim: Hidden layer dimension
            latent_dim: Dimension of latent space
            dropout: Dropout probability
        """
        super(GraphVAEDecoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder: embeddings -> latent distribution parameters
        self.encoder_mean = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.encoder_logvar = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Decoder: latent -> edge probability
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        logger.info(f"Initialized GraphVAE decoder with latent_dim={latent_dim}")
    
    def encode(self, user_emb: torch.Tensor, movie_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode user-movie pairs to latent distribution parameters.
        
        Args:
            user_emb: User embeddings [batch_size, embedding_dim]
            movie_emb: Movie embeddings [batch_size, embedding_dim]
            
        Returns:
            Tuple of (mean, logvar) for latent distribution
        """
        # Concatenate user and movie embeddings
        combined = torch.cat([user_emb, movie_emb], dim=-1)
        
        mean = self.encoder_mean(combined)
        logvar = self.encoder_logvar(combined)
        
        return mean, logvar
    
    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent distribution.
        
        Args:
            mean: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent variables
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent variables to edge probabilities.
        
        Args:
            z: Latent variables [batch_size, latent_dim]
            
        Returns:
            Edge probabilities [batch_size, 1]
        """
        return self.decoder(z)
    
    def forward(self, user_emb: torch.Tensor, movie_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through GraphVAE.
        
        Args:
            user_emb: User embeddings [batch_size, embedding_dim]
            movie_emb: Movie embeddings [batch_size, embedding_dim]
            
        Returns:
            Dictionary with reconstruction, mean, logvar, and latent variables
        """
        mean, logvar = self.encode(user_emb, movie_emb)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        
        return {
            'reconstruction': recon,
            'mean': mean,
            'logvar': logvar,
            'z': z
        }
    
    def generate_edges(self, 
                      user_emb: torch.Tensor, 
                      movie_emb: torch.Tensor,
                      threshold: float = 0.5) -> torch.Tensor:
        """
        Generate edges by sampling from the learned distribution.
        
        Args:
            user_emb: User embeddings
            movie_emb: Movie embeddings
            threshold: Probability threshold for edge existence
            
        Returns:
            Binary edge predictions
        """
        with torch.no_grad():
            output = self.forward(user_emb, movie_emb)
            probs = output['reconstruction']
            edges = (probs > threshold).float()
        
        return edges


class InnerProductDecoder(nn.Module):
    """
    Simple inner product decoder for link prediction.
    
    Computes edge probabilities as the sigmoid of the dot product
    between user and movie embeddings.
    """
    
    def __init__(self, dropout: float = 0.1):
        """
        Initialize inner product decoder.
        
        Args:
            dropout: Dropout probability
        """
        super(InnerProductDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, user_emb: torch.Tensor, movie_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute edge probabilities using inner product.
        
        Args:
            user_emb: User embeddings [batch_size, embedding_dim]
            movie_emb: Movie embeddings [batch_size, embedding_dim]
            
        Returns:
            Edge probabilities [batch_size, 1]
        """
        user_emb = self.dropout(user_emb)
        movie_emb = self.dropout(movie_emb)
        
        # Compute inner product
        scores = torch.sum(user_emb * movie_emb, dim=-1, keepdim=True)
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(scores)
        
        return probs


class MLPDecoder(nn.Module):
    """
    Multi-layer perceptron decoder for link prediction.
    
    Uses a feedforward network to predict edge probabilities
    from concatenated user and movie embeddings.
    """
    
    def __init__(self,
                 embedding_dim: int,
                 hidden_dims: List[int] = [64, 32],
                 dropout: float = 0.1):
        """
        Initialize MLP decoder.
        
        Args:
            embedding_dim: Dimension of input embeddings
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super(MLPDecoder, self).__init__()
        
        layers = []
        input_dim = embedding_dim * 2
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*layers)
        
        logger.info(f"Initialized MLP decoder with hidden_dims={hidden_dims}")
    
    def forward(self, user_emb: torch.Tensor, movie_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP decoder.
        
        Args:
            user_emb: User embeddings [batch_size, embedding_dim]
            movie_emb: Movie embeddings [batch_size, embedding_dim]
            
        Returns:
            Edge probabilities [batch_size, 1]
        """
        # Concatenate embeddings
        combined = torch.cat([user_emb, movie_emb], dim=-1)
        
        # Pass through MLP
        probs = self.mlp(combined)
        
        return probs


class AutoregressiveDecoder(nn.Module):
    """
    Autoregressive decoder for sequential edge generation.
    
    Generates edges one at a time, conditioning on previously
    generated edges to capture dependencies.
    """
    
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 max_edges: int = 100):
        """
        Initialize autoregressive decoder.
        
        Args:
            embedding_dim: Dimension of node embeddings
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            max_edges: Maximum number of edges to generate
        """
        super(AutoregressiveDecoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_edges = max_edges
        
        # LSTM for sequential modeling
        self.lstm = nn.LSTM(
            input_size=embedding_dim * 2 + 1,  # user_emb + movie_emb + prev_edge
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        logger.info(f"Initialized autoregressive decoder with {num_layers} LSTM layers")
    
    def forward(self, 
               user_emb: torch.Tensor,
               movie_emb: torch.Tensor,
               edge_sequence: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through autoregressive decoder.
        
        Args:
            user_emb: User embeddings [batch_size, seq_len, embedding_dim]
            movie_emb: Movie embeddings [batch_size, seq_len, embedding_dim]
            edge_sequence: Previous edge sequence [batch_size, seq_len, 1]
            
        Returns:
            Edge probabilities [batch_size, seq_len, 1]
        """
        batch_size, seq_len = user_emb.shape[:2]
        
        if edge_sequence is None:
            # Initialize with zeros (no previous edges)
            edge_sequence = torch.zeros(batch_size, seq_len, 1, 
                                      device=user_emb.device)
        
        # Combine embeddings and previous edges
        lstm_input = torch.cat([user_emb, movie_emb, edge_sequence], dim=-1)
        
        # Pass through LSTM
        lstm_output, _ = self.lstm(lstm_input)
        
        # Generate edge probabilities
        edge_probs = self.output_proj(lstm_output)
        
        return edge_probs
    
    def generate_sequence(self,
                         user_emb: torch.Tensor,
                         movie_emb: torch.Tensor,
                         temperature: float = 1.0) -> torch.Tensor:
        """
        Generate edge sequence autoregressively.
        
        Args:
            user_emb: User embeddings [batch_size, seq_len, embedding_dim]
            movie_emb: Movie embeddings [batch_size, seq_len, embedding_dim]
            temperature: Sampling temperature
            
        Returns:
            Generated edge sequence [batch_size, seq_len, 1]
        """
        batch_size, seq_len = user_emb.shape[:2]
        device = user_emb.device
        
        # Initialize hidden state
        h_0 = torch.zeros(self.lstm.num_layers, batch_size, self.hidden_dim, device=device)
        c_0 = torch.zeros(self.lstm.num_layers, batch_size, self.hidden_dim, device=device)
        hidden = (h_0, c_0)
        
        generated_edges = []
        prev_edge = torch.zeros(batch_size, 1, device=device)
        
        for t in range(seq_len):
            # Prepare input for current timestep
            current_input = torch.cat([
                user_emb[:, t:t+1, :],
                movie_emb[:, t:t+1, :],
                prev_edge.unsqueeze(1)
            ], dim=-1)
            
            # Forward pass
            lstm_out, hidden = self.lstm(current_input, hidden)
            edge_prob = self.output_proj(lstm_out)
            
            # Sample edge with temperature
            if temperature > 0:
                edge_prob = edge_prob / temperature
                edge = torch.bernoulli(torch.sigmoid(edge_prob))
            else:
                edge = (edge_prob > 0.5).float()
            
            generated_edges.append(edge)
            prev_edge = edge.squeeze(1)
        
        return torch.cat(generated_edges, dim=1).unsqueeze(-1)


class BilinearDecoder(nn.Module):
    """
    Bilinear decoder for link prediction.
    
    Uses a learnable bilinear transformation to compute
    compatibility scores between users and movies.
    """
    
    def __init__(self, embedding_dim: int, dropout: float = 0.1):
        """
        Initialize bilinear decoder.
        
        Args:
            embedding_dim: Dimension of input embeddings
            dropout: Dropout probability
        """
        super(BilinearDecoder, self).__init__()
        
        self.bilinear = nn.Bilinear(embedding_dim, embedding_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, user_emb: torch.Tensor, movie_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through bilinear decoder.
        
        Args:
            user_emb: User embeddings [batch_size, embedding_dim]
            movie_emb: Movie embeddings [batch_size, embedding_dim]
            
        Returns:
            Edge probabilities [batch_size, 1]
        """
        user_emb = self.dropout(user_emb)
        movie_emb = self.dropout(movie_emb)
        
        scores = self.bilinear(user_emb, movie_emb)
        probs = torch.sigmoid(scores)
        
        return probs


def vae_loss(recon_x: torch.Tensor,
             x: torch.Tensor,
             mean: torch.Tensor,
             logvar: torch.Tensor,
             beta: float = 1.0) -> Dict[str, torch.Tensor]:
    """
    Compute VAE loss with KL divergence and reconstruction terms.
    
    Args:
        recon_x: Reconstructed data
        x: Original data
        mean: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence term
        
    Returns:
        Dictionary with loss components
    """
    # Reconstruction loss (binary cross-entropy)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss
    }


def create_decoder(decoder_type: str,
                  embedding_dim: int,
                  **kwargs) -> nn.Module:
    """
    Factory function to create different types of decoders.
    
    Args:
        decoder_type: Type of decoder ('vae', 'inner_product', 'mlp', 'autoregressive', 'bilinear')
        embedding_dim: Dimension of input embeddings
        **kwargs: Additional arguments for specific decoders
        
    Returns:
        Initialized decoder module
    """
    if decoder_type == 'vae':
        return GraphVAEDecoder(
            embedding_dim=embedding_dim,
            hidden_dim=kwargs.get('hidden_dim', 32),
            latent_dim=kwargs.get('latent_dim', 16),
            dropout=kwargs.get('dropout', 0.1)
        )
    elif decoder_type == 'inner_product':
        return InnerProductDecoder(
            dropout=kwargs.get('dropout', 0.1)
        )
    elif decoder_type == 'mlp':
        return MLPDecoder(
            embedding_dim=embedding_dim,
            hidden_dims=kwargs.get('hidden_dims', [64, 32]),
            dropout=kwargs.get('dropout', 0.1)
        )
    elif decoder_type == 'autoregressive':
        return AutoregressiveDecoder(
            embedding_dim=embedding_dim,
            hidden_dim=kwargs.get('hidden_dim', 128),
            num_layers=kwargs.get('num_layers', 2),
            max_edges=kwargs.get('max_edges', 100)
        )
    elif decoder_type == 'bilinear':
        return BilinearDecoder(
            embedding_dim=embedding_dim,
            dropout=kwargs.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")
