"""
GenRecGraph Demo: Model Input/Output Example

This script demonstrates how to use the GenRecGraph model to make predictions.
It shows the input format and the corresponding output predictions.
"""
import os
import torch
import numpy as np
from torch_geometric.data import Data
from Src.models.genrecgraph import create_genrecgraph_model
from Src.models.encoders import create_encoder
from Src.models.decoders import create_decoder

# Configuration matching your setup
config = {
    'model': {
        'encoder': {
            'type': 'sage',  # Using GraphSAGE as per your setup
            'input_dim': 21,
            'hidden_dims': [128, 21],
            'output_dim': 21,
            'dropout': 0.1
        },
        'decoder': {
            'type': 'mlp',  # Example with MLP decoder
            'hidden_dims': [128, 64],
            'dropout': 0.1
        }
    },
    'num_users': 1000,  # Example number of users
    'num_movies': 2000,  # Example number of movies
    'embedding_dim': 21
}

def create_sample_data():
    """Create sample input data for demonstration."""
    # Number of nodes (users + movies)
    num_users = config['num_users']
    num_movies = config['num_movies']
    num_nodes = num_users + num_movies
    
    # Sample node features (21-dimensional as per your setup)
    x = torch.randn((num_nodes, config['model']['encoder']['input_dim']))
    
    # Sample edges (user-movie interactions)
    num_edges = 5000
    user_indices = torch.randint(0, num_users, (num_edges,))
    movie_indices = torch.randint(num_users, num_users + num_movies, (num_edges,))
    edge_index = torch.stack([user_indices, movie_indices], dim=0)
    
    # Create PyG Data object
    graph = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
    
    return graph, num_users, num_movies

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create sample data
    graph, num_users, num_movies = create_sample_data()
    graph = graph.to(device)
    
    # Create model
    model = create_genrecgraph_model({
        **config,
        'num_users': num_users,
        'num_movies': num_movies
    }).to(device)
    
    # Print model architecture
    print("\nModel Architecture:")
    print(model)
    
    # Show input example
    print("\nInput Example:")
    print(f"Graph nodes (first 5):\n{graph.x[:5].cpu().numpy()}")
    print(f"Graph edges (first 5):\n{graph.edge_index[:, :5].cpu().numpy()}")
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        # Get embeddings
        embeddings = model.encoder(graph.x, graph.edge_index)
        
        # Sample some user-movie pairs to predict
        test_users = torch.tensor([0, 1, 2], device=device)  # First 3 users
        test_movies = torch.tensor([num_users, num_users+1, num_users+2], device=device)  # First 3 movies
        
        # Get predictions
        user_embs = embeddings[test_users]
        movie_embs = embeddings[test_movies]
        predictions = model.decoder(user_embs, movie_embs)
        
        # Apply sigmoid to get probabilities
        if not isinstance(predictions, dict):  # Some decoders return dicts
            predictions = torch.sigmoid(predictions)
        
        print("\nExample Predictions:")
        for i, (user, movie) in enumerate(zip(test_users, test_movies)):
            print(f"User {user.item()} - Movie {movie.item() - num_users}: "
                  f"Predicted rating = {predictions[i].item():.4f}")

if __name__ == "__main__":
    main()
