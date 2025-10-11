"""
Visualization utilities for GenRecGraph

This module provides comprehensive data visualization and analysis capabilities
for MovieLens dataset and generated bipartite graphs.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import logging
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random

logger = logging.getLogger(__name__)


def create_visualizations(data_splits, metadata, graph, output_dir):
    """Create comprehensive visualizations of the processed data and graph."""

    # Create visualizations directory
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)

    logger.info("Creating data statistics visualizations...")

    # 1. Dataset Statistics Plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('MovieLens-25M Dataset Statistics', fontsize=16, fontweight='bold')

    # Ratings distribution
    ratings = data_splits['train']['rating'].value_counts().sort_index()
    axes[0, 0].bar(ratings.index, ratings.values, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Rating Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Rating')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(True, alpha=0.3)

    # User activity distribution
    user_activity = data_splits['train']['user_idx'].value_counts()
    axes[0, 1].hist(user_activity.values, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('User Activity Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Number of Ratings')
    axes[0, 1].set_ylabel('Number of Users')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)

    # Movie popularity distribution
    movie_popularity = data_splits['train']['movie_idx'].value_counts()
    axes[1, 0].hist(movie_popularity.values, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title('Movie Popularity Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Number of Ratings')
    axes[1, 0].set_ylabel('Number of Movies')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)

    # Genre distribution
    if 'genres_list' in metadata['movies'].columns:
        genre_counts = {}
        for genres in metadata['movies']['genres_list'].dropna():
            if isinstance(genres, list):
                for genre in genres:
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1

        genres = list(genre_counts.keys())
        counts = list(genre_counts.values())
        axes[1, 1].bar(range(len(genres)), counts, color='gold', edgecolor='black', alpha=0.7)
        axes[1, 1].set_title('Genre Distribution', fontweight='bold')
        axes[1, 1].set_xlabel('Genre')
        axes[1, 1].set_ylabel('Number of Movies')
        axes[1, 1].set_xticks(range(len(genres)))
        axes[1, 1].set_xticklabels(genres, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(viz_dir / 'dataset_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Temporal Analysis
    plt.figure(figsize=(15, 6))

    # Ratings over time
    ratings_by_year = data_splits['train'].copy()
    ratings_by_year['year'] = pd.to_datetime(ratings_by_year['timestamp'], unit='s').dt.year
    yearly_counts = ratings_by_year.groupby('year').size()

    plt.subplot(1, 2, 1)
    yearly_counts.plot(kind='bar', color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Ratings Distribution by Year', fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Number of Ratings')
    plt.grid(True, alpha=0.3)

    # Average rating by year
    plt.subplot(1, 2, 2)
    yearly_avg_rating = ratings_by_year.groupby('year')['rating'].mean()
    yearly_avg_rating.plot(kind='line', marker='o', color='darkblue', linewidth=2)
    plt.title('Average Rating by Year', fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Average Rating')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(viz_dir / 'temporal_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Graph Structure Visualizations
    logger.info("Creating graph structure visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Bipartite Graph Structure Analysis', fontsize=16, fontweight='bold')

    # Node degree distributions
    edge_index = graph.edge_index.cpu().numpy()

    # User degrees (first num_users nodes)
    user_nodes = edge_index[0][edge_index[0] < graph.num_users]
    user_degrees = np.bincount(user_nodes, minlength=graph.num_users)
    user_degrees = user_degrees[user_degrees > 0]  # Remove zero-degree nodes

    # Movie degrees (nodes after num_users)
    movie_nodes = edge_index[1][edge_index[1] >= graph.num_users] - graph.num_users
    movie_degrees = np.bincount(movie_nodes, minlength=graph.num_movies)
    movie_degrees = movie_degrees[movie_degrees > 0]  # Remove zero-degree nodes

    # User degree distribution
    axes[0, 0].hist(user_degrees, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('User Degree Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Degree')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xscale('log')
    axes[0, 0].grid(True, alpha=0.3)

    # Movie degree distribution
    axes[0, 1].hist(movie_degrees, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Movie Degree Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Degree')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_xscale('log')
    axes[0, 1].grid(True, alpha=0.3)

    # Graph density and connectivity
    num_possible_edges = graph.num_users * graph.num_movies
    actual_edges = graph.num_edges // 2  # Since bidirectional
    density = actual_edges / num_possible_edges

    axes[1, 0].text(0.5, 0.5,
                   f'Graph Statistics:\n\n'
                   f'Total Nodes: {graph.num_nodes:,}\n'
                   f'Total Edges: {graph.num_edges:,}\n'
                   f'Users: {graph.num_users:,}\n'
                   f'Movies: {graph.num_movies:,}\n'
                   f'Density: {density:.6f}\n'
                   f'Avg User Degree: {user_degrees.mean():.2f}\n'
                   f'Avg Movie Degree: {movie_degrees.mean():.2f}',
                   transform=axes[1, 0].transAxes,
                   fontsize=12,
                   verticalalignment='center',
                   horizontalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[1, 0].set_title('Graph Statistics', fontweight='bold')
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])

    # Rating distribution in graph edges
    edge_ratings = graph.edge_attr.cpu().numpy().flatten()
    axes[1, 1].hist(edge_ratings, bins=np.arange(0.5, 5.5, 0.5),
                   color='gold', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Edge Rating Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('Rating')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(viz_dir / 'graph_structure.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Heatmap of user-movie interactions (sample) - REMOVED due to performance issues
    # logger.info("Creating interaction heatmap...")
    # plt.figure(figsize=(12, 8))
    #
    # # Sample a subset for visualization
    # sample_size = min(100, min(graph.num_users, graph.num_movies))
    # sample_users = np.random.choice(graph.num_users, size=sample_size, replace=False)
    # sample_movies = np.random.choice(graph.num_movies, size=sample_size, replace=False)
    #
    # # Create interaction matrix
    # interaction_matrix = np.zeros((sample_size, sample_size))
    #
    # edge_index_np = graph.edge_index.cpu().numpy()
    # for i in range(edge_index_np.shape[1]):
    #     user_idx = edge_index_np[0, i]
    #     movie_idx = edge_index_np[1, i] - graph.num_users
    #
    #     if user_idx in sample_users and movie_idx in sample_movies:
    #         user_pos = np.where(sample_users == user_idx)[0][0]
    #         movie_pos = np.where(sample_movies == movie_idx)[0][0]
    #         interaction_matrix[user_pos, movie_pos] = 1
    #
    # # Create heatmap
    # sns.heatmap(interaction_matrix,
    #             cmap='Blues',
    #             cbar_kws={'label': 'Interaction'},
    #             square=True)
    # plt.title('User-Movie Interaction Heatmap (Sample)', fontweight='bold')
    # plt.xlabel('Movies (Sample)')
    # plt.ylabel('Users (Sample)')
    # plt.tight_layout()
    # plt.savefig(viz_dir / 'interaction_heatmap.png', dpi=300, bbox_inches='tight')
    # plt.close()

    # 4. Summary statistics text file (moved up)
    logger.info("Creating summary statistics...")
    stats_file = viz_dir / 'statistics_summary.txt'

    with open(stats_file, 'w') as f:
        f.write("GenRecGraph Dataset and Graph Statistics\n")
        f.write("=" * 50 + "\n\n")

        f.write("DATASET STATISTICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Ratings: {len(data_splits['train']):,}\n")
        f.write(f"Unique Users: {data_splits['train']['user_idx'].nunique():,}\n")
        f.write(f"Unique Movies: {data_splits['train']['movie_idx'].nunique():,}\n")
        f.write(f"Rating Range: {data_splits['train']['rating'].min()} - {data_splits['train']['rating'].max()}\n")
        f.write(f"Average Rating: {data_splits['train']['rating'].mean():.2f}\n\n")

        f.write("GRAPH STATISTICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Nodes: {graph.num_nodes:,}\n")
        f.write(f"Total Edges: {graph.num_edges:,}\n")
        f.write(f"Users: {graph.num_users:,}\n")
        f.write(f"Movies: {graph.num_movies:,}\n")
        f.write(f"Graph Density: {density:.6f}\n")
        f.write(f"Average User Degree: {user_degrees.mean():.2f}\n")
        f.write(f"Average Movie Degree: {movie_degrees.mean():.2f}\n")
        f.write(f"Maximum User Degree: {user_degrees.max()}\n")
        f.write(f"Maximum Movie Degree: {movie_degrees.max()}\n")
    plt.close()
    
    # 5. Bipartite Graph Visualization (Sample)
    logger.info("Creating bipartite graph visualization...")
    try:
        plot_bipartite_graph_sample(
            data_splits['train'], 
            metadata['movies'],
            n_users=10,  # Number of users to sample
            n_movies=15,  # Number of movies to sample
            output_path=viz_dir / 'bipartite_graph_sample.png'
        )
    except Exception as e:
        logger.warning(f"Could not create bipartite graph visualization: {e}")

    logger.info("All visualizations saved to %s", viz_dir)
    logger.info(f"Summary statistics saved to {stats_file}")

    return viz_dir

def plot_bipartite_graph_sample(ratings_df, movies_df, n_users=10, n_movies=15, output_path=None):
    """
    Visualize a sample of the bipartite graph showing user-movie interactions.
    
    Args:
        ratings_df: DataFrame containing user-movie ratings (with user_idx, movie_idx columns)
        movies_df: DataFrame containing movie information (with movieId, title columns)
        n_users: Number of users to include in the sample
        n_movies: Number of movies to include in the sample
        output_path: Path to save the visualization (if None, shows the plot)
    """
    # Sample users and movies
    user_sample = random.sample(sorted(ratings_df['user_idx'].unique().tolist()), 
                              min(n_users, len(ratings_df['user_idx'].unique())))
    
    # Get movies that these users have rated
    user_movies = ratings_df[ratings_df['user_idx'].isin(user_sample)]['movie_idx'].unique()
    
    # If we have fewer movies than requested, take all
    if len(user_movies) > n_movies:
        movie_sample = random.sample(list(user_movies), n_movies)
    else:
        movie_sample = user_movies
    
    # Filter ratings for our sample
    sample_ratings = ratings_df[
        (ratings_df['user_idx'].isin(user_sample)) & 
        (ratings_df['movie_idx'].isin(movie_sample))
    ]
    
    # Create mapping from movie_idx to movieId and then to title
    # Get unique movieIds from the ratings data
    movie_idx_to_id = dict(zip(ratings_df['movie_idx'], ratings_df['movieId']))
    movie_id_to_title = dict(zip(movies_df['movieId'], movies_df['title']))
    
    # Create movie_idx to title mapping
    movie_titles = {}
    for idx in movie_sample:
        movie_id = movie_idx_to_id.get(idx)
        if movie_id is not None:
            title = movie_id_to_title.get(movie_id, f"Movie {idx}")
            # Truncate long titles
            movie_titles[idx] = title[:30] + '...' if len(title) > 30 else title
        else:
            movie_titles[idx] = f"Movie {idx}"
    
    # Create a bipartite graph
    B = nx.Graph()
    
    # Add nodes with the node attribute "bipartite"
    B.add_nodes_from([f"U{u}" for u in user_sample], bipartite=0, type='user')
    B.add_nodes_from([f"M{m}" for m in movie_sample], bipartite=1, type='movie')
    
    # Add edges with weights based on ratings
    for _, row in sample_ratings.iterrows():
        B.add_edge(f"U{row['user_idx']}", f"M{row['movie_idx']}", weight=row['rating'])
    
    # Separate by group
    users = {n for n, d in B.nodes(data=True) if d["bipartite"] == 0}
    movies = set(B) - users
    
    # Set up the figure
    plt.figure(figsize=(14, 10))
    
    # Position nodes in two columns
    pos = {}
    
    # Position user nodes in left column
    for i, node in enumerate(sorted(users)):
        pos[node] = (0, i)
    
    # Position movie nodes in right column
    for i, node in enumerate(sorted(movies, key=lambda x: -B.degree(x))):
        pos[node] = (1, i * (len(users) / len(movies)))
    
    # Draw nodes
    nx.draw_networkx_nodes(B, pos, nodelist=users, node_color='lightblue', 
                          node_size=500, label='Users')
    nx.draw_networkx_nodes(B, pos, nodelist=movies, node_color='lightgreen', 
                          node_size=500, node_shape='s', label='Movies')
    
    # Draw edges with width based on rating
    edges = B.edges(data=True)
    edge_weights = [e[2]['weight'] for e in edges]
    nx.draw_networkx_edges(B, pos, edgelist=edges, 
                          width=[(w/5.0)*2 for w in edge_weights], 
                          alpha=0.7, edge_color='gray')
    
    # Add labels
    user_labels = {node: f"User\n{node[1:]}" for node in users}
    movie_labels = {node: movie_titles.get(int(node[1:]), f"Movie {node[1:]}") for node in movies}
    
    nx.draw_networkx_labels(B, pos, labels=user_labels, font_size=8, font_weight='bold')
    nx.draw_networkx_labels(B, pos, labels=movie_labels, font_size=7, 
                           bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5))
    
    # Add edge weight labels (ratings)
    edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in edges}
    nx.draw_networkx_edge_labels(B, pos, edge_labels=edge_labels, font_size=7)
    
    # Add legend
    plt.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
               markersize=10, label='Users'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgreen', 
               markersize=10, label='Movies')
    ], loc='upper right')
    
    plt.title(f'Bipartite Graph Sample\n{len(users)} Users - {len(movies)} Movies - {len(edges)} Edges', 
              fontweight='bold', pad=20)
    plt.axis('off')
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
