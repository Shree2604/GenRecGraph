"""
Data validation utilities for GenRecGraph

This module provides utilities for validating processed data and generated graphs.
"""

import pandas as pd
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


def validate_data_integrity(data_splits: Dict, metadata: Dict) -> Dict:
    """
    Validate the integrity of processed MovieLens data.

    Args:
        data_splits: Dictionary containing train/val/test data splits
        metadata: Dictionary containing movies and encoding info

    Returns:
        Dictionary with validation results and statistics
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }

    try:
        # Check data splits structure
        required_splits = ['train', 'val', 'test']
        for split in required_splits:
            if split not in data_splits:
                validation_results['errors'].append(f"Missing data split: {split}")
                validation_results['is_valid'] = False

        if not validation_results['is_valid']:
            return validation_results

        # Validate each split
        for split_name, split_data in data_splits.items():
            if not isinstance(split_data, pd.DataFrame):
                validation_results['errors'].append(f"Split {split_name} is not a DataFrame")
                validation_results['is_valid'] = False
                continue

            # Check required columns
            required_cols = ['user_idx', 'movie_idx', 'rating']
            for col in required_cols:
                if col not in split_data.columns:
                    validation_results['errors'].append(f"Missing column '{col}' in {split_name}")
                    validation_results['is_valid'] = False

            # Validate data types
            if split_data['user_idx'].dtype != 'int64':
                validation_results['warnings'].append(f"User indices not int64 in {split_name}")
            if split_data['movie_idx'].dtype != 'int64':
                validation_results['warnings'].append(f"Movie indices not int64 in {split_name}")
            if not pd.api.types.is_numeric_dtype(split_data['rating']):
                validation_results['errors'].append(f"Rating column not numeric in {split_name}")
                validation_results['is_valid'] = False

            # Validate rating range
            if split_data['rating'].min() < 0.5 or split_data['rating'].max() > 5.0:
                validation_results['warnings'].append(f"Rating values outside [0.5, 5.0] in {split_name}")

        # Validate metadata
        if 'movies' not in metadata:
            validation_results['errors'].append("Missing movies in metadata")
            validation_results['is_valid'] = False

        if 'encoding_info' not in metadata:
            validation_results['errors'].append("Missing encoding_info in metadata")
            validation_results['is_valid'] = False

        # Check for user/movie ID consistency
        if 'encoding_info' in metadata:
            enc_info = metadata['encoding_info']
            if 'num_users' in enc_info and 'num_movies' in enc_info:
                # Check that all user/movie indices are within valid ranges
                for split_name, split_data in data_splits.items():
                    max_user = split_data['user_idx'].max()
                    max_movie = split_data['movie_idx'].max()

                    if max_user >= enc_info['num_users']:
                        validation_results['errors'].append(f"User index {max_user} >= num_users in {split_name}")
                        validation_results['is_valid'] = False

                    if max_movie >= enc_info['num_movies']:
                        validation_results['errors'].append(f"Movie index {max_movie} >= num_movies in {split_name}")
                        validation_results['is_valid'] = False

        # Calculate statistics
        validation_results['statistics'] = {
            'total_interactions': sum(len(split) for split in data_splits.values()),
            'unique_users': len(set().union(*[set(split['user_idx']) for split in data_splits.values()])),
            'unique_movies': len(set().union(*[set(split['movie_idx']) for split in data_splits.values()])),
            'avg_rating': np.mean([split['rating'].mean() for split in data_splits.values()]),
            'rating_std': np.mean([split['rating'].std() for split in data_splits.values()]),
        }

    except Exception as e:
        validation_results['errors'].append(f"Validation error: {str(e)}")
        validation_results['is_valid'] = False

    return validation_results


def validate_graph_structure(graph) -> Dict:
    """
    Validate the structure and integrity of the generated bipartite graph.

    Args:
        graph: PyTorch Geometric Data object

    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }

    try:
        # Check required attributes
        required_attrs = ['x', 'edge_index', 'edge_attr']
        for attr in required_attrs:
            if not hasattr(graph, attr):
                validation_results['errors'].append(f"Missing graph attribute: {attr}")
                validation_results['is_valid'] = False

        if not validation_results['is_valid']:
            return validation_results

        # Validate edge_index shape
        if graph.edge_index.shape[0] != 2:
            validation_results['errors'].append(f"edge_index should have shape (2, num_edges), got {graph.edge_index.shape}")
            validation_results['is_valid'] = False

        # Validate node features shape
        num_nodes = graph.edge_index.max().item() + 1
        if graph.x.shape[0] != num_nodes:
            validation_results['warnings'].append(f"Node features shape {graph.x.shape[0]} != expected {num_nodes}")

        # Check for isolated nodes
        edge_index_flat = graph.edge_index.flatten()
        unique_nodes = torch.unique(edge_index_flat)
        if len(unique_nodes) != num_nodes:
            validation_results['warnings'].append(f"Found {len(unique_nodes)} connected nodes out of {num_nodes} total")

        # Calculate graph statistics
        num_edges = graph.edge_index.shape[1]
        if hasattr(graph, 'num_users') and hasattr(graph, 'num_movies'):
            expected_nodes = graph.num_users + graph.num_movies
            density = num_edges / (graph.num_users * graph.num_movies) if graph.num_users * graph.num_movies > 0 else 0

            validation_results['statistics'] = {
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'num_users': graph.num_users,
                'num_movies': graph.num_movies,
                'density': density,
                'avg_degree': 2 * num_edges / num_nodes if num_nodes > 0 else 0,
            }

        # Check for self-loops
        edge_index_np = graph.edge_index.cpu().numpy()
        self_loops = np.sum(edge_index_np[0] == edge_index_np[1])
        if self_loops > 0:
            validation_results['errors'].append(f"Found {self_loops} self-loops in graph")
            validation_results['is_valid'] = False

        # Check edge_index range
        if edge_index_np.min() < 0:
            validation_results['errors'].append("Negative indices found in edge_index")
            validation_results['is_valid'] = False

        if edge_index_np.max() >= num_nodes:
            validation_results['errors'].append(f"Edge index {edge_index_np.max()} >= num_nodes {num_nodes}")
            validation_results['is_valid'] = False

    except Exception as e:
        validation_results['errors'].append(f"Graph validation error: {str(e)}")
        validation_results['is_valid'] = False

    return validation_results


def run_comprehensive_validation(data_splits: Dict, metadata: Dict, graph) -> Dict:
    """
    Run comprehensive validation on all components.

    Args:
        data_splits: Processed data splits
        metadata: Metadata dictionary
        graph: Generated bipartite graph

    Returns:
        Comprehensive validation report
    """
    logger.info("Running comprehensive data validation...")

    # Validate data integrity
    data_validation = validate_data_integrity(data_splits, metadata)

    # Validate graph structure
    graph_validation = validate_graph_structure(graph)

    # Combine results
    combined_results = {
        'overall_valid': data_validation['is_valid'] and graph_validation['is_valid'],
        'data_validation': data_validation,
        'graph_validation': graph_validation,
        'summary': {
            'total_errors': len(data_validation['errors']) + len(graph_validation['errors']),
            'total_warnings': len(data_validation['warnings']) + len(graph_validation['warnings']),
        }
    }

    # Log results
    if combined_results['overall_valid']:
        logger.info("All validations passed successfully.")
    else:
        logger.error(f"Validation failed: {combined_results['total_errors']} errors, {combined_results['total_warnings']} warnings")

    for error in data_validation['errors'] + graph_validation['errors']:
        logger.error(f"  Error: {error}")

    for warning in data_validation['warnings'] + graph_validation['warnings']:
        logger.warning(f"  Warning: {warning}")

    return combined_results
