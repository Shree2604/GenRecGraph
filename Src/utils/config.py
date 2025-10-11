"""
Configuration Management for GenRecGraph

This module provides utilities for managing data processing configurations,
hyperparameters, and experimental settings for the current data pipeline.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    data_path: str = "Data/"
    min_user_interactions: int = 20
    min_movie_interactions: int = 5
    rating_threshold: float = 3.0
    cold_start_ratio: float = 0.1
    test_ratio: float = 0.2
    val_ratio: float = 0.1
    include_features: bool = True
    graph_type: str = "bipartite"  # "bipartite" or "heterogeneous"


@dataclass
class VisualizationConfig:
    """Configuration for data visualization and analysis."""
    save_plots: bool = True
    plot_dpi: int = 300
    show_plots: bool = False
    output_dir: str = "output/visualizations"
    generate_statistics: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration for current data pipeline."""
    experiment_name: str = "data_pipeline_experiment"
    timestamp: str = None
    random_seed: int = 42
    device: str = "auto"  # For future model training

    # Data configuration
    data: DataConfig = None

    # Visualization configuration
    visualization: VisualizationConfig = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.data is None:
            self.data = DataConfig()
        if self.visualization is None:
            self.visualization = VisualizationConfig()


class ConfigManager:
    """Manager for handling configuration files and settings."""

    def __init__(self, config_dir: str = None):
        """
        Initialize configuration manager.

        Args:
            config_dir: Directory to store configuration files (None to disable file operations)
        """
        self.config_dir = Path(config_dir) if config_dir else None
        self.current_config = None

    def create_config(self,
                     experiment_name: str = None,
                     **kwargs) -> ExperimentConfig:
        """
        Create a new experiment configuration.

        Args:
            experiment_name: Name for the experiment
            **kwargs: Override default configuration values

        Returns:
            Configured ExperimentConfig object
        """
        config = ExperimentConfig()

        if experiment_name:
            config.experiment_name = experiment_name

        # Update data config
        if 'data' in kwargs:
            for key, value in kwargs['data'].items():
                if hasattr(config.data, key):
                    setattr(config.data, key, value)

        # Update visualization config
        if 'visualization' in kwargs:
            for key, value in kwargs['visualization'].items():
                if hasattr(config.visualization, key):
                    setattr(config.visualization, key, value)

        # Update experiment config
        for key, value in kwargs.items():
            if key in ['data', 'visualization']:
                continue
            if hasattr(config, key):
                setattr(config, key, value)

        self.current_config = config
        logger.info(f"Created configuration: {config.experiment_name}")
        return config

    def save_config(self, config: ExperimentConfig, filename: str = None) -> Optional[Path]:
        """
        Save configuration to a YAML file.

        Args:
            config: Configuration to save
            filename: Output filename (default: {experiment_name}_config.yaml)

        Returns:
            Path to saved configuration file, or None if config_dir is not set
        """
        if self.config_dir is None:
            logger.debug("Skipping config save: config_dir not set")
            return None

        if filename is None:
            filename = f"{config.experiment_name.lower().replace(' ', '_')}_config.yaml"

        self.config_dir.mkdir(exist_ok=True, parents=True)
        config_path = self.config_dir / filename
        config_dict = asdict(config)
        
        # Add metadata to config
        config_dict['metadata'] = {
            'created_at': config.timestamp,
            'version': '1.0',
            'pipeline_stage': 'data_processing_only'
        }

        # Save as YAML
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to: {config_path}")
        return config_path

    def load_config(self, config_path: str) -> ExperimentConfig:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Loaded ExperimentConfig object
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Extract experiment config
        exp_dict = config_dict['experiment']

        # Create configuration object
        config = ExperimentConfig()
        config.experiment_name = exp_dict['experiment_name']
        config.timestamp = exp_dict['timestamp']
        config.random_seed = exp_dict['random_seed']
        config.device = exp_dict['device']

        # Load data config
        if 'data' in config_dict:
            data_dict = config_dict['data']
            for key, value in data_dict.items():
                if hasattr(config.data, key):
                    setattr(config.data, key, value)

        # Load visualization config
        if 'visualization' in config_dict:
            viz_dict = config_dict['visualization']
            for key, value in viz_dict.items():
                if hasattr(config.visualization, key):
                    setattr(config.visualization, key, value)

        self.current_config = config
        logger.info(f"Configuration loaded from: {config_path}")
        return config

    def get_config_dict(self) -> Dict[str, Any]:
        """
        Get current configuration as dictionary.

        Returns:
            Dictionary representation of current configuration
        """
        if self.current_config is None:
            raise ValueError("No configuration loaded. Create or load a config first.")

        return {
            'experiment': asdict(self.current_config),
            'data': asdict(self.current_config.data),
            'visualization': asdict(self.current_config.visualization)
        }

    def print_config(self, config: ExperimentConfig = None):
        """
        Print configuration details.

        Args:
            config: Configuration to print (uses current_config if None)
        """
        if config is None:
            config = self.current_config

        if config is None:
            logger.info("No configuration to display.")
            return

        print(f"\n{'='*60}")
        print(f"EXPERIMENT CONFIGURATION: {config.experiment_name}")
        print(f"{'='*60}")
        print(f"Timestamp: {config.timestamp}")
        print(f"Random Seed: {config.random_seed}")
        print(f"Device: {config.device}")

        print(f"\n{'-'*40}")
        print("DATA CONFIGURATION:")
        print(f"{'-'*40}")
        print(f"Data Path: {config.data.data_path}")
        print(f"Min User Interactions: {config.data.min_user_interactions}")
        print(f"Min Movie Interactions: {config.data.min_movie_interactions}")
        print(f"Rating Threshold: {config.data.rating_threshold}")
        print(f"Graph Type: {config.data.graph_type}")
        print(f"Include Features: {config.data.include_features}")

        print(f"\n{'-'*40}")
        print("VISUALIZATION CONFIGURATION:")
        print(f"{'-'*40}")
        print(f"Save Plots: {config.visualization.save_plots}")
        print(f"Plot DPI: {config.visualization.plot_dpi}")
        print(f"Output Directory: {config.visualization.output_dir}")
        print(f"Generate Statistics: {config.visualization.generate_statistics}")

        print(f"\n{'='*60}\n")


# Global configuration manager instance
config_manager = ConfigManager()


def get_default_config() -> ExperimentConfig:
    """Get default experiment configuration for current data pipeline."""
    return ExperimentConfig()


def create_data_analysis_config() -> ExperimentConfig:
    """Create configuration for comprehensive data analysis experiments."""
    return config_manager.create_config(
        experiment_name="data_analysis_experiment",
        data={
            'min_user_interactions': 20,
            'min_movie_interactions': 5,
            'rating_threshold': 3.0,
            'include_features': True
        },
        visualization={
            'save_plots': True,
            'plot_dpi': 300,
            'generate_statistics': True
        }
    )
