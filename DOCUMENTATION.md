# GenRecGraph: Comprehensive Project Documentation

## 📋 Project Overview

GenRecGraph is a recommendation system project that uses generative graph neural networks for cold-start recommendation problems. The project focuses on addressing the challenge of making recommendations for new users and items with no historical interaction data.

## 🏗️ Current Project Status

### ✅ Completed Components

#### 1. **Project Setup & Configuration**
- **`.gitignore`**: Configured to ignore CSV files in the `Data/` folder while preserving documentation files
- **`requirements.txt`**: Minimal, clean dependency list with only actually used packages:
  - `pandas`, `numpy`, `torch` - Core data processing and ML
  - `torch-geometric` - Graph neural networks
  - `scikit-learn` - Data preprocessing
  - `networkx` - Graph operations

#### 2. **Data Management**
- **MovieLens-25M Dataset Integration**: Full support for the MovieLens-25M dataset
- **Data Loading Pipeline** (`Src/datapipe/movielens_loader.py`):
  - Loads all dataset files: ratings, movies, tags, links, genome data
  - Handles large files efficiently with chunked reading
  - Provides data filtering and preprocessing capabilities
  - Encodes user/movie IDs for graph construction
  - Creates temporal train/validation/test splits with cold-start scenarios

- **Graph Construction** (`Src/datapipe/graph_builder.py`):
  - Creates bipartite user-item graphs for GNN training
  - Supports both homogeneous and heterogeneous graph representations
  - Generates rich node features from user behavior and movie content
  - Handles edge attributes (ratings) and negative sampling
  - Provides subgraph extraction utilities

#### 3. **Data Documentation**
- **Enhanced `Data/README.txt`**: Added clear setup instructions for new users
- **Main usage script** (`main.py`): Demonstrates complete pipeline from data loading to graph creation

## 📁 Project Structure

```
GenRecGraph/
├── .gitignore              # Ignores CSV files, keeps documentation
├── requirements.txt        # Minimal dependencies (only used packages)
├── main.py                 # Complete usage example
├── README.md              # Project description and research context
├── LICENSE                # MIT License
├── Data/                  # MovieLens dataset location
│   ├── README.txt         # Dataset documentation + setup instructions
│   └── *.csv              # MovieLens data files (gitignored)
└── Src/
    └── datapipe/          # Data loading and processing
        ├── __init__.py    # Module exports
        ├── movielens_loader.py  # Dataset loading and preprocessing
        └── graph_builder.py     # Graph construction utilities
```

## 🔧 Technical Implementation

### Data Loading Pipeline

```python
from Src.datapipe.movielens_loader import load_movielens_data

# Load and preprocess MovieLens data
data_splits, metadata = load_movielens_data(
    data_path="Data/",
    filter_params={
        'min_user_interactions': 20,
        'min_movie_interactions': 5,
        'rating_threshold': 3.0
    }
)
```

**Features:**
- Filters users/movies by interaction thresholds
- Encodes IDs for continuous indexing
- Creates temporal splits for evaluation
- Handles cold-start scenarios

### Graph Construction

```python
from Src.datapipe.graph_builder import create_graph_from_ratings

# Create bipartite graph
graph = create_graph_from_ratings(
    ratings=data_splits['train'],
    movies=metadata['movies'],
    encoding_info=metadata['encoding_info']
)
```

**Features:**
- Bipartite user-item graph structure
- Rich node features (user behavior, movie content)
- Edge attributes (ratings)
- PyTorch Geometric compatibility

## 🚀 Usage Instructions

### 1. **Setup**
```bash
# Install dependencies
pip install -r requirements.txt

# Download MovieLens-25M dataset
# Place CSV files in Data/ folder
# Files will be automatically gitignored
```

### 2. **Basic Usage**
```bash
python main.py
```

The script will:
- Load and preprocess MovieLens data
- Create bipartite graph
- **Generate comprehensive visualizations and statistics**
- Save processed data and graph for model training

**Generated Visualizations:**
- `output/visualizations/dataset_statistics.png` - Rating distributions, user/movie activity, genre distribution
- `output/visualizations/temporal_analysis.png` - Ratings over time, average rating trends
- `output/visualizations/graph_structure.png` - Degree distributions, graph statistics, edge ratings
- `output/visualizations/statistics_summary.txt` - Complete numerical statistics

**Saved Data:**
- `output/preprocessed_data.pkl` - Train/validation/test splits
- `output/metadata.pkl` - Encoding info and movie data
- `output/bipartite_graph.pkl` - PyTorch Geometric graph object
- `output/graph_tensors.pt` - Graph tensors for easy loading


**✅ Data Processing Pipeline (Complete)**
- **MovieLens-25M Dataset Loading**: Efficient loading and preprocessing of all dataset files
- **Data Filtering & Encoding**: Quality filtering, ID encoding, and temporal splitting
- **Bipartite Graph Construction**: PyTorch Geometric compatible graphs with rich features
- **Comprehensive Visualizations**: Statistical plots, graph structure analysis, and data insights
- **Configuration Management**: Centralized hyperparameter and experiment configuration system 

**See the [DOCUMENTATION](DOCUMENTATION.md) for detailed technical information about the current implementation.**

## 🔬 Research Context

This project implements the foundation for:
- **Graph neural networks**: Learning from user-item interaction graphs
- **Generative models**: Predicting plausible new interactions
- **Multiple scenarios**: User cold-start, item cold-start, sparse data

## 🛠️ Missing Components (Future Development)

The current implementation provides the **data pipeline**. To complete the full GenRecGraph system, these components are planned:

1. **Model Architecture** (`Src/models/`):
   - GNN encoders (GCN, GAT, GraphSAGE)
   - Generative decoders (VAE, autoregressive, bilinear)
   - Main GenRecGraph model class

2. **Training Infrastructure** (`Src/utils/`):
   - ~~Configuration management~~ ✅ **IMPLEMENTED**: `utils/config.py` (data processing & visualization only)
   - Evaluation metrics (Precision@K, NDCG, Coverage)
   - Training loops and optimization

3. **Experiment Framework** (`experiments/`):
   - Configuration files for different scenarios
   - Hyperparameter tuning
   - Results tracking and visualization

4. **Documentation & Examples** (`notebooks/`):
   - Usage tutorials
   - Model interpretation
   - Performance analysis

## 🎯 Current Capabilities

✅ **Data ingestion and preprocessing**
✅ **Graph construction for GNN training**
✅ **Cold-start scenario preparation**
✅ **Comprehensive visualizations and analysis**
✅ **Configuration management system** 🔄 **NEW!**
✅ **Clean, modular code structure**

## 📈 Next Steps

To continue development:

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Run the pipeline**: `python main.py`
3. **Develop model components** in `Src/models/`
4. **Add training scripts** for end-to-end experimentation

This foundation provides everything needed to build state-of-the-art generative recommendation models for cold-start problems.
