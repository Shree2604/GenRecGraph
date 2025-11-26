<div align="center">

# GenRecGraph: Generative Graph Models for Cold-Start Recommendation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.0%2B-3A4E8D?logo=pytorch&logoColor=white)](https://pytorch-geometric.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

A comprehensive framework for addressing cold-start problems in recommendation systems using generative graph neural networks. This project implements various GNN architectures combined with generative decoders to predict and generate user-item interactions for new users and items.

## 🚀 Features

- **Multiple GNN Architectures**:
  - Graph Convolutional Networks (GCN)
  - Graph Attention Networks (GAT)
  - GraphSAGE (for inductive learning)

- **Advanced Decoders**:
  - Graph Variational Autoencoder (GraphVAE)
  - Autoregressive Decoder
  - Bilinear Decoder
  - MLP Decoder

- **End-to-End Pipeline**:
  - Data loading and preprocessing
  - Graph construction and feature engineering
  - Model training and evaluation
  - Cold-start recommendation generation

## 🏗️ Architecture

### Implemented Components

**1. Data Processing Pipeline**
- MovieLens-25M Dataset loading and preprocessing
- Data filtering, cleaning, and feature engineering
- Bipartite graph construction with PyTorch Geometric
- Train/validation/test split with temporal holdout

**2. Graph Neural Network Encoders**
- **GCN**: Graph Convolutional Networks
- **GAT**: Graph Attention Networks
- **GraphSAGE**: For inductive learning scenarios

**3. Generative Decoders**
- **GraphVAE**: Variational Autoencoder for graph generation
- **Autoregressive**: Sequential prediction of edges
- **Bilinear**: Efficient dot-product based decoder
- **MLP**: Multi-layer perceptron decoder

**4. Training & Evaluation**
- Custom training loops with early stopping
- Comprehensive evaluation metrics
- Cold-start specific evaluation protocols
- Model checkpointing and logging

3. **Training Infrastructure**: Complete ML pipeline
   - Configuration management and hyperparameter tuning
   - Training loops with early stopping and validation
   - Comprehensive evaluation metrics for cold-start scenarios

## Dataset

**MovieLens-25M**: A large-scale benchmark dataset containing:
- 25M+ ratings from 162K+ users
- 62K+ movies with rich metadata
- Genre information, tags, and temporal data
- External knowledge graph integration support

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 1.10+ (with CUDA if using GPU)
- PyTorch Geometric
- Other dependencies in `requirements.txt`

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Shree2604/GenRecGraph.git
   cd GenRecGraph
   ```

2. **Set up a virtual environment (recommended)**
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install PyTorch with CUDA support (recommended for GPU)**
   ```bash
   # For CUDA 11.3
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113
   ```

4. **Install PyTorch Geometric and other dependencies**
   ```bash
   # Install PyTorch Geometric
   pip install torch-geometric
   
   # Install project dependencies
   pip install -r requirements.txt
   ```

### Running the Code

1. **Prepare the dataset**
   ```bash
   # Download and extract MovieLens-25M to Data/ directory
   mkdir -p Data
   wget https://files.grouplens.org/datasets/movielens/ml-25m.zip -P Data/
   unzip Data/ml-25m.zip -d Data/
   ```

2. **Run training**
   ```bash
   # Basic training with default parameters
   python main.py
   
   # Example: Train GAT model with Bilinear decoder
   python main.py --model gat --decoder bilinear --epochs 100 --batch_size 1024 --hidden_dim 64
   
   # Enable GPU training (if available)
   python main.py --use_cuda
   ```

3. **Evaluate a trained model**
   ```bash
   python evaluate.py --model_path output/best_model.pt --test_batch_size 1024
   ```

4. **Generate recommendations**
   ```bash
   python recommend.py --user_id 123 --top_k 10 --model_path output/best_model.pt
   ```

### Usage

1. **Prepare the dataset**
   - Download the [MovieLens-25M dataset](https://grouplens.org/datasets/movielens/25m/)
   - Extract the contents to the `Data/` directory

2. **Run the training pipeline**
   ```bash
   python main.py --model gcn --decoder bilinear --epochs 100
   ```

3. **Evaluate the model**
   ```bash
   python evaluate.py --model_path output/best_model.pt
   ```

### Available Command Line Arguments

```
--model [gcn|gat|graphsage]  # Choose GNN architecture
--decoder [vae|autoregressive|bilinear|mlp]  # Select decoder type
--epochs N                   # Number of training epochs
--batch_size N               # Batch size for training
--learning_rate FLOAT        # Learning rate
--hidden_dim N               # Hidden dimension size
--dropout FLOAT             # Dropout rate
--use_cuda                  # Use GPU if available
```

## 🗂️ Project Structure

```
GenRecGraph/
├── main.py                    # Main training script
├── evaluate.py                # Model evaluation script
├── requirements.txt           # Project dependencies
│
├── Src/
│   ├── models/               # Model implementations
│   │   ├── encoders.py       # GNN encoders (GCN, GAT, GraphSAGE)
│   │   ├── decoders.py       # Decoder architectures
│   │   └── genrecgraph.py    # Main model class
│   │
│   ├── datapipe/             # Data processing
│   │   ├── movielens_loader.py    # Dataset loading
│   │   └── graph_builder.py       # Graph construction
│   │
│   └── utils/                # Utility functions
│       ├── visualization.py  # Plotting and visualization
│       ├── config.py         # Configuration management
│       └── metrics.py        # Evaluation metrics
│
├── Data/                     # Dataset directory
├── output/                   # Training outputs
│   ├── best_encoder/        # Best model checkpoints
│   ├── logs/                # Training logs
│   └── visualizations/       # Generated plots and figures
│
└── tests/                   # Unit tests
```

## 📊 Evaluation

### Core Metrics

**Ranking Metrics**
- Precision@K, Recall@K, F1@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)
- Hit Rate@K

**Cold-Start Performance**
- Cold-start user/item coverage
- Performance on sparse interactions
- Comparison with warm-start scenarios

**Beyond-Accuracy Metrics**
- **Coverage**: Catalog coverage
- **Diversity**: Intra-list diversity
- **Novelty**: Recommendation novelty
- **Serendipity**: Unexpected but relevant recommendations

## 📈 Results

Our experiments demonstrate strong performance on cold-start scenarios:

| Model | NDCG@10 | Recall@10 | Coverage |
|-------|---------|-----------|----------|
| GCN + VAE | 0.187 | 0.243 | 0.782 |
| GAT + Bilinear | 0.201 | 0.267 | 0.812 |
| GraphSAGE + MLP | 0.176 | 0.231 | 0.795 |

*Results on MovieLens-25M test set with 20% cold-start users*

## Research Applications

- **Recommendation Systems**: E-commerce, streaming platforms, social networks
- **Graph Generation**: Learning and sampling from graph distributions
- **Transfer Learning**: Adapting models to new domains
- **Few-Shot Learning**: Learning from limited interaction data

## Key Features

- **Modular Design**: Easy to swap encoders and decoders
- **Scalable**: Efficient implementation for large graphs
- **Configurable**: Extensive configuration system
- **Comprehensive**: Full pipeline from data loading to evaluation
- **Research-Ready**: Built for experimentation and extension

## Contributing

We welcome contributions! Please see our contributing guidelines for details on:
- Code style and standards
- Testing requirements
- Documentation expectations
- Pull request process

## 🛠️ Technologies Used

### Core Technologies
- **Python**: Primary programming language
- **PyTorch**: Deep learning framework
- **PyTorch Geometric**: Graph neural networks
- **NumPy & Pandas**: Data manipulation
- **Matplotlib & Seaborn**: Visualizations
- **Scikit-learn**: Evaluation metrics and utilities

### Development Tools
- **Git & GitHub**: Version control
- **Black & Flake8**: Code formatting and linting
- **Weights & Biases**: Experiment tracking
- **Jupyter**: Interactive development

## 📖 Documentation

For detailed documentation, please refer to:
- [API Documentation](docs/API.md)
- [Tutorial Notebooks](notebooks/)
- [Model Architecture](docs/ARCHITECTURE.md)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Shreeraj Mummidivarapu & Rahul Tarachand

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 📚 Citation

If you use GenRecGraph in your research, please cite:

```bibtex
@misc{genrecgraph2024,
  title={GenRecGraph: Generative Graph Models for Cold-Start Recommendation},
  author={ShreeRaj Mummidivarapu and Rahul Tarachand},
  year={2024},
  url={https://github.com/Shree2604/GenRecGraph}
}
```

## 🙏 Acknowledgments

- [MovieLens](https://grouplens.org/datasets/movielens/) dataset provided by GroupLens Research
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) for graph neural network implementations
- The open-source community for their valuable contributions

## 📞 Contact

For questions, issues, or collaboration opportunities:

- 📧 Shreeraj Mummidivarapu: [shreeraj.m22@iiits.in](mailto:shreeraj.m22@iiits.in)
- 📧 Rahul Tarachand: [rahul.r22@iiits.in](mailto:rahul.r22@iiits.in)
- 📝 [Create an issue](https://github.com/Shree2604/GenRecGraph/issues) on GitHub

---

<div align="center">
  <p>Built with ❤️ for Advancing Recommendation Systems</p>
  <p>© 2024 Shreeraj Mummidivarapu & Rahul Tarachand</p>
  <a href="https://github.com/Shree2604/GenRecGraph">
    <img src="https://img.shields.io/github/stars/Shree2604/GenRecGraph?style=social" alt="GitHub stars">
  </a>
  <a href="https://github.com/Shree2604/GenRecGraph/fork">
    <img src="https://img.shields.io/github/forks/Shree2604/GenRecGraph?style=social" alt="GitHub forks">
  </a>
</div>
