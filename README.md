# GenRecGraph: Generative Graph Models for Cold-Start Recommendation

A comprehensive framework for addressing cold-start problems in recommendation systems using generative graph neural networks. This project implements various GNN architectures combined with generative decoders to predict and generate user-item interactions for new users and items.

## Overview

The cold-start problem is one of the most persistent challenges in recommender systems, occurring when new users or items have no historical interaction data. GenRecGraph addresses this by:

- **Learning graph patterns**: Using GNNs to capture complex relational dependencies in user-item interaction graphs
- **Generating realistic edges**: Employing generative models (GraphVAE, autoregressive decoders) to predict plausible interactions
- **Handling multiple scenarios**: Supporting both cold-start users and cold-start items
- **Flexible architecture**: Modular design allowing different encoder-decoder combinations

## Documentation

For comprehensive technical documentation covering the current implementation, see [`DOCUMENTATION.md`](DOCUMENTATION.md). This includes:
- Detailed usage examples and code snippets
- Technical implementation details
- Current capabilities and limitations
- Development roadmap and next steps

## Architecture

### Current Implementation

**Data Processing Pipeline (Complete)**
- **MovieLens-25M Dataset Loading**: Efficient loading and preprocessing of all dataset files
- **Data Filtering & Encoding**: Quality filtering, ID encoding, and temporal splitting
- **Bipartite Graph Construction**: PyTorch Geometric compatible graphs with rich features

**See [`DOCUMENTATION.md`](DOCUMENTATION.md) for detailed technical information about the current implementation.**

### Planned Components

1. **Graph Encoders**: Learn node embeddings from bipartite user-item graphs
   - Graph Convolutional Networks (GCN)
   - Graph Attention Networks (GAT)
   - GraphSAGE (for inductive learning)
   - Heterogeneous GNNs

2. **Generative Decoders**: Predict edge probabilities and generate new interactions
   - Graph Variational Autoencoder (GraphVAE)
   - Inner Product Decoder
   - Multi-Layer Perceptron (MLP) Decoder
   - Autoregressive Decoder
   - Bilinear Decoder

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

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Shree2604/GenRecGraph.git
cd GenRecGraph

# Install dependencies
pip install -r requirements.txt
```

### Getting Started

1. **Download MovieLens-25M dataset** from [GroupLens](http://grouplens.org/datasets/) and place CSV files in the `Data/` folder
2. **Run the data pipeline**: `python main.py` to load data, create bipartite graphs, and generate comprehensive visualizations
3. **See comprehensive documentation** in [`DOCUMENTATION.md`](DOCUMENTATION.md) for detailed usage

### Current Implementation Status

**Data Pipeline Complete**: Load, preprocess, and create graphs from MovieLens data
**Models in Development**: GNN encoders and generative decoders (see Architecture section below)
**See [`DOCUMENTATION.md`](DOCUMENTATION.md) for complete technical details**

## Project Structure

```
GenRecGraph/
├── DOCUMENTATION.md           # Comprehensive technical documentation
├── main.py                    # Data loading and graph creation script
├── Src/
│   ├── datapipe/             # IMPLEMENTED: Data loading and processing
│   │   ├── movielens_loader.py    # Dataset loading and preprocessing
│   │   └── graph_builder.py       # Bipartite graph construction
│   └── utils/                # IMPLEMENTED: Visualization utilities
│       ├── visualization.py       # Comprehensive data and graph visualizations
│       ├── config.py             # Configuration management system
│       └── __init__.py           # Module exports
├── Data/                     # MovieLens dataset (CSV files)
├── requirements.txt          # Dependencies (cleaned to used packages only)
└── README.md                 # This file
```

## Evaluation Metrics

### Ranking Metrics
- Precision@K, Recall@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- Hit Rate@K
- MAP@K (Mean Average Precision)

### Beyond-Accuracy Metrics
- **Coverage**: Fraction of catalog items recommended
- **Diversity**: Intra-list diversity of recommendations
- **Novelty**: Based on item popularity

### Cold-Start Specific
- Separate metrics for cold-start vs warm users
- Cold-start item coverage
- Performance comparison across scenarios

## Experiments

The framework supports various experimental scenarios:

1. **Cold-Start Users**: New users with no interaction history
2. **Cold-Start Items**: New items with no ratings
3. **Sparse Data**: Users/items with very few interactions
4. **Temporal Cold-Start**: Future time periods with new entities

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

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

- MovieLens dataset provided by GroupLens Research
- PyTorch Geometric for graph neural network implementations
- The broader graph machine learning community

## 📞 Contact

For questions, issues, or collaboration opportunities:
- Create an issue on GitHub
- Email: Shreeraj Mummidivarapu [shreeraj.m22@iiits.in]
- Email: Rahul Tarachand [rahul.r22@iiits.in]

---

**Built with ❤️ for Advancing Recommendation Systems by Shreeraj Mummidivarapu & Rahul Tarachand**
