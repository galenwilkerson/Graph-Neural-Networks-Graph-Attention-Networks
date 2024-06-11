# Graph Neural Networks and Graph Attention Networks

This repository contains a Jupyter notebook demonstrating the usage of Graph Neural Networks (GNNs) and Graph Attention Networks (GATs) using the PyTorch Geometric library (`pyg`). The notebook covers various functionalities of the `pyg` library, including data handling, transformations, building, training, and evaluating GNN models, and visualizing the network and metrics.

## Repository Contents

- `GNNs_GATs.ipynb`: The main Jupyter notebook demonstrating the functionalities and implementation of GNNs and GATs.

## Installation

To run the notebook, you need to have Python and Jupyter Notebook installed. Additionally, you need to install the required libraries. You can install them using `pip`:

```bash
pip install torch torch-geometric matplotlib networkx
```

## Notebook Overview

### 1. Data Handling

- **Creating a Graph**: Demonstrates how to create a graph data object using the `Data` class in PyTorch Geometric.
- **Data Transformations**: Shows how to normalize node features using the `NormalizeFeatures` transform.

### 2. Datasets

- **Loading Datasets**: Explains how to load popular datasets like Cora using the `Planetoid` class.

### 3. Building Graph Neural Networks

- **Graph Convolutional Network (GCN)**: Provides an example of building a simple GCN using `GCNConv` layers.

### 4. Training GNNs

- **Training Loop**: Demonstrates how to integrate PyTorch's training loop to train the GCN on the Cora dataset.

### 5. Evaluating GNNs

- **Evaluation**: Shows how to evaluate the model's performance on test data.

### 6. Visualizing the Network and Metrics

- **Graph Visualization**: Includes functions to visualize the graph before and after training.
- **Metrics Plotting**: Provides functions to plot training and validation metrics such as loss and accuracy.

## Tutorials

### Understanding Graph Neural Networks (GNNs) and Graph Convolutional Networks (GCNs)

Graph Neural Networks (GNNs) are a class of neural networks designed to operate on graph-structured data. They leverage the connections (edges) between data points (nodes) to capture the relationships and dependencies that traditional neural networks might miss.

#### What is a Graph?

A graph is a collection of nodes (or vertices) connected by edges. Graphs can represent various types of data, such as social networks (where nodes are people and edges are friendships), molecular structures (where nodes are atoms and edges are chemical bonds), and many more.

#### Intuition Behind GCNs

Graph Convolutional Networks (GCNs) extend the idea of convolutional neural networks (CNNs) to graph-structured data. In a CNN, filters (or kernels) slide over the image to capture local patterns. Similarly, in a GCN, filters aggregate information from a node’s neighbors to learn node representations.

1. **Node Feature Aggregation**: Each node updates its feature vector by aggregating the feature vectors of its neighbors.
2. **Weight Sharing**: Similar to CNNs, GCNs share weights across different parts of the graph, making the model more efficient and scalable.
3. **Activation and Pooling**: After aggregation, the updated features are passed through an activation function (like ReLU) and potentially a pooling layer.

The goal is to learn a function that maps each node (and its neighbors) to an embedding space where nodes with similar roles or features are close to each other.

### Understanding Graph Attention Networks (GATs)

Graph Attention Networks (GATs) introduce attention mechanisms to GNNs, allowing the model to weigh the importance of different neighbors when aggregating node features.

#### Intuition Behind GATs

In a standard GCN, each neighbor contributes equally to the node's updated feature vector. However, in many scenarios, some neighbors might be more important than others. GATs address this by using attention mechanisms to assign different weights to different neighbors.

1. **Self-Attention**: Each node computes attention scores with its neighbors. These scores indicate the importance of each neighbor.
2. **Weighted Aggregation**: Neighbors' feature vectors are weighted by their attention scores and aggregated to update the node’s feature vector.
3. **Multi-Head Attention**: To stabilize the learning process, GATs use multiple attention heads, each learning different attention scores independently. The results are then concatenated (or averaged) to form the final representation.

The attention mechanism allows GATs to focus on the most relevant parts of the graph, potentially improving performance on tasks like node classification and link prediction.

## Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/galenwilkerson/Graph-Neural-Networks-Graph-Attention-Networks.git
   cd Graph-Neural-Networks-Graph-Attention-Networks
   ```

2. **Install the required libraries**:
   ```bash
   pip install torch torch-geometric matplotlib networkx
   ```

3. **Run the Jupyter notebook**:
   ```bash
   jupyter notebook GNNs_GATs.ipynb
   ```

4. **Follow the instructions in the notebook** to execute the code cells and explore the functionalities of GNNs and GATs.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project uses the following libraries:
- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)
- [Matplotlib](https://matplotlib.org/)
- [NetworkX](https://networkx.github.io/)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue if you have any suggestions or improvements.
