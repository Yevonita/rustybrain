# rustybrain

<p align="center">
  <img src="doc/logo.png" alt="rustybrain logo" width="200"/>
</p>

**rustybrain** is a simple neural network library written in Rust.  
It provides a minimal, educational implementation of a feedforward neural network with backpropagation, designed for learning and experimentation.

> **Inspired by the book "[Make Your Own Neural Network](https://makeyourownneuralnetwork.blogspot.com/)" by Tariq Rashid.**  
> This project follows many of the concepts and structure presented in the book. Full credit to Tariq Rashid for his clear and accessible introduction to neural networks.

## Features

- Feedforward neural network with one hidden layer
- Customizable network size and learning rate
- Sigmoid activation function
- Matrix-based math utilities
- Forward pass (`query`) and training (`train`) methods
- Unit tests for core math operations

## Example Usage
More real usage examples will come!

```rust
use rustybrain::NeuralNetwork;

fn main() {
    // Create a network with 3 input nodes, 1 hidden layer with 3 nodes, and 1 output node
    let mut nn = NeuralNetwork::new(3, 1, 3, 0.4);

    // Query the network
    let result = nn.query(vec![1.0, 2.0, 3.0]);
    println!("Query result: {:?}", result);

    // Train the network
    nn.train(vec![1.0, 2.0, 3.0], vec![0.0, 1.0, 0.0]);
    let result_after_train = nn.query(vec![1.0, 2.0, 3.0]);
    println!("Query after train: {:?}", result_after_train);
}
```

## MNIST Digit Classification Example

The library includes a complete example demonstrating neural network training on the MNIST dataset for digit classification. This example shows how to use rustybrain for a real-world machine learning task.

### Running the Example

```bash
cargo run --example mnist_csv
```

### What the Example Does

The MNIST example demonstrates:

- **Data Loading**: Reading CSV files containing MNIST digit data
- **Data Preprocessing**: Normalizing pixel values (0-255 → 0-1) and one-hot encoding labels
- **Network Architecture**: 784 input nodes (28×28 pixels), 200 hidden nodes, 10 output nodes (digits 0-9)
- **Training**: 50 epochs of backpropagation with a learning rate of 0.3
- **Evaluation**: Testing on a separate dataset and calculating accuracy

### Expected Output

```
Training on 100 samples
Epoch 10
Epoch 20
Epoch 30
Epoch 40
Epoch 50
Testing on 10 samples
Predicted: 7, Actual: 7
Predicted: 3, Actual: 2
Predicted: 1, Actual: 1
Predicted: 0, Actual: 0
Predicted: 4, Actual: 4
...
Accuracy: 50.00%
```

### Data Files

The example uses two CSV files in the `examples/data/` directory:
- `mnist_train_100.csv`: 100 training samples
- `mnist_test_10.csv`: 10 test samples

Each CSV file contains:
- First column: digit label (0-9)
- Remaining 784 columns: pixel values (0-255)

### Key Features Demonstrated

- **Matrix Operations**: Efficient handling of large input vectors (784 dimensions)
- **Backpropagation**: Proper weight updates using gradient descent
- **Classification**: Multi-class classification with one-hot encoded targets
- **Performance**: Achieving reasonable accuracy on a real dataset

This example serves as both a practical demonstration and a template for implementing other classification tasks with rustybrain.

## Roadmap

Planned future improvements:

- [ ] **Error handling:** Add robust checks and clear error messages for all matrix operations.
- [ ] **Flexible activation functions:** Allow users to select or provide custom activation functions.
- [ ] **Persistence:** Implement methods to save and load network weights to/from disk.
- [ ] **Batch training:** Add support for training on batches of data for improved performance.
- [ ] **Evaluation utilities:** Provide functions for accuracy, loss calculation, and other metrics.
- [ ] **Documentation:** Expand documentation and add more usage examples.
- [ ] **Multiple hidden layers:** Support for deeper networks with more than one hidden layer.

## License

[MIT License](LICENSE.md)

---

**rustybrain** is a learning project. Contributions and suggestions are welcomed!