# digit-classifier
Handwritten digit classification using neural networks. This repository contains a simple PyTorch implementation of a neural network for recognizing black-and-white images of handwritten digits using the MNIST dataset. The model achieves an accuracy of **98.44%** on the test set. The model runs in the browser via ONNX.js. The UI code was adapted from [Elliot Waite](https://github.com/elliotwaite).

### Model Architecture:
- **Input Layer**: 28x28 grayscale image flattened to 784 input features.
- **Hidden Layer**: Linear layer with 512 neurons followed by a ReLU activation function.
- **Dropout**: Dropout layer with a probability of 0.2 to reduce overfitting.
- **Output Layer**: Final linear layer with 10 neurons (one for each digit), followed by a sigmoid activation function to output probabilities for each digit.

The model outputs 10 probabilities, each corresponding to the likelihood that the input image represents a digit from 0 to 9.