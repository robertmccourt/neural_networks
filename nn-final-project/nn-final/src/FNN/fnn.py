import numpy as np

from FNN.layer import Layer

class FNN:
    """
    A Feed-Forward Neural Network.
    """

    # Initialize the network with a list of layers
    def __init__(self, layers):
        self.layers = layers

    # Perform forward propagation through all layers
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    """
    Calculate gradients for all layers.
    X: Input data
    y: True labels
    y_pred: Predicted output from the forward pass
    loss_func: Loss function ('mse' or 'nll')
    """
    def backward(self, y, y_pred, loss_func='mse'):
        if loss_func == 'mse':
            #NewtonCases
            dL_dout = (y_pred - y) / y.shape[0]
            #Regular
            #dL_dout = 2 * (y_pred - y) / y.shape[0]
        elif loss_func == 'nll':
            dL_dout = y_pred - y
        gradients_W = []
        # Proceeding backward through the layers, add each new calculation to the front
        # to create the gradients array
        for layer in reversed(self.layers):
            grad_W, dL_dout = layer.backward(dL_dout)
            gradients_W.insert(0, grad_W)
        return gradients_W

    # Update weights and biases using gradient descent
    def gd(self, gradients_W, learning_rate):
        for layer, grad_W in zip(self.layers, gradients_W):
            layer.weights -= learning_rate * grad_W


    def sgd(self, X, y, batch_size, learning_rate, loss_func='mse'):
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)

        for start_idx in range(0, X.shape[0] - batch_size + 1, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            # Forward pass
            y_pred = self.forward(X_batch)

            # Backward pass
            gradients = self.backward(y_batch, y_pred, loss_func)

            # Update weights
            for layer, gradient in zip(self.layers, gradients):
                layer.weights -= learning_rate * gradient

    # Train the network using forward and backward propagation
    def train(self, X, y, learning_rate, epochs):
        for _ in range(epochs):
            y_pred = self.forward(X)
            gradients_W = self.backward(y,y_pred)
            self.gd(gradients_W, learning_rate)
# Train the network using stochastic gradient descent
    def trainsgd(self, X, y, learning_rate, epochs, batch_size, loss_func='mse'):
        for epoch in range(epochs):
            self.sgd(X, y, batch_size, learning_rate, loss_func)

            # Calculate and print loss for monitoring
            y_pred = self.forward(X)
            loss = self._calculate_loss(y, y_pred, loss_func)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

    def _calculate_loss(self, y, y_pred, loss_func):
        if loss_func == 'mse':
            return np.mean((y_pred - y) ** 2)
        elif loss_func == 'nll':
            return -np.mean(y * np.log(y_pred + 1e-8))
        else:
            raise ValueError("Unsupported loss function")
