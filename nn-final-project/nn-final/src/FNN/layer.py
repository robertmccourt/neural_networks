import random

import numpy as np

class Layer:
    """
    A layer in the Feedforward Neural Network (FNN).
    """

    # Randomly initialize weights and biases
    def __init__(self, n_input, n_output, activation='relu'):
        random.seed(2400)
        self.weights = np.random.randn(n_input+1, n_output) * 0.01
        self.activation_function = activation
        self.n_input = n_input

        # for adam
        self.A = np.zeros_like(self.weights)
        self.F = np.zeros_like(self.weights)

    def forward(self, X):    
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        
        self.z = np.dot(X, self.weights)
        self.a = self.activate(self.z)
        self.input_data = X
        
        return self.a

    # Activation functions
    def activate(self, z):
        activations = {
            'relu': lambda z: np.maximum(0, z),
            'sigmoid': lambda z: 1 / (1 + np.exp(-z)),
            'id': lambda z: z,
            'sign': lambda z: np.sign(z),
            'tanh': lambda z: np.tanh(z),
            'hard tanh': lambda z: np.clip(z, -1, 1),
            'logsoftmax': lambda z: z - np.log(np.sum(np.exp(z - np.max(z, axis=1, keepdims=True)), axis=1, keepdims=True) + 1e-8)
        }

        return activations[self.activation_function](z)

    # Derivatives of activation functions
    """
    If an error arises using the 'sign' activation function, it is because the derivative is undefined at z = 0. (Will return NaN)
    """
    def activation_deriv(self, z):
        derivs = {
            'relu': lambda z: np.where(z > 0, 1, 0),
            'sigmoid': lambda z: (sig := 1 / (1 + np.exp(-z))) * (1 - sig),
            'id': lambda _: np.ones_like(z),
            'sign': lambda z: np.zeros_like(z),  # Derivative undefined at z = 0
            'tanh': lambda z: 1 - np.tanh(z) ** 2,
            'hard tanh': lambda z: np.where(np.abs(z) <= 1, 1, 0),
            # logsoftmax derivative here
            'logsoftmax': lambda z: np.exp(z - np.max(z, axis=1, keepdims=True)) / (
                        np.sum(np.exp(z - np.max(z, axis=1, keepdims=True)), axis=1, keepdims=True) + 1e-8)
        }

        return derivs[self.activation_function](z)

    def backward(self, dL_dout):
        dL_dout = np.nan_to_num(dL_dout)
        if self.activation_function != 'logsoftmax':
            activation_deriv = self.activation_deriv(self.z)
            dL_dout *= activation_deriv
        # partial derivative of the loss w.r.t. the weights
        grad_W = np.dot(self.input_data.T, dL_dout)
        # accumulation of partial derivative of the loss for each layer
        dL_din = np.dot(dL_dout, self.weights.T)

        # Remove the bias
        dL_din = dL_din[:, :-1]

        grad_W = np.clip(grad_W, -3, 3)

        return grad_W, dL_din


    def update_A(self, gradients_W, rho=0.999):
        gradients_W = np.clip(gradients_W, -3, 3)
        self.A = rho*self.A + (1 - rho) * (gradients_W ** 2)
            
    def update_F(self, gradients_W, rho_f=0.9):
        gradients_W = np.clip(gradients_W, -3, 3)
        self.F = rho_f * self.F + (1-rho_f) * gradients_W
