import random

import numpy as np
from scipy.special import logsumexp


class Layer:
    """
    A layer in the Feedforward Neural Network (FNN).
    """

    # Randomly initialize weights and biases
    def __init__(self, n_input, n_output, init_type, activation):
        random.seed(2400)
        if init_type == 'xavier':
            self.weights = self._xavier_init(n_input, n_output)
        else:
            self.weights = np.random.uniform(-1, 1, (n_input+1, n_output))*.5
        self.activation_function = activation
        self.n_input = n_input
        #self.gradient_W = None

    def _xavier_init(self, n_input, n_output):
        """
        Helper method for Xavier initialization with uniform distribution.
        """
        limit = np.sqrt(0.5 / (n_input + n_output))
        return np.random.uniform(-limit, limit, (n_input+1, n_output))

    def forward(self, X):    
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        
        self.z = np.dot(X, self.weights)
        self.a = self.activate(self.z)
        #print(
        #    f"Layer forward pass - Activation {self.activation_function} mean: {np.mean(self.a)}, "
        #    f"std: {np.std(self.a)}, min: {np.min(self.a)}, max: {np.max(self.a)}")
        self.input_data = X
        
        return self.a

    def logsoftmax(self, z):
        return z - np.log(np.sum(np.exp(z - np.max(z, axis=1, keepdims=True)), axis=1, keepdims=True) + 1e-8)

    # Activation functions
    def activate(self, z):
        activations = {
            'relu': lambda z: np.maximum(0, z),
            'sigmoid': lambda z: 1 / (1 + np.exp(-np.clip(z, -100, 100))),
            'id': lambda z: z,
            'sign': lambda z: np.sign(z),
            'tanh': lambda z: np.tanh(z),
            'hard tanh': lambda z: np.clip(z, -1, 1),
            'logsoftmax': lambda z: self.logsoftmax(z),
            'leaky_tanh': lambda z: np.where(z > 0, np.tanh(z), 0.01 * z),
            'softplus': lambda z: np.where(z > 20, z, np.log1p(np.exp(z)))
        }

        return activations[self.activation_function](z)

    # Derivatives of activation functions
    """
    If an error arises using the 'sign' activation function, it is because the derivative is undefined at z = 0. (Will return NaN)
    """
    def activation_deriv(self, z):
        derivs = {
            'relu': lambda z: np.where(z > 0, 1, 0),
            'sigmoid': lambda z: (sig := 1 / (1 + np.exp(-np.clip(z, -100, 100)))) * (1 - sig),
            'id': lambda _: np.ones_like(z),
            'sign': lambda z: np.zeros_like(z),  # Derivative undefined at z = 0
            'tanh': lambda z: 1 - np.tanh(z) ** 2,
            'hard tanh': lambda z: np.where(np.abs(z) <= 1, 1, 0),
            'logsoftmax': lambda z: self._logsoftmax_derivative(z),
            'leaky_tanh': lambda z: np.where(z > 0, 1 - np.tanh(z) ** 2, 0.01),
            'softplus': lambda z: 1 / (1 + np.exp(-z))
        }

        return derivs[self.activation_function](z)

    def _logsoftmax_derivative(self, z):
        # Reshape z to ensure it is at least 2D for consistent axis handling
        if z.ndim == 0:
            z = z.reshape(1, 1)
        elif z.ndim == 1:
            z = z.reshape(1, -1)

        # Compute log-softmax and its derivative
        softmax = np.exp(z - np.max(z, axis=1, keepdims=True)) / np.sum(np.exp(z - np.max(z, axis=1, keepdims=True)),
                                                                        axis=1, keepdims=True)
        return softmax * (1 - softmax) + 1e-5

    def backward(self, dL_dout):
        dL_dout = np.nan_to_num(dL_dout)
        activation_deriv = self.activation_deriv(self.z)
        dL_dout *= activation_deriv
        #print(f"Backward dL_dout: mean {np.mean(dL_dout)}, min {np.min(dL_dout)}, max {np.max(dL_dout)}")
        # partial derivative of the loss w.r.t. the weights
        grad_W = np.dot(self.input_data.T, dL_dout)

        #print(
        #    f"Layer forward pass - Activation {self.activation_function} mean: {np.mean(self.a)}, "
        #    f"std: {np.std(self.a)}, min: {np.min(self.a)}, max: {np.max(self.a)}")
        # accumulation of partial derivative of the loss for each layer
        dL_din = np.dot(dL_dout, self.weights.T)

        # Remove the bias
        dL_din = dL_din[:, :-1]

        grad_W = np.clip(grad_W, -3, 3)
        #self.gradient_W = grad_W
        
        return grad_W, dL_din

    def activation_second_deriv(self, z):
        """
        Compute the second derivative of the activation function.
        """
        second_derivs = {
            'relu': lambda z: np.zeros_like(z),
            'sigmoid': lambda z: (sig := 1 / (1 + np.exp(-np.clip(z, -100, 100)))) * (1 - sig) * (1 - 2 * sig),
            'id': lambda _: np.zeros_like(z),
            'tanh': lambda z: -2 * np.tanh(z) * (1 - np.tanh(z) ** 2),
            'leaky_tanh': lambda z: np.where(z > 0, -2 * np.tanh(z) * (1 - np.tanh(z) ** 2), 0),
            'softplus': lambda z: np.exp(-z) / ((1 + np.exp(-z)) ** 2)
        }
        return second_derivs.get(self.activation_function, lambda z: np.zeros_like(z))(z)
    
    def hessian_diagonal(self, dL_dout):
        dL_dout = np.nan_to_num(dL_dout)
        # Compute the first and second derivatives of the activation function
        activation_deriv = self.activation_deriv(self.z)
        second_activation_deriv = self.activation_second_deriv(self.z)

        # Calculate the diagonal of the Hessian with respect to weights in this layer
        # hessian_diag = np.sum((self.input_data ** 2) * (dL_dout * second_activation_deriv), axis=0)
        hessian_diag = np.sum((self.input_data ** 2).T @ (dL_dout * second_activation_deriv), axis=0)
        
        # Calculate dL_din for second-order backpropagation
        dL_din = np.dot(dL_dout * activation_deriv, self.weights.T)
        second_order_term = np.dot(dL_dout * second_activation_deriv, self.weights.T)
        dL_din += second_order_term

        dL_din = dL_din[:, :-1]

        return hessian_diag, dL_din


    def hessian_entry(self, i, j, y_pred, y):
        # inputs for weights w_i and w_j
        input_i = self.input_data.flatten()[i]
        input_j = self.input_data.flatten()[j]

        # Calculate z
        local_z = input_i * self.weights.flatten()[i] + input_j * self.weights.flatten()[j]

        activation_deriv = self.activation_deriv(local_z)
        activation_second_deriv = self.activation_second_deriv(local_z)

        term1 = activation_second_deriv * input_i * input_j * (y_pred - y)
        term2 = activation_deriv ** 2 * input_i * input_j
        # Hessian for (i,j)
        return np.sum(term1 + term2)
