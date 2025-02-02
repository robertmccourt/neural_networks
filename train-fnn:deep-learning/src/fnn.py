import numpy as np

class FNN:
    """
    A Feed-Forward Neural Network.
    """

    # Initialize the network with a list of layers
    def __init__(self, layers, momentum=0.9, use_nesterov=False):
        self.layers = layers
        self.A = [np.zeros_like(layer.weights) for layer in self.layers]
        self.F = [np.zeros_like(layer.weights) for layer in self.layers]
        self.t=1
        # Add Nesterov-related parameters
        self.momentum = momentum
        self.use_nesterov = use_nesterov
        # Initialize velocities for each layer
        for layer in self.layers:
            layer.velocity = np.zeros_like(layer.weights)
        
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
            dL_dout = 2 * (y_pred - y) / y.shape[0]
        elif loss_func == 'nll':
            dL_dout = y_pred - y
        gradients_W = []
        # Proceeding backward through the layers, add each new calculation to the front
        # to create the gradients array
        #reg_lambda = 1e-2
        for layer in reversed(self.layers):
            grad_W, dL_dout = layer.backward(dL_dout)
            #grad_W += reg_lambda * layer.weights
            grad_W = np.clip(grad_W, -.5, .5)
            gradients_W.insert(0, grad_W)
        return gradients_W
    
    def hessian_diagonal_backward(self, y, y_pred, loss_func='mse'):
        # Initialize dL_dout based on the loss function
        if loss_func == 'mse':
            dL_dout = 2 * (y_pred - y) / y.shape[0]
        elif loss_func == 'nll':
            dL_dout = y_pred - y
        
        # Initialize dL_dout_prev for the first pass
        hessians_diag = []

        # Loop through layers in reverse order to accumulate second-order terms
        for layer in reversed(self.layers):
            # Each layer's hessian function should return both hessian_diag and an updated dL_dout for the next layer
            hessian_diag, dL_dout = layer.hessian_diagonal(dL_dout)
            hessians_diag.insert(0, hessian_diag)  # Insert at the beginning to accumulate in the correct order
            
        return hessians_diag

    def newton_update_diagonal_approx(self, X, y, learning_rate=0.01, loss_func='mse'):
        y_pred = self.forward(X)
        gradients = self.backward(y, y_pred, loss_func)
        diag_hessians = self.hessian_diagonal_backward(y, y_pred, loss_func)

        # Use Hessians (diagonal) and gradients to apply Newton's update rule
        for layer, grad_W, hessian_diag in zip(self.layers, gradients, diag_hessians):
            # Prevent division by zero by adding a small constant to hessian_diag
            # and invert only the diagonal
            hessian_diag_inv = 1.0 / (hessian_diag + 1e-8)
            update = learning_rate * (grad_W * hessian_diag_inv)
            # print(hessian_diag_inv)
            layer.weights -= update

    # Update weights and biases using gradient descent
    def gd(self, gradients_W, learning_rate):
        if self.use_nesterov:
            self.nesterov_momentum_update(gradients_W, learning_rate)
        else:
            for layer, grad_W in zip(self.layers, gradients_W):
                layer.weights -= learning_rate * grad_W

    def sgd(self, X, y, batch_size, learning_rate, use_adam, loss_func='mse'):
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)

        for start_idx in range(0, X.shape[0] - batch_size + 1, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            if self.use_nesterov:
                # Look ahead with current velocity
                for layer in self.layers:
                    layer.weights += self.momentum * layer.velocity
                
                # Forward pass with look-ahead weights
                y_pred = self.forward(X_batch)
                
                # Revert weights
                for layer in self.layers:
                    layer.weights -= self.momentum * layer.velocity
                
                # Backward pass
                gradients = self.backward(y_batch, y_pred, loss_func)
                
                # Update with Nesterov momentum
                self.nesterov_momentum_update(gradients, learning_rate)
            
            # Check flag if we want to use adam optimizer
            elif use_adam:
                # Forward pass
                y_pred = self.forward(X_batch)
                # Backward pass
                gradients = self.backward(y_batch, y_pred, loss_func)
                
                self.update_A(gradients)
                self.update_F(gradients)
                
                # Perform adam update
                self.adam_update(learning_rate)
            
            else:
                # Forward pass
                y_pred = self.forward(X_batch)
                # Backward pass
                gradients = self.backward(y_batch, y_pred, loss_func)
                # Update weights using standard SGD
                for layer, gradient in zip(self.layers, gradients):
                    layer.weights -= learning_rate * gradient


                
    def update_A(self, gradients_W, rho=0.999):
        for i, gradient in enumerate(gradients_W):
            self.A[i] = rho*self.A[i] + (1 - rho) * (gradient ** 2)
            
    def update_F(self, gradients_W, rho_f=0.9):
        for i, gradient in enumerate(gradients_W):
            self.F[i] = rho_f * self.F[i] + (1-rho_f) * gradient
            
    def adam_update(self, learning_rate, rho=0.999, rho_f=0.9, epsilon=1e-8):
        for i, layer in enumerate(self.layers):
            
            A_hat_i = self.A[i] / (1 - (rho ** self.t))
            F_hat_i = self.F[i] / (1 - (rho_f ** self.t))
            
            # Calculate alpha_t for the current time step
            alpha_t = learning_rate * ((np.sqrt(1 - (rho ** self.t))) / (1 - (rho_f ** self.t)))
                    
            # Calculate the adaptive step
            adaptive_step = alpha_t * F_hat_i / (np.sqrt(A_hat_i) + epsilon)
            
            # Update weights
            layer.weights -= adaptive_step


    # New method for Nesterov momentum update
    def nesterov_momentum_update(self, gradients_W, learning_rate):
        for layer, grad_W in zip(self.layers, gradients_W):
            # Update velocity
            layer.velocity = self.momentum * layer.velocity - learning_rate * grad_W
            # Update weights using Nesterov momentum
            layer.weights += layer.velocity

    # Train the network using forward and backward propagation
    def train(self, X, y, learning_rate, epochs, use_adam):
        print(f"Using adam optimizer ..." if use_adam else "")
        print(f"Using Nesterov momentum ..." if self.use_nesterov else "")
        
        for _ in range(epochs):
            y_pred = self.forward(X)
            gradients_W = self.backward(y,y_pred)
            
            
            self.gd(gradients_W, learning_rate)
                
    def train_adam(self, X, y, learning_rate, epochs):
        for _ in range(epochs):
            y_pred = self.forward(X)
            gradients_W = self.backward(y, y_pred)
            self.update_A(gradients_W)
            self.update_F(gradients_W)
            self.adam_update(learning_rate)
            self.t += 1
            
    def train_adam_batch(self, X, y, learning_rate, epochs, batch_size, loss_func):
        for epoch in range(epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)

            for start_idx in range(0, X.shape[0] - batch_size + 1, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                # Forward pass
                y_pred = self.forward(X_batch)

                # Backward pass
                gradients_W = self.backward(y_batch, y_pred, loss_func)

                # Update moments and weights using Adam optimizer
                self.update_A(gradients_W)
                self.update_F(gradients_W)
                self.adam_update(learning_rate)
                self.t += 1

            # Optionally, print loss and other metrics for monitoring
            y_pred_full = self.forward(X)
            #loss = self._calculate_loss(y, y_pred_full, loss_func)
            #print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")
            
    # Train the network using stochastic gradient descent
    def trainsgd(self, X, y, learning_rate, epochs, batch_size, use_adam, loss_func='mse'):
        print(f"Using adam optimizer for SGD ..." if use_adam else "")
        print(f"Using Nesterov momentum for SGD ..." if self.use_nesterov else "")
        
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
            # Clip predictions to prevent log(0)
            y_pred = np.clip(y_pred, 1e-10, 1.0)
            return -np.mean(np.sum(y * np.log(y_pred), axis=1))
        else:
            raise ValueError("Unsupported loss function")

    def hessian_matrix(self, X, y):
        y_pred = self.forward(X)

        # Flatten weights across all layers
        num_total_weights = sum(layer.weights.size for layer in self.layers)
        full_hessian = np.zeros((num_total_weights, num_total_weights))

        # Track the starting index for each layer
        start_idx = 0

        for layer_idx, layer in enumerate(self.layers):
            layer_size = layer.weights.size

            # calculate the Hessian entry for each pair of weights in the layer
            for i in range(layer_size):
                for j in range(layer_size):
                    hessian_entry = layer.hessian_entry(i, j, y_pred, y)
                    full_hessian[start_idx + i, start_idx + j] = hessian_entry

            start_idx += layer_size

        return full_hessian

    def newton_update(self, X, y, learning_rate=0.01, loss_func='mse', reg_lambda=1e-4):
        y_pred = self.forward(X)
        gradients = self.backward(y, y_pred, loss_func)
        gradient_vector = np.concatenate([grad.flatten() for grad in gradients])

        # calculate hessian matrix
        full_hessian = self.hessian_matrix(X, y)
        #Regularize
        full_hessian += reg_lambda * np.eye(full_hessian.shape[0])

        # pseudoinverse of the Hessian
        hessian_pinv = np.linalg.pinv(full_hessian)

        # Newton update
        update_step = -learning_rate * np.dot(hessian_pinv, gradient_vector)

        # update weights
        start_idx = 0
        for layer in self.layers:
            layer_size = layer.weights.size
            layer_update = update_step[start_idx:start_idx + layer_size].reshape(layer.weights.shape)
            layer.weights += layer_update
            start_idx += layer_size
