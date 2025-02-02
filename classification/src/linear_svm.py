import numpy as np
import numpy.typing as npt

"""
TODO:
Do we need to use regularization in hinge loss function?
"""

class Linear_SVM:

    def __init__(self, weights: np.ndarray, learning_rate: float, C: float) -> None:
        self.weights = weights 
        self.learning_rate = learning_rate
        self.C = C

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute raw scores for input data X."""
        return X @ self.weights 

    def loss(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        """Hinge loss calculation."""
        return np.maximum(0, 1 - y * yhat)

    def update(self, yhat: np.ndarray, y: np.ndarray, X: np.ndarray) -> None:
        """Update weights using gradient descent."""
        for i in range(len(y)):
            if y[i] * yhat[i] < 1:
                regularization_term = self.weights*(1 - self.learning_rate*self.C)
                self.weights = regularization_term + self.learning_rate * (y[i] * X[i, :])

    def fit(self, X: np.ndarray, y: np.ndarray, max_epochs) -> np.ndarray:
        """Fit the model to the data."""
        loss_per_epoch = []
        for epoch in range(max_epochs):
            #print(f"Epoch: {epoch}")
            yhat = self.forward(X)  
            loss = self.loss(y, yhat)  
            self.update(yhat, y, X) 

            epoch_loss = np.sum(loss) / len(y)
            loss_per_epoch.append(epoch_loss)

            if np.all(loss == 0):
                break

        return np.array(loss_per_epoch)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X."""
        return np.where(self.forward(X) >= 0, 1, -1) 
