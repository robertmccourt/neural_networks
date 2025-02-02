import numpy as np

class LogisticRegression:
    """
    Logistic Regression classifier.
    """

    def __init__(self, learning_rate=0.01, num_iterations=100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        return self.sigmoid(np.dot(X, self.weights) + self.bias)

    def loss(self, y_pred, y):
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        loss_per_epoch = []

        for _ in range(self.num_iterations):
            y_pred = self.forward(X)

            # Compute the loss and store it for the current epoch
            epoch_loss = self.loss(y_pred, (y + 1) / 2)
            loss_per_epoch.append(epoch_loss)

            # Gradient descent
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - (y + 1) / 2))
            db = (1 / num_samples) * np.sum(y_pred - (y + 1) / 2)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return np.array(loss_per_epoch)

    def predict(self, X):
        y_pred = self.forward(X)
        return np.where(y_pred > 0.5, 1, -1)
