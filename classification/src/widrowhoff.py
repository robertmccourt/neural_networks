import numpy as np

class WidrowHoff:
    def __init__(self, num_inputs, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = np.random.uniform(-1, 1, num_inputs)
        self.errors = []

    def forward(self, X):
        return np.clip(X @ self.weights, -1e10, 1e10)  # Clipping to prevent overflow

    def loss(self, y_pred, y):
        return np.clip((1 - y_pred * y) ** 2, -1e10, 1e10)  # Clipping to prevent overflow

    def fit(self, X_train, y_train, max_epochs):
        for epoch in range(max_epochs):
            total_error = 0

            for X, y in zip(X_train, y_train):
                y_pred = self.forward(X)

                #calculate the error
                error = self.loss(np.array([y_pred]), np.array([y]))
                total_error += error

                self.weights += self.learning_rate * y * (1 - y * y_pred) * X

                if np.all(error == 0):
                    print(f"Converged at epoch {epoch}")
                    break

            avg_error = total_error / len(y_train)
            self.errors.append(avg_error)
            #print(f"Epoch {epoch} - Total Error: {total_error}")

        return np.array(self.errors)


    def predict(self, X):
        return np.where(self.forward(X) >= 0, 1, -1)
