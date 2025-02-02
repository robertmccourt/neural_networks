import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class weston_watkins:
    def __init__(self, n_classes, n_features, learning_rate=.001, epochs=500):
        # weight matrix
        np.random.seed(25)
        self.W = np.random.uniform(-1, 1, size=(n_classes, n_features))
        # learning rate
        self.lr = learning_rate
        # number of epochs
        self.epochs = epochs

    # get loss for debugging
    def loss_function(self, X_i, y_i):
        correct_class_score = np.dot(self.W[y_i], X_i)
        loss = 0
        # loop over classes
        for r in range(self.W.shape[0]):
            if r != y_i:
                margin = np.dot(self.W[r], X_i) - correct_class_score + 1
                loss += max(0, margin)
        return loss

    def gradient(self, X_i, y_i):
        # get gradient wrt weights
        grad_W = np.zeros_like(self.W)
        correct_class_val = np.dot(self.W[y_i], X_i)

        # update for margin violation, to ensure correct class is only updated once
        margin_violation = False
        # loop over classes
        for r in range(self.W.shape[0]):
            if r != y_i:
                margin = np.dot(self.W[r], X_i) - correct_class_val + 1
                if margin > 0:
                    margin_violation = True
                    # incorrect class gradient
                    grad_W[r] += X_i
        if margin_violation:
            # correct class gradient
            grad_W[y_i] -= X_i
        return grad_W





    def fit(self, X, y):
        for epoch in range(self.epochs):
            total_loss = 0
            # loop over samples
            for i in range(len(y)):
                X_i = X[i]
                y_i = y[i]

                # get loss for debugging
                loss = self.loss_function(X_i, y_i)
                # get gradient
                grad_W = self.gradient(X_i, y_i)

                # update weights
                self.W -= self.lr * grad_W

                total_loss += loss

            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(y)}")

    def predict(self, X):
        scores = np.dot(X, self.W.T)  # Compute scores for each class
        # use class with highest score for prediction
        return np.argmax(scores, axis=1)

wine = load_wine()
X, y = wine.data, wine.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=25)

num_classes = len(set(y_train))
num_features = X_train.shape[1]

model = weston_watkins(n_classes=num_classes, n_features=num_features, learning_rate=.001, epochs=500)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

wine_acc = accuracy_score(y_test, y_pred)
print(f"Wine dataset accuracy: {wine_acc:.6f}")

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[f"Class {i}" for i in range(num_classes)], yticklabels=[f"Class {i}" for i in range(num_classes)])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()