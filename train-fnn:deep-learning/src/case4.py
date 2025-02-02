import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fnn import FNN
from layer import Layer

# Set random seed for reproducibility
random_seed = 24
np.random.seed(random_seed)

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# One-hot encode the target labels
num_classes = len(np.unique(y))
y_onehot = np.zeros((y.shape[0], num_classes))
for i, label in enumerate(y):
    y_onehot[i, label] = 1

# Verify the one-hot encoding
print("Original labels:", y[:5])
print("One-hot encoded labels:\n", y_onehot[:5])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.1, random_state=random_seed)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Network configuration
hidden_layer = 10
layers = [
    Layer(n_input=4, n_output=hidden_layer, init_type='xavier', activation='tanh'),
    Layer(n_input=hidden_layer, n_output=hidden_layer, init_type='xavier', activation='tanh'),
    Layer(n_input=hidden_layer, n_output=3, init_type='xavier', activation='logsoftmax')
]
nn = FNN(layers)

# Training parameters
epochs = 25
learning_rate = 0.001
reg_lambda = 1e-4
method = "diagonal_approx"

# Start the timer
start_time = time.time()

# Train using Newton's method
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")

    if method == "diagonal_approx":
        nn.newton_update_diagonal_approx(X_train, 
                                          y_train,
                                          learning_rate=learning_rate,
                                          loss_func="nll")
    elif method == "newton_update":
        nn.newton_update(X_train,
                          y_train,
                          learning_rate=learning_rate,
                          loss_func="nll",
                          reg_lambda=reg_lambda)

    # Forward pass to get predictions on training set
    y_train_pred = nn.forward(X_train)
    train_loss = nn._calculate_loss(y_train, y_train_pred, loss_func='nll')
    train_accuracy = accuracy_score(y_train.argmax(axis=1), y_train_pred.argmax(axis=1))

    # Print training statistics
    print(f"Epoch {epoch + 1}/{epochs}")
    print(
        f"y_train_pred mean: {np.mean(y_train_pred)}, std: {np.std(y_train_pred)}, min: {np.min(y_train_pred)}, max: {np.max(y_train_pred)}")
    print(f"train_loss: {train_loss}, train_accuracy: {train_accuracy}")

    # Gradient inspection
    gradients_W = nn.backward(y_train, y_train_pred, loss_func='nll')
    for i, grad_W in enumerate(gradients_W):
        print(
            f"Layer {i + 1} Gradient Mean: {np.mean(grad_W)}, Std: {np.std(grad_W)}, Range: {np.min(grad_W)} to {np.max(grad_W)}")

# End the timer
end_time = time.time()
training_duration = end_time - start_time
print(f"Total training time: {training_duration:.2f} seconds")

# Test the network
y_test_pred = nn.forward(X_test)
test_loss = nn._calculate_loss(y_test, y_test_pred, loss_func='nll')
test_accuracy = accuracy_score(y_test.argmax(axis=1), y_test_pred.argmax(axis=1))

print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy * 100:.2f}%")