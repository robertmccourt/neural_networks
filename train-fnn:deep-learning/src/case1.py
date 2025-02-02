import matplotlib.pyplot as plt
import numpy as np

from layer import Layer
from fnn import FNN

# Generate Data
X = np.random.uniform(-3, 3, 1000).reshape(-1, 1)  # Reshape X to be (n_samples, 1)
y = np.sin(X)

# Create Layers
layer1 = Layer(n_input=1, n_output=30, init_type='uniform',  activation='relu')  # Input layer with 1 feature (x)
layer2 = Layer(n_input=30, n_output=30, init_type='uniform', activation='relu') # Hidden layer with 10 neurons
layer3 = Layer(n_input=30, n_output=1, init_type='uniform', activation='id')    # Output layer with 1 neuron (regression output)

# Create FNN
fnn = FNN(layers=[layer1, layer2, layer3])

# Train
learning_rate = 0.01
epochs = 1000

fnn.train(X, y, learning_rate, epochs, use_adam=True)

y_pred = fnn.forward(X)

plt.figure()
plt.scatter(X, y, label='True Data')
plt.scatter(X, y_pred, color='red', label='FNN Predictions')
plt.title("FNN Approximation of sin(x)")
plt.legend()
plt.show()

