import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import random

from layer import Layer
from fnn import FNN


random_seed = 24
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)

def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=1000, shuffle=True)


def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    for batch_X, batch_y in test_data:
        batch_X = batch_X.view(batch_X.shape[0], -1).numpy()
        batch_y = batch_y.numpy()

        outputs = net.forward(batch_X)
        predicted = np.argmax(outputs, axis=1)
        n_correct += (predicted == batch_y).sum()
        n_total += batch_y.shape[0]

    return n_correct / n_total


# Create network
input_size = 28 * 28
hidden_size = 64
output_size = 10

layer1 = Layer(input_size, hidden_size, init_type='xavier', activation='relu')
layer2 = Layer(hidden_size, hidden_size, init_type='xavier', activation='relu')
layer3 = Layer(hidden_size, hidden_size, init_type='xavier', activation='relu')
layer4 = Layer(hidden_size, output_size, init_type='xavier', activation='logsoftmax')

net = FNN(layers=[layer1, layer2, layer3, layer4])

# Get data
train_data = get_data_loader(is_train=True)
test_data = get_data_loader(is_train=False)

# Initial accuracy
print("Initial accuracy:", evaluate(test_data, net))

# Training
learning_rate = 0.001
epochs = 100
batch_size = 25

# Prepare the entire training dataset
X_train = []
y_train = []
for batch_X, batch_y in train_data:
    X_train.append(batch_X.view(batch_X.shape[0], -1).numpy())
    y_train.append(batch_y.numpy())

X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)

# Convert labels to one-hot encoding
y_train_one_hot = np.zeros((y_train.size, output_size))
y_train_one_hot[np.arange(y_train.size), y_train] = 1

# Train using SGD
loss_func = 'nll'
use_adam = False
for epoch in range(epochs):
    net.sgd(X_train, y_train_one_hot, batch_size, learning_rate, use_adam, loss_func)
    
    # Evaluate after each epoch
    accuracy = evaluate(test_data, net)
    print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.4f}")

# Visualize some predictions
for n, (x, _) in enumerate(test_data):
    if n > 5:
        break

    x_flat = x[0].view(-1).numpy()
    pred = np.argmax(net.forward(x_flat.reshape(1, -1)))

    plt.figure(n)
    plt.imshow(x[0].view(28, 28), cmap='gray')
    plt.title(f"Prediction: {pred}")

plt.show()
