import numpy as np
import torch
import random
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from layer import Layer
from fnn import FNN
from sklearn.model_selection import train_test_split
import time

# Set random seed for reproducibility
random_seed = 24
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)

def get_data_loader(is_train, subset_size=1000, downsample_size=14):
    # Transform: Resize and convert to tensor
    to_tensor = transforms.Compose([
        transforms.Resize((downsample_size, downsample_size)),  # Resize to smaller image
        transforms.ToTensor()
    ])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)

    # Reduce to a subset if specified
    if subset_size and is_train:
        indices = list(range(len(data_set)))
        subset_indices, _ = train_test_split(indices, train_size=subset_size, stratify=[data_set[i][1] for i in indices], random_state=random_seed)
        data_set = Subset(data_set, subset_indices)

    return DataLoader(data_set, batch_size=subset_size, shuffle=True)

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

# Create network with case4's Layer and FNN
input_size = 14 * 14  # Reduced dimensionality due to downsampling
hidden_size = 10
output_size = 10

# Define layers
layer1 = Layer(input_size, hidden_size, init_type= 'uniform', activation='relu')
layer2 = Layer(hidden_size, hidden_size, init_type= 'uniform', activation='relu')
layer3 = Layer(hidden_size, hidden_size, init_type= 'uniform', activation='relu')
layer4 = Layer(hidden_size, output_size, init_type= 'uniform', activation='logsoftmax')

# Initialize FNN with the modified architecture
net = FNN(layers=[layer1, layer2, layer3])

# Get data loaders
train_data = get_data_loader(is_train=True, subset_size=1000, downsample_size=14)  # Smaller input and subset
test_data = get_data_loader(is_train=False, downsample_size=14)

# Prepare training dataset as a single array
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

# Initial accuracy
initial_accuracy = evaluate(test_data, net)
print("Initial accuracy:", initial_accuracy)

# Training parameters
learning_rate = 0.001
epochs = 100
reg_lambda = 1e-2

# Start the timer
start_time = time.time()

# Train using Newton's update from case4
loss_func = 'nll'
use_adam = True
batch_size = 25
method = "diagonal_approx"
for epoch in range(epochs):

    if method == "diagonal_approx":
        net.newton_update_diagonal_approx(X_train, 
                                          y_train_one_hot,
                                          learning_rate=learning_rate,
                                          loss_func=loss_func)
    elif method == "newton_update":
        net.newton_update(X_train,
                          y_train_one_hot,
                          learning_rate=learning_rate,
                          loss_func=loss_func,
                          reg_lambda=reg_lambda)
    
    # Evaluate accuracy after each epoch
    accuracy = evaluate(test_data, net)
    print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.4f}")

# End the timer
end_time = time.time()
training_duration = end_time - start_time
print(f"Total training time: {training_duration:.2f} seconds")

# Visualize some predictions
for n, (x, _) in enumerate(test_data):
    if n > 5:
        break

    x_flat = x[0].view(-1).numpy()
    pred = np.argmax(net.forward(x_flat.reshape(1, -1)))

    plt.figure(n)
    plt.imshow(x[0].view(14, 14), cmap='gray')
    plt.title(f"Prediction: {pred}")

plt.show()