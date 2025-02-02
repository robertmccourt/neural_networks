import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import torch

from layer import Layer
from fnn import FNN

# Generate Data - this code comes from the vanderpol.py file
# from lecture
def ode_model(x, t):
    return [x[1], -x[0] + (1 - x[0]**2)*x[1]]

def Phi(x):
    t = np.linspace(0, 0.05, 101)
    sol = odeint(ode_model, x, t)
    return sol[-1]

# compute the samples
# X is a set of samples in a 2D plane
# Y consists of the corresponding outputs of the samples in X

N = 101 # number of samples in each dimension
samples_x1 = torch.linspace(-3, 3, N)
samples_x2 = torch.linspace(-3, 3, N)

X = torch.empty((0,2))

for x1 in samples_x1:
    for x2 in samples_x2:
        sample_x = torch.Tensor([[x1,x2]])
        X = torch.cat((X, sample_x))

Y = torch.empty((0,2))
for x in X:
    y = Phi(x)
    sample_y = torch.Tensor([[y[0],y[1]]])
    Y = torch.cat((Y, sample_y))

# Plot Data
X_np = X.numpy()
Y_np = Y.numpy()

# Pre-Process Data
# X = torch.cat((X, torch.ones(X.shape[0], 1)), dim=1)
print(X.shape, Y.shape)

# Create Layers
layer1 = Layer(n_input=2, n_output=64, activation='relu')
layer2 = Layer(n_input=64, n_output=64, activation='relu')
layer3 = Layer(n_input=64, n_output=64, activation='relu')
layer4 = Layer(n_input=64, n_output=2, activation='id')

# Create FNN
fnn = FNN(layers=[layer1, layer2, layer3, layer4])

# Train
learning_rate = 0.01
epochs = 1000
#fnn.train(X_np, Y_np, learning_rate, epochs)
fnn.trainsgd(X_np, Y_np, learning_rate, epochs, 25, use_adam=True, loss_func="mse")

x0 = np.array([1.25, 2.35])
#Predict
Y_pred = fnn.forward(X_np)

plt.figure(figsize=(8, 8))
for i in range(150):
    y = Phi(x0)
    plt.plot(y[0], y[1], 'b.')
    x0 = y

x0 = np.array([1.25, 2.35])
for i in range(150):
    y_pred = fnn.forward(x0.reshape(1, -1))
    plt.plot(y_pred[0, 0], y_pred[0, 1], 'r.')
    x0 = y_pred.flatten()

plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Van der Pol Oscillator')
plt.grid(True)
plt.show()