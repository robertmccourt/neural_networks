import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset = "cifar"

with open(f"{dataset}_adam_runs.pkl", "rb") as f:
    adam_runs = pickle.load(f)

with open(f"{dataset}_gd_runs.pkl", "rb") as f:
    gd_runs = pickle.load(f)

# Plot the training runs (ADAM)
plt.figure()
for i in range(3):
    accuracies = adam_runs[i][0] 
    epochs = list(range(len(accuracies)))
    plt.plot(epochs, accuracies, label=f"Run {i+1}")

# Add labels and title
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy")
plt.title(f"{dataset} Training Accuracy vs Epoch for ADAM")
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

# Plot the training runs (GD)
plt.figure()
for i in range(3):
    accuracies = gd_runs[i][0] 
    epochs = list(range(len(accuracies)))
    plt.plot(epochs, accuracies, label=f"Run {i+1}")

# Add labels and title
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy")
plt.title(f"{dataset} Training Accuracy vs Epoch for GD")
plt.legend()

# Show the plot
plt.grid(True)
plt.show()


## Bar Plots
# Bar width
bar_width = 0.4

adam_test_accuracies = [run[1] for run in adam_runs.values()]
gd_test_accuracies = [run[1] for run in gd_runs.values()]
runs = [1,2,3]
# Plotting
runs = np.arange(3)
plt.bar(runs - bar_width/2, adam_test_accuracies, width=bar_width, label='ADAM', color='blue')
plt.bar(runs + bar_width/2, gd_test_accuracies, width=bar_width, label='Gradient Descent', color='orange')

# Adding labels and title
plt.xlabel('Runs')
plt.ylabel('Testing Accuracy')
plt.title(f'{dataset} Testing Accuracy Comparison: ADAM vs Gradient Descent')
plt.xticks(runs, [f'Run {i+1}' for i in runs])  # Label the runs
plt.legend()

# Display the plot
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()