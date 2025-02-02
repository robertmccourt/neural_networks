import numpy as np
from sklearn.datasets import load_breast_cancer, fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from widrowhoff import WidrowHoff


# Load the LFW dataset
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)  # Adjust parameters as needed
X = lfw_people.data
y = lfw_people.target

# For binary classification (adjust according to your needs)
# Example: Use only the first class (e.g., first person) for binary classification
y = np.where(y == 0, 1, -1)  # Change this condition based on your target class

# Normalize the data and add bias
X = preprocessing.normalize(X)
X = np.hstack([X, np.ones((X.shape[0], 1))])  # Add bias to data

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=5)

# Create and train the model
model = WidrowHoff(X_train.shape[1])
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = np.sum(np.where(y_test == y_pred, 1, 0)) / y_test.shape[0]
print(f"Accuracy: {accuracy}")
