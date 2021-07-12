from src.utils import load_data
from src.nn import DeepNeuralNetwork
import numpy as np

X_train, y_train, X_test, y_test, classes = load_data()
# Reshape the training and test examples
X_train = X_train.reshape(X_train.shape[0], -1).T / 255.0
X_test = X_test.reshape(X_test.shape[0], -1).T / 255.0

# Define Neural Network architecture
# y_train = np.eye(2)[y_train.reshape(-1)].T
layers = [
    (12288, "relu"),
    (20, "relu"),
    (7, "relu"),
    (5, "relu"),
    (1, "sigmoid"),
    # (2, "softmax"),
]
model = DeepNeuralNetwork(layers)

# Train model
model.train(X_train, y_train, print_cost=True)
