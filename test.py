from src.utils import load_data
from src.nn import DeepNeuralNetwork

X_train, y_train, X_test, y_test, classes = load_data()
# Reshape the training and test examples
X_train = X_train.reshape(X_train.shape[0], -1).T / 255.0
X_test = X_test.reshape(X_test.shape[0], -1).T / 255.0

# Define Neural Network architecture

layers = [
    (X_train.shape[0], "relu"),
    (20, "relu"),
    (7, "relu"),
    (5, "relu"),
    (1, "sigmoid"),
]
model = DeepNeuralNetwork(layers)

# Train model
model.train(X_train, y_train, print_cost=True, num_iterations=2400)

accuracy, p, probas = model.predict(X_test, y_test)
print("Accuracy: " + str(accuracy))
