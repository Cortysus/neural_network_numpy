from src.utils import load_data_cat
from src.nn import DeepNeuralNetwork

X_train, y_train, X_test, y_test, _ = load_data_cat()

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
