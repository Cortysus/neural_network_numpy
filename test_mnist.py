from src.nn import ANN
from src.utils import load_data_mnist


X_train, y_train, X_test, y_test = load_data_mnist()

# Define Neural Network architecture
layers = [
    (X_train.shape[0], "sigmoid"),
    (50, "sigmoid"),
    (10, "softmax"),
]
model = ANN(layers)

# Train model
model.train(X_train, y_train, print_cost=True, num_iterations=1000, learning_rate=0.1)

accuracy, p, probas = model.predict(X_test, y_test)
print(f"Accuracy: {accuracy}")
