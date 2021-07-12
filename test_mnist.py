import datasets.mnist.loader as mnist
from sklearn.preprocessing import OneHotEncoder
from src.nn import DeepNeuralNetwork


X_train, y_train, X_test, y_test = mnist.get_data()

# Reshape the training and test examples
X_train = X_train.reshape(X_train.shape[0], -1).T / 255.0
X_test = X_test.reshape(X_test.shape[0], -1).T / 255.0

enc = OneHotEncoder(sparse=False, categories="auto")
y_train = enc.fit_transform(y_train.reshape(len(y_train), -1)).T

y_test = enc.transform(y_test.reshape(len(y_test), -1)).T

# Define Neural Network architecture
layers = [
    (X_train.shape[0], "sigmoid"),
    (50, "sigmoid"),
    (10, "softmax"),
]
model = DeepNeuralNetwork(layers)

# Train model
model.train(X_train, y_train, print_cost=True, num_iterations=1000, learning_rate=0.1)

accuracy, p, probas = model.predict(X_test, y_test)
print("Accuracy: " + str(accuracy))
