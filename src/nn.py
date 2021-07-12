import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt


class NeuralNetMixin:
    def sigmoid(self, Z):
        """
        Implements the sigmoid activation in numpy

        Arguments:
        Z -- numpy array of any shape

        Returns:
        A -- output of sigmoid(z), same shape as Z
        cache -- returns Z as well, useful during backpropagation
        """
        A = 1 / (1 + np.exp(-Z))
        cache = Z
        return A, cache

    def relu(self, Z):
        """
        Implement the RELU function.

        Arguments:
        Z -- Output of the linear layer, of any shape

        Returns:
        A -- Post-activation parameter, of the same shape as Z
        cache -- a python dictionary containing "A" ; stored for computing
        the backward pass efficiently
        """

        A = np.maximum(0, Z)

        assert A.shape == Z.shape

        cache = Z
        return A, cache

    def softmax(self, Z):
        """
        Implement the SoftMax function.

        Arguments:
        Z -- Output of the linear layer, of any shape

        Returns:
        A -- Post-activation parameter, of the same shape as Z
        cache -- a python dictionary containing "A" ; stored for computing
        the backward pass efficiently
        """
        exps = np.exp(Z - Z.max())
        A = exps / np.sum(exps, axis=0)
        cache = Z

        return A, cache

    def relu_backward(self, dA, cache):
        """
        Implement the backward propagation for a single RELU unit.

        Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

        Returns:
        dZ -- Gradient of the cost with respect to Z
        """

        Z = cache
        dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

        # When z <= 0, you should set dz to 0 as well.
        dZ[Z <= 0] = 0

        assert dZ.shape == Z.shape

        return dZ

    def sigmoid_backward(self, dA, cache):
        """
        Implement the backward propagation for a single SIGMOID unit.

        Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

        Returns:
        dZ -- Gradient of the cost with respect to Z
        """

        Z = cache

        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)

        assert dZ.shape == Z.shape

        return dZ

    def softmax_backward(self, dA, cache):
        """
        Implement the backward propagation for a single SOFTMAX unit.

        Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

        Returns:
        dZ -- Gradient of the cost with respect to Z
        """

        Z = cache
        exps = np.exp(Z - Z.max())
        s = exps / np.sum(exps, axis=0)
        dZ = dA * s * (1 - s)

        assert dZ.shape == Z.shape

        return dZ

    def linear_forward(self, A, W, b):
        """
        Implement the linear part of a layer's forward propagation.

        Arguments:
        A -- activations from previous layer (or input data):
         (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape
         (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        Z -- the input of the activation function
        cache -- a python dictionary containing "A", "W" and "b" ;
         stored for computing the backward pass efficiently
        """

        Z = W.dot(A) + b

        assert Z.shape == (W.shape[0], A.shape[1])
        cache = (A, W, b)

        return Z, cache

    def compute_cost(self, AL, Y):
        """
        Implement the cost function defined by equation (7).

        Arguments:
        AL -- probability vector corresponding to your label predictions,
         shape (K, number of examples)
        Y -- true "label" vector,
         shape (K, number of examples)

        Returns:
        cost -- cross-entropy cost
        """
        K, m = Y.shape[0], Y.shape[1]
        # Compute loss from aL and y.
        if K == 1:
            cost = (1.0 / m) * (
                -np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T)
            )
        else:
            temp = np.sum(np.log(AL) * Y, axis=0)
            cost = (1.0 / m) * (-np.dot(temp, temp.T))

        cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect
        assert cost.shape == ()

        return cost

    def linear_backward(self, dZ, cache):
        """
        Implement the linear portion of backward propagation for a single layer
         (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output
         (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in
         the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous
         layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = 1.0 / m * np.dot(dZ, A_prev.T)
        db = 1.0 / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        assert dA_prev.shape == A_prev.shape
        assert dW.shape == W.shape
        assert db.shape == b.shape

        return dA_prev, dW, db


class DeepNeuralNetwork(NeuralNetMixin):
    def __init__(self, layers: List[Tuple[int, str]]):
        self.layer_dims, self.layer_actifun = map(list, zip(*layers))

        # Initialize Parameters
        self.params = self.initialize_parameters()

    def initialize_parameters(self):
        """
        Initializes the network parameters

        Returns:
        parameters -- python dictionary containing the parameters
         "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """

        np.random.seed(1)
        parameters = {}
        L = len(self.layer_dims)  # number of layers in the network

        for layer in range(1, L):
            parameters["W" + str(layer)] = np.random.randn(
                self.layer_dims[layer], self.layer_dims[layer - 1]
            ) / np.sqrt(self.layer_dims[layer - 1])
            parameters["b" + str(layer)] = np.zeros((self.layer_dims[layer], 1))

            assert parameters["W" + str(layer)].shape == (
                self.layer_dims[layer],
                self.layer_dims[layer - 1],
            )
            assert parameters["b" + str(layer)].shape == (self.layer_dims[layer], 1)

        return parameters

    def update_parameters(self, grads, learning_rate):
        """
        Update parameters using gradient descent

        Arguments:
        parameters -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients, output of L_model_backward

        Returns:
        parameters -- python dictionary containing your updated parameters
                    parameters["W" + str(l)] = ...
                    parameters["b" + str(l)] = ...
        """

        L = len(self.params) // 2  # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
        for layer in range(L):
            self.params["W" + str(layer + 1)] = (
                self.params["W" + str(layer + 1)]
                - learning_rate * grads["dW" + str(layer + 1)]
            )
            self.params["b" + str(layer + 1)] = (
                self.params["b" + str(layer + 1)]
                - learning_rate * grads["db" + str(layer + 1)]
            )

        return self

    def linear_activation_forward(self, A_prev, W, b, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- activations from previous layer (or input data):
         (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape
         (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string

        Returns:
        A -- the output of the activation function
        cache -- a python dictionary containing "linear_cache" and "activation_cache";
                stored for computing the backward pass efficiently
        """

        Z, linear_cache = self.linear_forward(A_prev, W, b)
        if activation == "sigmoid":
            A, activation_cache = self.sigmoid(Z)

        elif activation == "relu":
            A, activation_cache = self.relu(Z)

        elif activation == "softmax":
            A, activation_cache = self.softmax(Z)

        else:
            raise ValueError(f"Activation function {activation} is not supported.")

        assert A.shape == (W.shape[0], A_prev.shape[1])
        cache = (linear_cache, activation_cache)

        return A, cache

    def forward_pass(self, X):
        """
        Implement forward propagation

        Arguments:
        X -- data, numpy array of shape (input size, number of examples)

        Returns:
        A -- last post-activation value
        caches -- list of caches
        """

        caches = []
        A = X
        L = len(self.params) // 2  # number of layers in the neural network

        # Forward pass through all layers. Add "cache" to the "caches" list.
        for layer in range(1, L + 1):
            A_prev = A
            A, cache = self.linear_activation_forward(
                A_prev,
                self.params["W" + str(layer)],
                self.params["b" + str(layer)],
                activation=self.layer_actifun[layer],
            )
            caches.append(cache)

        assert A.shape == (self.layer_dims[-1], X.shape[1])

        return A, caches

    def linear_activation_backward(self, dA, cache, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.

        Arguments:
        dA -- post-activation gradient for current layer l
        cache -- tuple of values (linear_cache, activation_cache) we store for
         computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation
         (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        linear_cache, activation_cache = cache

        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)

        elif activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, activation_cache)

        elif activation == "softmax":
            dZ = self.softmax_backward(dA, activation_cache)

        else:
            raise ValueError(f"Activation function {activation} is not supported.")

        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    def backward_pass(self, AL, Y, caches):
        """
        Implement  backward propagation

        Arguments:
        AL -- probability vector, output of the forward propagation
        Y -- true "label" vector
        caches -- list of caches
        Returns:
        grads -- A dictionary with the gradients
                grads["dA" + str(l)] = ...
                grads["dW" + str(l)] = ...
                grads["db" + str(l)] = ...
        """
        grads = {}
        L = len(caches)  # the number of layers
        K = AL.shape[0]
        Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

        # Initializing the backpropagation
        if K == 1:
            dA_prev_temp = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        else:
            dA_prev_temp = -np.divide(Y, AL)

        # Backward pass through all layers.
        for layer in reversed(range(L)):
            current_cache = caches[layer]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(
                dA_prev_temp,
                current_cache,
                activation=self.layer_actifun[layer + 1],
            )
            grads["dA" + str(layer)] = dA_prev_temp
            grads["dW" + str(layer + 1)] = dW_temp
            grads["db" + str(layer + 1)] = db_temp

        return grads

    def train(self, X, Y, learning_rate=0.0075, num_iterations=3000, print_cost=False):
        """
        Train the network

        Arguments:
        X -- data, numpy array of shape (number of features, number of examples)
        Y -- true "label" vector of shape (number of classes, number of examples)
        layers_dims -- list containing the input size and each layer size, of length
         (number of layers + 1).
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps

        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """

        np.random.seed(1)
        costs = []  # keep track of cost

        # Loop (gradient descent)
        for i in range(0, num_iterations):
            # Forward propagation
            AL, caches = self.forward_pass(X)

            # Compute cost
            cost = self.compute_cost(AL, Y)

            # Backward propagation
            grads = self.backward_pass(AL, Y, caches)

            # Update parameters
            self.update_parameters(grads, learning_rate)

            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
            if print_cost and i % 100 == 0:
                costs.append(cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel("cost")
        plt.xlabel("iterations (per hundreds)")
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        return self
