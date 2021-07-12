import numpy as np
from typing import List, Tuple


class DeepNeuralNetwork:
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
