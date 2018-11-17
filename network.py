import numpy as np
from types import *


class Network:
    def __init__(
            self,
            number_of_input_neurons,
            hidden_layer_neuron_distribution,
            number_of_output_neurons
    ):
        """
        Sets up the Network by specifying the number of input
        neurons, the number of output neurons, the number of
        hidden layers, and the number of neurons in each hidden
        layer. The last two sets of information are obtained from
        the hidden_layer_neuron_distribution, which is an array
        of integers. Each integer specifies the number of neurons
        in each hidden layer. The order of the integers must match
        the order of the hidden layers with the first being the hidden
        layer directly connected to the input layer, and the last being
        the hidden layer connected to the output layer.
        :param number_of_input_neurons:
        :param hidden_layer_neuron_distribution:
        :param number_of_output_neurons:
        """
        assert type(number_of_input_neurons) is IntType
        assert type(number_of_output_neurons) is IntType
        assert type(hidden_layer_neuron_distribution) is np.ndarray

        self.number_of_input_neurons = number_of_input_neurons
        self.number_of_output_neurons = number_of_output_neurons
        self.number_of_hidden_layers = np.alen(hidden_layer_neuron_distribution)
        self.hidden_layer_neuron_distribution = hidden_layer_neuron_distribution
