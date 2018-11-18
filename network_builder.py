from types import *
from hidden_layer_neuron import HiddenLayerNeuron
from activation_function import sigmoid
from network import Network
import numpy as np


class NetworkBuilder:
    def __init__(self):
        self._number_of_input_neurons = None
        self._hidden_layer_distribution = {}
        self._hidden_layer_weights = {}
        self._hidden_layer_biases = {}
        self._number_of_output_neurons = None
        self._output_neuron_weights = {}
        self._output_neuron_biases = {}

    def add_input_layer(self, number_of_neurons):
        # type: (IntType) -> NetworkBuilder
        assert type(number_of_neurons) is IntType
        self._number_of_input_neurons = number_of_neurons

        return self

    def add_hidden_layer(self, number_of_neurons, hidden_layer_weights, hidden_layer_biases):
        # type: (int, ListType, ListType) -> NetworkBuilder
        assert type(number_of_neurons) is IntType
        assert type(hidden_layer_weights) is ListType
        assert type(hidden_layer_biases) is ListType

        if self._number_of_input_neurons is None:
            raise ValueError(
                "Cannot add a hidden layer before adding an input layer using the add_input_layer method."
            )
        if len(hidden_layer_weights) != number_of_neurons:
            raise ValueError(
                "The number of elements in hidden_layer_weights {0} does not equal the number of neurons {1}".format(
                    len(hidden_layer_weights),
                    number_of_neurons
                )
            )
        if len(hidden_layer_biases) != number_of_neurons:
            raise ValueError(
                "The number of elements in hidden_layer_biases {0} does not equal the number of neurons {1}".format(
                    len(hidden_layer_biases),
                    number_of_neurons
                )
            )

        # TODO: Check that the dimensions of each weights array are correct, i.e. that their first
        # dimension is the same as the dimension of the previous set of weights' second
        # dimension (or the input layer's dimension if it is the first hidden layer).

        self._hidden_layer_distribution.update({len(self._hidden_layer_distribution)-1: number_of_neurons})
        self._hidden_layer_weights.update({len(self._hidden_layer_weights)-1: hidden_layer_weights})
        self._hidden_layer_biases.update({len(self._hidden_layer_biases)-1: hidden_layer_biases})

        return self

    def add_output_layer(self, number_of_neurons, output_layer_weights, output_layer_biases):
        # type: (int, np.ndarray, np.ndarray) -> NetworkBuilder
        assert type(number_of_neurons) is IntType
        assert type(output_layer_weights) is np.ndarray
        assert type(output_layer_biases) is np.ndarray

        # TODO: Check that the dimensions of the weights of the output layer match those of the last
        # hidden layer neurons' weights

        self._number_of_output_neurons = number_of_neurons
        self._output_neuron_weights = output_layer_weights
        self._output_neuron_biases = output_layer_biases

        return self

    def build_network(self):
        # type: () -> Network
        if (
            self._number_of_input_neurons is None or
            self._number_of_output_neurons is None
        ):
            raise ValueError(
                'A network must have both an input layer and an output layer in order to be built.'
            )

        hidden_layer_neurons = [
            [
                HiddenLayerNeuron(
                    self._hidden_layer_weights[key][i],
                    self._hidden_layer_biases[key][i],
                    sigmoid()
                )
                for i in range(0, number_of_neurons)
            ]
            for key, number_of_neurons in self._hidden_layer_distribution.iteritems()
        ]

        output_layer_neurons = [
            HiddenLayerNeuron(
                self._output_neuron_weights[i],
                self._output_neuron_biases[i],
                sigmoid()
            )
            for i in range(0, self._number_of_output_neurons)
        ]

        return Network(hidden_layer_neurons, output_layer_neurons)
