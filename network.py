import numpy as np
from types import *


class Network:

    def __init__(
            self,
            hidden_layer_neurons,
            output_layer_neurons
    ):
        # type: (ListType, ListType) -> None
        """
        Sets up the Network by taking in a nested list of hidden
        layer neurons and a list of output neurons.
        :param hidden_layer_neurons:
        :param output_layer_neurons:
        """
        assert type(hidden_layer_neurons) is ListType
        assert type(output_layer_neurons) is ListType

        # Check that the dimensions of weights of the hidden and output
        # layers match up in the correct way.

        self.hidden_layer_neurons = hidden_layer_neurons
        self.output_layer_neurons = output_layer_neurons

    def output(self, input_data):
        # type: (np.ndarray) -> float

        # TODO: Currently this is just looping over the first hidden layer, feeding each neuron
        # the input data, getting those outputs, and feeding those to the output layer neurons.
        # We need to feed the output of each hidden layer to the next hidden layer, then feed
        # the output of the final hidden layer into the output layer

        hidden_layer_outputs = [
            [
                neuron.output(input_data)
                for neuron in hidden_layer
            ]
            for hidden_layer in self.hidden_layer_neurons
        ]

        output_layer_outputs = [
            neuron.output(hidden_layer_outputs[0])
            for neuron in self.output_layer_neurons
        ]

        return output_layer_outputs[0]
