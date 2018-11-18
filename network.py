import numpy as np
from types import *

from hidden_layer_neuron import HiddenLayerNeuron


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

        for neuron in output_layer_neurons:
            last_hidden_layer_neurons = hidden_layer_neurons[-1]
            if len(last_hidden_layer_neurons) != neuron.get_first_weights_dimension():
                print(len(last_hidden_layer_neurons), neuron.get_first_weights_dimension())
                raise ValueError(
                    "The final hidden layer has {0} neurons but the output layer neurons require {1} inputs. These "
                    "values should be equal.".format(
                        len(last_hidden_layer_neurons),
                        neuron.get_first_weights_dimension()
                    )
                )

        self.hidden_layer_neurons = hidden_layer_neurons
        self.output_layer_neurons = output_layer_neurons

    def output(self, input_data):
        # type: (np.ndarray) -> float

        # TODO: Currently this is just looping over the first hidden layer, feeding each neuron
        # the input data, getting those outputs, and feeding those to the output layer neurons.
        # We need to feed the output of each hidden layer to the next hidden layer, then feed
        # the output of the final hidden layer into the output layer

        for hidden_layer in self.hidden_layer_neurons:
            for neuron in hidden_layer:  # type: HiddenLayerNeuron
                if neuron.get_first_weights_dimension() != len(input_data):
                    raise ValueError(
                        "The input_data {0} and weights {1} array dimensions do not match".format(
                            len(input_data),
                            neuron.get_first_weights_dimension()
                        )
                    )

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
