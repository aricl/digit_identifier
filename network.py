import numpy as np
from types import *
from hidden_or_output_layer_neuron import HiddenOrOutputLayerNeuron


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
        assert type(hidden_layer_neurons) is ListType, 'The hidden layer neurons passed is not a list'
        assert type(output_layer_neurons) is ListType, 'The output layer neurons passed is not a list'

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

        self._hidden_layer_neurons = hidden_layer_neurons
        self._output_layer_neurons = output_layer_neurons

    def output(self, input_data):
        # type: (np.ndarray) -> ListType
        assert type(input_data) is np.ndarray, 'The input data passed is not an np.ndarray'

        number_of_neurons_in_previous_layer = len(input_data)
        for hidden_layer in self._hidden_layer_neurons:
            for neuron in hidden_layer:  # type: HiddenOrOutputLayerNeuron
                first_dimension_of_current_weights = neuron.get_first_weights_dimension()
                if first_dimension_of_current_weights != number_of_neurons_in_previous_layer:
                    raise ValueError(
                        "The number of neurons in the previous layer {0} "
                        "and weights first dimension {1} do not match".format(
                            first_dimension_of_current_weights,
                            number_of_neurons_in_previous_layer
                        )
                    )
            number_of_neurons_in_previous_layer = len(hidden_layer)

        input_values = input_data
        output_values = []
        for hidden_layer in self._hidden_layer_neurons:
            output_values = [
                neuron.output(input_values)
                for neuron in hidden_layer
            ]
            input_values = output_values

        output_layer_outputs = [
            neuron.output(output_values)
            for neuron in self._output_layer_neurons
        ]

        return output_layer_outputs
