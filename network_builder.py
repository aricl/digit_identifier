from types import *
from hidden_or_output_layer_neuron import HiddenOrOutputLayerNeuron
from activation_function import sigmoid
from network import Network


class NetworkBuilder:
    def __init__(self):
        self._number_of_input_neurons = None
        self._hidden_layers_distribution = {}
        self._hidden_layers_weights = {}
        self._hidden_layers_biases = {}
        self._number_of_output_neurons = None
        self._output_neuron_weights = {}
        self._output_neuron_biases = {}

    def add_input_layer(self, number_of_neurons):
        # type: (IntType) -> NetworkBuilder
        assert type(number_of_neurons) is IntType, 'The number of input neurons passed is not an integer'
        self._number_of_input_neurons = number_of_neurons

        return self

    def add_hidden_layer(self, number_of_neurons, hidden_layer_weights, hidden_layer_biases):
        # type: (int, ListType, ListType) -> NetworkBuilder
        assert type(number_of_neurons) is IntType, 'The number of hidden layer neurons passed is not an integer'
        assert type(hidden_layer_weights) is ListType, 'The hidden layer weights passed is not a list'
        assert type(hidden_layer_biases) is ListType, 'The hidden layer biases passed is not a list'

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

        hidden_layer_weights_dimension = len(hidden_layer_weights[0])
        if len(self._hidden_layers_weights) == 0:
            if hidden_layer_weights_dimension != self._number_of_input_neurons:
                raise ValueError(
                    'The dimension of the weights array in the hidden layer {0} and the number of inputs {1} do not '
                    'match.'.format(
                        hidden_layer_weights_dimension,
                        self._number_of_input_neurons
                    )
                )
        else:
            previous_hidden_layer_key = len(self._hidden_layers_weights) - 1
            number_of_neurons_in_previous_hidden_layer = len(self._hidden_layers_weights[previous_hidden_layer_key])
            if hidden_layer_weights_dimension != number_of_neurons_in_previous_hidden_layer:
                raise ValueError(
                    'The dimension of the weights array in the hidden layer {0} and number of neurons in the previous '
                    'hidden array {1} do not match.'.format(
                        hidden_layer_weights_dimension,
                        number_of_neurons_in_previous_hidden_layer
                    )
                )

        self._hidden_layers_distribution.update({len(self._hidden_layers_distribution): number_of_neurons})
        self._hidden_layers_weights.update({len(self._hidden_layers_weights): hidden_layer_weights})
        self._hidden_layers_biases.update({len(self._hidden_layers_biases): hidden_layer_biases})

        return self

    def add_output_layer(self, number_of_neurons, output_layer_weights, output_layer_biases):
        # type: (int, ListType, ListType) -> NetworkBuilder
        assert type(number_of_neurons) is IntType, 'The number of output neurons passed is not an integer'
        assert type(output_layer_weights) is ListType, 'The output layer weights passed is not a list'
        assert type(output_layer_biases) is ListType, 'The output layer biases passes is not a list'

        output_layer_weights_dimension = len(output_layer_weights[0])
        hidden_layer_key = len(self._hidden_layers_weights) - 1
        number_of_neurons_in_nearest_hidden_layer = len(self._hidden_layers_weights[hidden_layer_key])
        if output_layer_weights_dimension != number_of_neurons_in_nearest_hidden_layer:
            raise ValueError(
                'The dimension of the weights array in the output layer {0} and number of neurons in the nearest '
                'hidden array {1} do not match.'.format(
                    output_layer_weights_dimension,
                    number_of_neurons_in_nearest_hidden_layer
                )
            )

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
                HiddenOrOutputLayerNeuron(
                    self._hidden_layers_weights[key][i],
                    self._hidden_layers_biases[key][i],
                    sigmoid()
                )
                for i in range(0, number_of_neurons)
            ]
            for key, number_of_neurons in self._hidden_layers_distribution.iteritems()
        ]

        output_layer_neurons = [
            HiddenOrOutputLayerNeuron(
                self._output_neuron_weights[i],
                self._output_neuron_biases[i],
                sigmoid()
            )
            for i in range(0, self._number_of_output_neurons)
        ]

        return Network(hidden_layer_neurons, output_layer_neurons)
