import numpy as np


class HiddenLayerNeuron:
    _activation_function = None  # type: function
    _bias = None  # type: float
    _weights = None  # type: np.ndarray

    def __init__(self, weights, bias, activation_function_input):
        """
        Constructs a hidden layer neuron by specifying its
        weights, bias, and the activation function it uses.
        :param weights: ListType
        :param bias: float
        :param activation_function_input:
        """
        assert type(weights) is np.ndarray, 'The weights provided to the HiddenLayer Neuron are not an ndarray, ' \
                                            'they are %r' % type(weights)
        assert isinstance(
            bias,
            (float, np.float64)
        ), 'The bias provided to the HiddenLayer Neuron is not an float, ' \
           'it is a %r' % type(bias)

        self._weights = weights
        self._bias = bias
        self._activation_function = activation_function_input

    def output(self, input_data):
        """
        Takes input data, a numpy array of the same dimensionality as the weights
        and returns the dot product of those two arrays plus the bias.
        :param input_data: np.ndarray
        :return: float
        """
        if len(input_data) == len(self._weights):
            uncompressed_output = np.dot(self._weights, input_data) + self._bias
            compressed_output = self._activation_function(uncompressed_output)
        else:
            raise ValueError(
                "The input_data {0} and weights {1} array dimensions do not match".format(
                    len(input_data),
                    len(self._weights)
                )
            )

        return compressed_output

    def get_first_weights_dimension(self):
        # type: () -> tuple
        return self._weights.shape[0]
