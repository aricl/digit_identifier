from activation_function import sigmoid
from hidden_layer_neuron import HiddenLayerNeuron
import numpy as np
from network_builder import NetworkBuilder

if __name__ == '__main__':
    number_of_inputs = 10
    number_of_hidden_neurons = 12

    hidden_layer_weights = [
        np.random.uniform(0, 1, number_of_inputs)
        for i in range(0, number_of_hidden_neurons)
    ]

    hidden_layer_biases = [
        np.random.uniform(-5, 0)
        for i in range(0, number_of_hidden_neurons)
    ]

    hidden_layer_neurons = [
        HiddenLayerNeuron(
            hidden_layer_weights[i],
            hidden_layer_biases[i],
            sigmoid()
        )
        for i in range(0, number_of_hidden_neurons)
    ]

    hidden_layer_neurons_output = [
        neuron.output(np.random.uniform(0, 1, number_of_inputs))
        for neuron in hidden_layer_neurons
    ]

    output_neuron_weights = np.random.uniform(0, 1, number_of_hidden_neurons)
    output_neuron_bias = np.random.uniform(-5, 0)
    output_neuron = HiddenLayerNeuron(
        output_neuron_weights,
        output_neuron_bias,
        sigmoid()
    )

    network = NetworkBuilder()\
        .add_input_layer(number_of_inputs)\
        .add_hidden_layer(len(hidden_layer_weights), hidden_layer_weights, hidden_layer_biases) \
        .add_output_layer(
            1,
            np.array([output_neuron_weights]),
            np.array([output_neuron_bias])
        )\
        .build_network()

    print(network.output(np.random.uniform(0, 1, number_of_inputs)))
