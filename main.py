import numpy as np
from network_builder import NetworkBuilder

if __name__ == '__main__':
    number_of_inputs = 10
    number_of_hidden_neurons = 12
    number_of_hidden_neurons_second_layer = 14
    number_of_output_neurons = 10

    hidden_layer_weights = [
        np.random.uniform(0, 1, number_of_inputs)
        for i in range(0, number_of_hidden_neurons)
    ]

    hidden_layer_biases = [
        np.random.uniform(-5, 0)
        for i in range(0, number_of_hidden_neurons)
    ]

    second_hidden_layer_weights = [
        np.random.uniform(0, 1, number_of_hidden_neurons)
        for i in range(0, number_of_hidden_neurons_second_layer)
    ]

    second_hidden_layer_biases = [
        np.random.uniform(-5, 0)
        for i in range(0, number_of_hidden_neurons_second_layer)
    ]

    output_neuron_weights = [
        np.random.uniform(0, 1, number_of_hidden_neurons_second_layer)
        for i in range(0, number_of_output_neurons)
    ]
    output_neuron_biases = [
        np.random.uniform(-5, 0)
        for i in range(0, number_of_output_neurons)
    ]

    network = NetworkBuilder() \
        .add_input_layer(number_of_inputs) \
        .add_hidden_layer(len(hidden_layer_weights), hidden_layer_weights, hidden_layer_biases) \
        .add_hidden_layer(len(second_hidden_layer_weights), second_hidden_layer_weights, second_hidden_layer_biases) \
        .add_output_layer(number_of_output_neurons, output_neuron_weights, output_neuron_biases) \
        .build_network()

    print(network.output(np.random.uniform(0, 1, number_of_inputs)))
