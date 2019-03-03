import numpy as np
from network_builder import NetworkBuilder
from data_loader import load_formatted_data
from sklearn.metrics import mean_squared_error


if __name__ == '__main__':
    training_data = load_formatted_data()[0]
    pixel_data = training_data[0][0]
    identifying_digit = training_data[0][1]
    number_of_inputs = len(pixel_data)
    number_of_hidden_neurons = 16
    number_of_hidden_neurons_second_layer = 16
    number_of_output_neurons = 10

    hidden_layer_weights = [
        np.random.uniform(0, 1, number_of_inputs)
        for i in range(0, number_of_hidden_neurons)
    ]

    hidden_layer_biases = [
        np.random.uniform(-np.exp(100), 0)
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

    network_output = network.output(pixel_data)

    root_mean_squared_error = np.sqrt(mean_squared_error(network_output, identifying_digit))
    print(root_mean_squared_error)
