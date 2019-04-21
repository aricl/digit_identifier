from data_loader import load_formatted_data
from error_calculator import calculate_error
from network_builder import NetworkBuilder


def calculate_cost(weights: list, biases: list):
    training_data = load_formatted_data()[0]
    number_of_inputs = len(training_data[0][0])

    hidden_layer_weights = weights[0]
    second_hidden_layer_weights = weights[1]
    hidden_layer_biases = biases[0]
    second_hidden_layer_biases = biases[1]
    number_of_output_neurons = len(weights[2])
    output_neuron_weights = weights[2]
    output_neuron_biases = biases[2]

    network = NetworkBuilder() \
        .add_input_layer(number_of_inputs) \
        .add_hidden_layer(len(hidden_layer_weights), hidden_layer_weights, hidden_layer_biases) \
        .add_hidden_layer(len(second_hidden_layer_weights), second_hidden_layer_weights, second_hidden_layer_biases) \
        .add_output_layer(number_of_output_neurons, output_neuron_weights, output_neuron_biases) \
        .build_network()

    return calculate_error(network, training_data)
