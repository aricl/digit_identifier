import numpy as np

from cost_function import calculate_cost
from data_loader import load_formatted_data


if __name__ == '__main__':
    training_data = load_formatted_data()[0]
    number_of_inputs = len(training_data[0][0])
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

    cost = calculate_cost(
        [hidden_layer_weights, second_hidden_layer_weights, output_neuron_weights],
        [hidden_layer_biases, second_hidden_layer_biases, output_neuron_biases]
    )

    print(cost)