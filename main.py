import activation_function
import hidden_layer_neuron
import numpy as np

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
        hidden_layer_neuron.HiddenLayerNeuron(
            hidden_layer_weights[i],
            hidden_layer_biases[i],
            activation_function.sigmoid()
        )
        for i in range(0, number_of_hidden_neurons)
    ]

    hidden_layer_neurons_output = [
        neuron.output(np.random.uniform(0, 1, number_of_inputs))
        for neuron in hidden_layer_neurons
    ]

    output_neuron_weights = np.random.uniform(0, 1, number_of_hidden_neurons)
    output_neuron_bias = np.random.uniform(-5, 0)
    output_neuron = hidden_layer_neuron.HiddenLayerNeuron(
        output_neuron_weights,
        output_neuron_bias,
        activation_function.sigmoid()
    )

    print(output_neuron.output(hidden_layer_neurons_output))
