import activation_function
import hidden_layer_neuron
import numpy as np

if __name__ == '__main__':
    number_of_inputs = 10
    number_of_hidden_neurons = 12

    neurons_parameters = [
        [
            np.random.uniform(0, 1, number_of_inputs),
            np.random.uniform(-5, 0)
        ]
        for i in range(0, number_of_hidden_neurons)
    ]

    neurons = [
        hidden_layer_neuron.HiddenLayerNeuron(
            neuron_parameters[0],
            neuron_parameters[1],
            activation_function.sigmoid()
        )
        for neuron_parameters in neurons_parameters
    ]

    neurons_output = [
        neuron.output(np.random.uniform(0, 1, number_of_inputs))
        for neuron in neurons
    ]

    output_neuron = hidden_layer_neuron.HiddenLayerNeuron(
        np.random.uniform(0, 1, number_of_hidden_neurons),
        np.random.uniform(-5, 0),
        activation_function.sigmoid()
    )

    print(output_neuron.output(neurons_output))
