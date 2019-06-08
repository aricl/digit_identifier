import numpy as np


def flatten(
        first_hidden_layer_values: list,
        second_hidden_layer_values: list,
        output_layer_values: list
):
    """
    This method is used to take all the weights or biases of a network, arranged as three
    arrays of differing dimensions, and flatten the whole thing into a vector.
    :param first_hidden_layer_values: list
    :param second_hidden_layer_values: list
    :param output_layer_values: list
    :return: list
    """
    flattened_first_hidden_layer_values = _flatten_layer_values(first_hidden_layer_values)
    flattened_second_hidden_layer_values = _flatten_layer_values(second_hidden_layer_values)
    flattened_output_layer_values = _flatten_layer_values(output_layer_values)

    return np.concatenate(
        [
            flattened_first_hidden_layer_values,
            flattened_second_hidden_layer_values,
            flattened_output_layer_values
        ]
    ).tolist()


def _flatten_layer_values(layer_values):
    layer_shape = np.shape(layer_values)
    return np.reshape(
        layer_values,
        (layer_shape[0] * layer_shape[1],)
    )


def unflatten(
    flattened_values: list,
    number_of_input_nodes: int,
    number_of_first_hidden_layer_nodes: int,
    number_of_second_hidden_layer_nodes: int,
    number_of_output_nodes: int
):
    """
    This method is used to take a single one-dimensional of weights or biases flattened by the
    above flatten function and return the weights or biases as three two-dimensional lists
    containing the weights or biases of the first hidden layer, second hidden layer, and output
    layer respectively. The three two-dimensional lists are themselves enclosed in a list.
    :param flattened_values:
    :param number_of_input_nodes:
    :param number_of_first_hidden_layer_nodes:
    :param number_of_second_hidden_layer_nodes:
    :param number_of_output_nodes:
    :return:
    """
    number_of_first_layer_values = number_of_input_nodes * number_of_first_hidden_layer_nodes
    first_hidden_layer_values = np.reshape(
        flattened_values[0:number_of_first_layer_values],
        (number_of_first_hidden_layer_nodes, number_of_input_nodes)
    )

    number_of_second_hidden_layer_values = number_of_first_hidden_layer_nodes * number_of_second_hidden_layer_nodes
    second_hidden_layer_values = np.reshape(
        flattened_values[
            number_of_first_layer_values
            :number_of_first_layer_values + number_of_second_hidden_layer_values
        ],
        (number_of_second_hidden_layer_nodes, number_of_first_hidden_layer_nodes)
    )

    number_of_output_layer_values = number_of_second_hidden_layer_nodes * number_of_output_nodes
    output_layer_values = np.reshape(
        flattened_values[
            number_of_first_layer_values + number_of_second_hidden_layer_values
            :number_of_first_layer_values + number_of_second_hidden_layer_values + number_of_output_layer_values
        ],
        (number_of_output_nodes, number_of_second_hidden_layer_nodes)
    )

    return [first_hidden_layer_values, second_hidden_layer_values, output_layer_values]
