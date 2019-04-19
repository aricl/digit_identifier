import numpy as np
from network import Network


def calculate_error(network: Network, training_data: list):
    """
    Calculates the error using this formula:

        error = (1 / (2 * n)) * (sum over x ( y(x) - a )**2)

    where 'x' is a piece of training data, i.e. a 784 element vector representing a 28x28 pixel
    grid of grayscale values represented by values in the range 0..1.
    'a' is the digit 'i' that identifies the piece of training data represented as a 10-dimensional
    vector with 0's for all its values except the (i-1)'th place which has the value 1.
    y(x) is the network function that takes a piece of training data and returns a similar vector
    to 'a' which is the network's guess at the true value.
    :return: float
    """
    training_data_errors = []
    for training_datum in training_data:
        identifying_digit = training_datum[1]
        pixel_data = training_datum[0]
        network_output = network.output(pixel_data)
        training_datum_error = np.sum((network_output - identifying_digit) ** 2)
        training_data_errors = np.append(training_data_errors, training_datum_error)

    mean_squared_error = (1 / (2.0 * len(training_data_errors))) * np.sum(training_data_errors)

    return mean_squared_error
