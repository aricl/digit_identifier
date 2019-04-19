import pickle
import gzip
import numpy as np


def load_data():
    """
    This method imports the data contained in the MNIST file into the
    python runtime. It comes in three parts: the training, validation,
    and test data-sets.

    The training data is a tuple of two parts. The first part contains
    50,000 entries. Each entry is an ndarray of 784 values representing
    the 28 * 28 = 784 pixels of the hand-written MNIST test digits.

    The second part also has 50,000 entries, but these entries are single
    digits. These entries are just digits values from 0..9 identifying the
    what digit is represented by the corresponding image in the first part
    of the training data tuple.

    The validation and test data-sets are similar to the training data,
    but they only have 10,000 entries apiece.
    :return:
    """
    file_to_be_loaded = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(file_to_be_loaded, encoding='latin1')
    file_to_be_loaded.close()

    return training_data, validation_data, test_data


def load_formatted_data():
    """
    This method formats the data loaded by the load_data method into a
    more useful format for the network. The training data is converted
    into another two element tuple, but this time the first part is
    a set of 784 element vectors rather than 28 x 28 arrays. The
    second part of the tuple is an output vector which has 0's for all
    of its elements except a 1 at the identifying digit's position.
    For the validation and test data the formatting step for the
    first part, the digit data, is carried out, but not the second
    step for the digit identifiers.
    :return:
    """
    training_data, validation_data, test_data = load_data()
    # The pixel grayscale values represented as a 784 element ndarray
    training_inputs = [np.reshape(digit_image_data, (784,)) for digit_image_data in training_data[0]]
    training_results = [_convert_digit_into_output_vector(digit) for digit in training_data[1]]
    formatted_training_data = list(zip(training_inputs, training_results))
    # The pixel grayscale values represented as a 784 element ndarray
    validation_inputs = [np.reshape(digit_image_data, (784,)) for digit_image_data in validation_data[0]]
    formatted_validation_data = list(zip(validation_inputs, validation_data[1]))
    # The pixel grayscale values represented as a 784 element ndarray
    test_inputs = [np.reshape(digit_image_data, (784,)) for digit_image_data in test_data[0]]
    formatted_test_data = list(zip(test_inputs, test_data[1]))

    return formatted_training_data, formatted_validation_data, formatted_test_data


def _convert_digit_into_output_vector(digit: int) -> np.ndarray:
    """
    This method takes a unitary integer digit 'n' and then returns an
    ndarray where the 'n'th element of that ndarray has the value of 1.0
    and all the other elements have the value 0.0.
    :param digit:
    :return: ndarray
    """
    output_vector = np.zeros((10,))
    output_vector[digit] = 1.0

    return output_vector
