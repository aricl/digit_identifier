import bigfloat


def sigmoid():
    """
    Returns a lambda function that takes the form of the sigmoid function.
    The package bigfloat is used to get more precision
    :return: float
    """
    return lambda z: float(1 / (1 + bigfloat.exp(z, bigfloat.precision(100))))


def perceptron():
    """
    Returns a lambda function that takes the form of the perceptron or step
    function.
    :return: int
    """
    return lambda z: 1 if z > 0 else 0


def relu():
    """
    Returns a lambda function that takes the form of the rectified linear
    unit or ReLU function.
    :return: float
    """
    return lambda z: z if z > 0 else 0
