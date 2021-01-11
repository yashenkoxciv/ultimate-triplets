import numpy as np


def euclidean_distance(a, b):
    """ Computes Euclidean distance between to sets of vectors.

    :param a: ndarray (N, K)
    :param b: ndarray (M, K)
    :return: ndarray (N, M)
    """
    ed = np.sqrt(np.mean(np.square(a[:, np.newaxis, :] - b), axis=2))
    return ed


