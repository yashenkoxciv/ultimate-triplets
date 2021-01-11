import numpy as np
from utils import euclidean_distance


def batch_hard(axs, ays):
    """ Returns hard positives and hard negatives for anchors (axs argument).

    Computes euclidean distance using numpy functionalities (W/O CUDA).
    Then, for each embedding finds farthest embedding as hard positive from the same class.
    As hard negatives it chooses nearest embedding from different class.

    :param axs: ndarray (N, EMBEDDING_SIZE)
    :param ays: ndarray (N,)
    :return: tuple, (hard_positives: ndarray (N, EMBEDDING_SIZE) , hard_negatives: ndarray (N, EMBEDDING_SIZE))
    """
    distance_matrix = euclidean_distance(axs, axs)
    # matrix side size
    ays = ays.reshape(-1, 1)
    num = ays.shape[0]
    diagonal_idxs = np.diag_indices(num)
    # mask items where labels are equal
    y_equals = ays == ays.T
    # let's find HARD POSITIVES
    y_equals[diagonal_idxs] = False
    d2p = distance_matrix.copy()
    # we are going to find hard positive - the most distant positive
    # therefore set -infinity distance to the same items
    d2p[y_equals == False] = -np.inf
    p_idxs = np.argmax(d2p, axis=1)
    hard_positives = axs[p_idxs]
    # let's find HARD NEGATIVES
    d2n = distance_matrix.copy()
    y_equals[diagonal_idxs] = True
    d2n[y_equals == True] = np.inf
    n_idxs = np.argmin(d2n, axis=1)
    hard_negatives = axs[n_idxs]
    return hard_positives, hard_negatives


