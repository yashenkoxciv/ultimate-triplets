import os
import sys
sys.path.insert(0, os.path.abspath('../'))
import unittest
import numpy as np
from parameterized import parameterized
from utils import euclidean_distance
from sklearn.datasets import make_blobs


class EDTestCase(unittest.TestCase):
    @parameterized.expand([
        [*make_blobs(n_samples=10, centers=2, n_features=2)],
        [*make_blobs(n_samples=100, centers=2, n_features=2)],
        [*make_blobs(n_samples=1000, centers=2, n_features=2)],
        [*make_blobs(n_samples=1000, centers=5, n_features=2)],
        [*make_blobs(n_samples=1000, centers=7, n_features=2)],
        [*make_blobs(n_samples=1000, centers=7, n_features=10)],
        [*make_blobs(n_samples=1000, centers=10, n_features=20)],
        [*make_blobs(n_samples=1000, centers=10, n_features=200)],
        [*make_blobs(n_samples=1000, centers=100, n_features=200)],
    ])
    def test_euclidean_distance(self, xs, ys):
        """ Tests equivalence RMSE and computed distance."""
        distance_matrix = euclidean_distance(xs, xs)
        is_ok = True
        for i in range(distance_matrix.shape[0]):
            rmse = np.sqrt(np.mean(np.square(xs[i:i+1] - xs), 1))
            dm_1st_row = distance_matrix[i, :]
            is_ok = is_ok and np.array_equal(dm_1st_row, rmse)
        self.assertEqual(is_ok, True)


if __name__ == '__main__':
    unittest.main()
