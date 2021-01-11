import os
import sys
sys.path.insert(0, os.path.abspath('../'))
import unittest
import numpy as np
from parameterized import parameterized
from batch_generator import batch_hard
from sklearn.datasets import make_blobs


class BatchHardTestCase(unittest.TestCase):
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
    def test_batch_generator(self, xs, ys):
        hard_positives, hard_negatives = batch_hard(xs, ys)
        is_ok = True
        for i in range(xs.shape[0]):
            x = xs[i]
            x_hard_positive_idx = None
            x_hard_negative_idx = None
            x_hp_d = -np.inf
            x_hn_d = np.inf
            for j in range(xs.shape[0]):
                if i == j:
                    continue
                z = xs[j]
                rmse = np.sqrt(np.mean(np.square(x - z)))
                if ys[j] == ys[i]:  # looking for hard positive
                    if rmse > x_hp_d:
                        x_hp_d = rmse
                        x_hard_positive_idx = j
                else:  # looking for hard negative
                    if rmse < x_hn_d:
                        x_hn_d = rmse
                        x_hard_negative_idx = j
            is_hard_positive_ok = np.array_equal(hard_positives[i], xs[x_hard_positive_idx])
            is_hard_negative_ok = np.array_equal(hard_negatives[i], xs[x_hard_negative_idx])
            is_ok = is_ok and is_hard_positive_ok and is_hard_negative_ok
        self.assertEqual(is_ok, True)


if __name__ == '__main__':
    unittest.main()
