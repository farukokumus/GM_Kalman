import unittest
import numpy as np


from ime_fgs.unscented_utils import unscented_transform_gaussian, SigmaPointScheme
from ime_fgs.utils import col_vec


def fun(vector):
    vector = col_vec(vector)
    return np.linalg.norm(np.split(vector, 2))


class TestUnscentedFilter(unittest.TestCase):

    def test_different_in_out_dimensions(self):
        mean = [[1],
                [2],
                [3],
                [4]]

        cov = [[1, 0.5, 0, 0],
               [0.5, 1, 0, 0],
               [0, 0, 1, 0.5],
               [0, 0, 0.5, 1]]

        # test unscented dimensions
        for scheme in SigmaPointScheme:
            [res_mean, res_cov, cross_cov] = unscented_transform_gaussian(mean, cov, fun, sigma_point_scheme=scheme)

            self.assertTrue(res_cov.shape == (1, 1))
            self.assertTrue(res_mean.shape == (1, 1))
            self.assertTrue(cross_cov.shape == (4, 1))
            # todo check result numerically
