import numpy as np
from numpy.testing import assert_array_almost_equal
from stochastic_process.transition_matrix import random_transition_matrix


def test_transition_matrix_random_generator():
    test_random_sample = np.array([[0, 2, 5],
                                   [4, 2, 6],
                                   [7, 4, 1]])
    transition_matrix = random_transition_matrix(
        3, test_random_sample)
    assert_array_almost_equal(
        transition_matrix,
        np.array([[-7, 2, 5],
                  [4, -10, 6],
                  [7, 4, -11]]))
