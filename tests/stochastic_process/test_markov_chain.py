import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest

from stochastic_process.markov_chain import MarkovChain, Sigma, mask, next_value, Q_from_coefficients


def test_mask():
    M = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3]).reshape((4, 3))
    r = np.array([2, 3, 7, 2])
    expected_mask = np.array([[False, False,  True],
                              [True,  True,   True],
                              [False,  True,  True],
                              [False, False,  True]])
    assert_array_equal(mask(r, M), expected_mask)
    arg = next_value(expected_mask)
    assert_array_equal(arg, np.array([2, 0, 1, 2]))


class TestMarkovChainWithTwoStates():

    # Q = np.array([[- 0.7, 0.7], [0.2, - 0.2]])
    transition_coefficients = (0.7, 0.2)
    X = MarkovChain(*transition_coefficients)

    @pytest.mark.parametrize(
        "step, expected_P",
        [
            (1, np.array([[0.3, 0.7],
                          [0.2, 0.8]])),
            (0.5, np.array([[0.65, 0.35],
                            [0.1, 0.9]]))
        ])
    def test_P(self, step, expected_P):
        computed_P = self.X.P(step)
        assert_array_almost_equal(computed_P, expected_P)

    def test_Q(self):
        assert_array_equal(self.X.Q, np.array([[- 0.7, 0.7], [0.2, - 0.2]]))

    @pytest.mark.parametrize(
        "step, P_step",
        [
            (1, np.array([[0.3, 0.7],
                          [0.2, 0.8]])),
            (0.5, np.array([[0.65, 0.35],
                            [0.1, 0.9]]))
        ])
    def test_P_with_step_equal_one(self, step, P_step):
        P = self.X.P(step)
        assert_array_almost_equal(P, P_step)

    step = 0.5
    random_sample = np.array([[0.2, 0.7],
                              [0.9, 0.3]])

    def test_paths(self):
        paths = self.X.paths(self.random_sample, self.step)
        r, c = self.random_sample.shape

        assert paths.shape == (r, c + 1)
        assert_array_almost_equal(paths, np.array([[0, 0, 1],
                                                   [0, 1, 1]]))

    def test_sigma(self):
        sigma = Sigma(0.3, 0.2)
        paths = self.X.paths(self.random_sample, self.step)
        assert_array_equal(sigma.paths(paths), np.array(
            [[0.3, 0.3, 0.2], [0.3, 0.2, 0.2]]))


class TestContinuousMarkovChainWithThreeStates():

    X = MarkovChain(0.7, 0.2, 0.2, 0.3, 0.4, 0.6)

    def test_Q(self):
        assert_array_almost_equal(self.X.Q, np.array([[- 0.9, 0.7, 0.2],
                                                      [0.2, - 0.5, 0.3],
                                                      [0.4, 0.6, -1.]]))

    @pytest.mark.parametrize(
        "step, P_step",
        [(1, np.array([[0.1, 0.7, 0.2],
                       [0.2, 0.5, 0.3],
                       [0.4, 0.6, 0]])),
         (0.5, np.array([[0.55, 0.35, 0.1],
                         [0.1, 0.75, 0.15],
                         [0.2, 0.3, 0.5]]))])
    def test_P(self, step, P_step):
        P = self.X.P(step)
        assert_array_almost_equal(P, P_step)

    step = 0.5
    random_sample = np.array([[0.2, 0.7, 0.3, 0.91],
                              [0.9, 0.3, 0.05, 0.9],
                              [0.95, 0.05, 0.6, 0.05],
                              [0.6, 0.9, 0.1, 0.7]])

    def test_generate_paths(self):
        paths = self.X.paths(self.random_sample, self.step)
        r, c = self.random_sample.shape

        assert paths.shape == (r, c + 1)
        assert_array_almost_equal(paths, np.array([[0, 0, 1, 1, 2],
                                                   [0, 2, 1, 0, 2],
                                                   [0, 2, 0, 1, 0],
                                                   [0, 1, 2, 0, 1]]))


@pytest.mark.parametrize('coefficients, expected_shape', [
    (np.array([1, 2]), (2, 2)),
    (np.array([4, 3, 3, 5, 7, 8]), (3, 3))
])
def test_Q_from_coefficients_shape(coefficients, expected_shape):
    assert Q_from_coefficients(
        coefficients).shape == expected_shape


def test_Q_from_coefficient_raise_value_error_when_len_of_coefficient_is_not_good():
    with pytest.raises(ValueError):
        Q_from_coefficients(np.array([1, 2, 3, 4, 5, 6, 7]))


@pytest.mark.parametrize('coefficients, expected_Q', [
    (np.array([1, 2]), np.array([[-1, 1], [2, -2]])),
    (np.array([4, 3]), np.array([[-4, 4], [3, -3]])),
    (np.array([1, 2, 3, 4, 5, 6]), np.array(
        [[-3, 1, 2], [3, -7, 4], [5, 6, -11]]))
])
def test_Q_from_coefficients(coefficients, expected_Q):
    assert_array_equal(Q_from_coefficients(coefficients), expected_Q)
