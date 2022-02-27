import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest

from stochastic_process import ContinuousMarkovChain, agregate_matrix


class TestContinuousMarkovChainWithTwoStates():

    Q = np.array([[- 0.7, 0.7], [0.2, - 0.2]])
    X = ContinuousMarkovChain(Q)

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

    number_of_paths = 2
    length_of_paths = 3
    step = 0.5
    test_sample = np.array([[0.2, 0.7],
                            [0.9, 0.3]])

    def test_generate_paths(self):
        paths = self.X.generate_paths(
            self.number_of_paths, self.length_of_paths, self.step, self.test_sample)

        assert paths.shape == (self.number_of_paths, self.length_of_paths)
        assert_array_almost_equal(paths, np.array([[0, 0, 1],
                                                   [0, 1, 1]]))


class TestContinuousMarkovChainWithThreeStates():

    Q = np.array([[- 0.9, 0.7, 0.2],
                  [0.2, - 0.5, 0.3],
                  [0.4, 0.6, -1.]])
    X = ContinuousMarkovChain(Q)

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

    number_of_paths = 4
    length_of_paths = 5
    step = 0.5
    test_sample = np.array([[0.2, 0.7, 0.3, 0.91],
                            [0.9, 0.3, 0.05, 0.9],
                            [0.95, 0.05, 0.6, 0.05],
                            [0.6, 0.9, 0.1, 0.7]])

    def test_generate_paths(self):
        paths = self.X.generate_paths(
            self.number_of_paths, self.length_of_paths, self.step, self.test_sample)

        assert paths.shape == (self.number_of_paths, self.length_of_paths)
        assert_array_almost_equal(paths, np.array([[0, 0, 1, 1, 2],
                                                   [0, 2, 1, 0, 2],
                                                   [0, 2, 0, 1, 0],
                                                   [0, 1, 2, 0, 1]]))
        pass


def test_agregate_matrix():
    M = np.array([[1, 2, 3],
                  [2, 4, 5],
                  [0, 5, 3]])
    print(agregate_matrix(M))
    assert_array_equal(agregate_matrix(M),  np.array([[1, 3, 6],
                                                      [2, 6, 11],
                                                      [0, 5, 8]]))
