import numpy as np

from stochastic_process import BrownianMotion


def test_BrownianMotion():
    W = BrownianMotion()
    number_of_paths = 2
    length_of_paths = 4
    step = 1

    assert W.generate_paths(number_of_paths, length_of_paths, step).shape == (
        number_of_paths, length_of_paths)

    random_sample = np.array([[2, 1, -1],
                              [2, 1, 1]])
    assert (W.generate_paths(number_of_paths, length_of_paths, step, random_sample)
            == np.array([[0, 2, 3, 2], [0, 2, 3, 4]])).all()
