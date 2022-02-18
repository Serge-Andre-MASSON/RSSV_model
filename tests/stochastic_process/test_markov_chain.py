# TODO: Pretty this
import numpy as np
from stochastic_process.markov_chain import ContinuousMarkovChain


def test_ContinuousMarkovChain():
    X = ContinuousMarkovChain(0.7, 0.2)

    #######################
    # Test generate_paths #
    #######################

    number_of_paths = 2
    length_of_paths = 4
    step = 1

    test_sample = np.array([[0.71, 0.68, 0.17],
                            [0.5, 0.12, 0.35]])

    paths = X.generate_paths(
        number_of_paths=number_of_paths,
        length_of_paths=length_of_paths,
        step=step,
        random_sample=test_sample)

    assert paths.shape == (number_of_paths, length_of_paths)

    assert (paths == np.array([[0, 0, 1, 1],
                               [0, 1, 1, 0]])).all()

    #############
    # Test generate_one_path #
    #############

    assert (X.generate_one_path(length_of_path=length_of_paths,
                                step=step, random_sample=test_sample[0]) == np.array([0, 0, 1, 1])).all()
    assert (X.generate_one_path(length_of_path=length_of_paths,
                                step=step, random_sample=test_sample[1]) == np.array([0, 1, 1, 0])).all()

    ########################
    # Test next_value_of_x #
    ########################

    assert X.next_value_of_X(0, 0.69, 1) == 1
    assert X.next_value_of_X(0, 0.71, 1) == 0
    assert X.next_value_of_X(1, 0.19, 1) == 1
    assert X.next_value_of_X(1, 0.21, 1) == 0
