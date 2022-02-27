import numpy as np


def random_transition_matrix(number_of_states, test_random_sample: np.ndarray = None):
    if test_random_sample is None:
        transition_matrix = np.random.rand(number_of_states, number_of_states)
    else:
        transition_matrix = test_random_sample

    for i in range(number_of_states):
        transition_matrix[i, i] = 0
        transition_matrix[i, i] = - transition_matrix[i].sum()
    return transition_matrix
