import numpy as np

# TODO: Make that rows sum to zero, not cols.

def transition_matrix_for_two_states(a, b):
    return np.array([[-a, a], [b, -b]])

def transition_matrix_for_more_than_two_states(array):
    n = len(array)
    transition_matrix = np.zeros((n, n))
    for i in range(n):
        transition_matrix[i, i] = - array[i].sum()
        transition_matrix[i, :i] = array[i, :i]
        transition_matrix[i, i + 1:] = array[i, i:]
    return transition_matrix
        