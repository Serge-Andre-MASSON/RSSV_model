from continuous_time_markov_chain import transition_matrix_for_two_states
from continuous_time_markov_chain.markov_chain import next_value_of_X, two_states_markov_chain


A = transition_matrix_for_two_states(0.7, 0.2)

def test_two_states_markov_chain():
    assert two_states_markov_chain(A, 1, 0.1).shape == (11,)

def test_next_value_of_X():
    # Here delta_t = 1. In reality it will be <<1.
    #TODO: Maybe change a little bit : use 0.7 in place of 1-0.7=0.3 for more understandabilty.
    assert next_value_of_X(0, 0.69, A, 1) == 1
    assert next_value_of_X(0, 0.71, A, 1) == 0
    assert next_value_of_X(1, 0.19, A, 1) == 1
    assert next_value_of_X(1, 0.21, A, 1) == 0