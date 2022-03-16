import numpy as np

from numpy.testing import assert_array_equal

import pytest
from data.local_data.parameters_generator import MuStatesGenerator, SigmaStatesGenerator, TransitionCoefficientsGenerator


@pytest.fixture
def mu_states():
    return MuStatesGenerator()


@pytest.fixture
def sigma_states():
    return SigmaStatesGenerator()


@pytest.fixture
def transition_coefficients():
    return TransitionCoefficientsGenerator()


# @pytest.fixture
# def y_train_without_mu(sigma_states, transition_coefficients):
#     size = 5
#     number_of_states = 3
#     return YTrain(size, number_of_states, sigma_states, transition_coefficients)


# @pytest.fixture
# def y_train_with_mu(mu_states, sigma_states, transition_coefficients):
#     size = 5
#     number_of_states = 3
#     return YTrain(size, number_of_states, mu_states, sigma_states, transition_coefficients)


# def test_flatten_parameters_without_mu(y_train_without_mu: YTrain):
#     flatten_coeff = y_train_without_mu.flatten()
#     assert flatten_coeff.shape == (5, 3 + 6)


# def test_flatten_parameters_with_mu(mu_states, sigma_states, transition_coefficients):
#     size = 5
#     number_of_states = 3
#     y_train_with_mu = YTrain(size, number_of_states,
#                              mu_states, sigma_states, transition_coefficients)
#     print(y_train_with_mu.generators)
#     flatten_coeff = y_train_with_mu.flatten()
#     assert flatten_coeff.shape == (5, 3 + 3 + 6)


# def test_Ytrain_raise_error_when_bad_object_are_given():
#     with pytest.raises(ValueError):
#         sigma = SigmaStatesGenerator()
#         Y = YTrain(1, 2, sigma, 2, sigma)


def test_sigma_generator_with_max_equal_min():
    sigma = SigmaStatesGenerator()
    min = sigma.min
    sigma_states = sigma.generate(1, 2, **{'max': min})
    assert_array_equal(sigma_states, np.zeros((2,)) + min)
