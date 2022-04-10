import pytest
from data.local_data.parameters_generator import MuStatesGenerator, SigmaStatesGenerator, StatesGenerator, TransitionCoefficientsGenerator


@pytest.fixture
def states_generator() -> StatesGenerator:
    return StatesGenerator()


@pytest.fixture
def mu_states_generator():
    return MuStatesGenerator()


@pytest.fixture
def sigma_states_generator():
    return SigmaStatesGenerator()


@pytest.fixture
def transition_coefficients_generator():
    return TransitionCoefficientsGenerator()


# Test the shape of the random sample use to generate states or transition coefficients
def test_shape_of_states_generator_random_sample(states_generator: StatesGenerator):
    assert states_generator.shape_of_random_sample(
        size=1,
        number_of_states=3
    ) == (1, 3)


def test_shape_of_transition_coefficients_generator_random_sample(transition_coefficients_generator: TransitionCoefficientsGenerator):
    assert transition_coefficients_generator.shape_of_random_sample(
        size=1,
        number_of_states=3
    ) == (1, 6)


# Test the shape of states like generated parameters
def test_shape_of_sigma_parameters_when_size_equal_one(sigma_states_generator: SigmaStatesGenerator):
    assert sigma_states_generator.generate(
        size=1, number_of_states=2).shape == (2,)


def test_shape_of_sigma_parameters(sigma_states_generator: SigmaStatesGenerator):
    assert sigma_states_generator.generate(
        size=3, number_of_states=2).shape == (3, 2)


# Test the shape of transition coefficients like parameters
def test_shape_of_transition_coefficients_parameters_when_size_equal_one(transition_coefficients_generator: TransitionCoefficientsGenerator):
    assert transition_coefficients_generator.generate(
        size=1,
        number_of_states=3
    ).shape == (6,)


def test_shape_of_transition_coefficients_parameters_when_size_equal_one(transition_coefficients_generator: TransitionCoefficientsGenerator):
    assert transition_coefficients_generator.generate(
        size=7,
        number_of_states=3
    ).shape == (7, 6)


# Test that set_min_or_max works properly
def test_set_min_or_max(states_generator: StatesGenerator):
    assert states_generator.min is None
    assert states_generator.max is None

    states_generator.set_min_or_max(min=0.5)
    assert states_generator.min == 0.5

    states_generator.set_min_or_max(max=0.7)
    assert states_generator.max == 0.7

    states_generator.set_min_or_max(min=0.2, max=0.4)
    assert states_generator.min == 0.2
    assert states_generator.max == 0.4
