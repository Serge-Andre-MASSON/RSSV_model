from datetime import datetime
import pickle

import numpy as np
import pytest

from data.external_data.yahoo import Yahoo
from data.local_data.data_generator import DataGenerator
from data.local_data.parameters_generator import SigmaStatesGenerator, TransitionCoefficientsGenerator


with open('tests/data/external_data/ora.pickle', 'rb') as f:
    ora: Yahoo = pickle.load(f)


@pytest.fixture
def data_generator():
    return DataGenerator(ora, 2, 3, SigmaStatesGenerator(), TransitionCoefficientsGenerator())


def test_y_train_generator(data_generator: DataGenerator):
    y_train = data_generator.generate_y_train()
    assert y_train.shape == (2, 9)


def test_x_train_generator(data_generator: DataGenerator):
    x_train = data_generator.generate_x_train()
    assert x_train.shape == (2, 34)


def test_number_of_calls(data_generator: DataGenerator):
    number_of_calls = data_generator.number_of_calls
    assert number_of_calls == 34


def test_y_train_with_data_size_equal_one():
    y_train = DataGenerator(ora, 1, 3, SigmaStatesGenerator(
    ), TransitionCoefficientsGenerator()).generate_y_train()
    assert y_train.shape == (9,)


def test_x_real(data_generator: DataGenerator):
    x_real = data_generator.generate_x_real()
    assert x_real.shape == (34, )
