from abc import ABC, abstractmethod
from typing import List
import numpy as np


class ParametersGenerator(ABC):
    min: float
    max: float

    def generate(self, size: int, number_of_states: int, **kwargs) -> np.ndarray:
        self.set_min_or_max(**kwargs)

        r, c = self.shape_of_random_sample(size, number_of_states)

        if r == 1:
            random_sample = np.random.rand(c)
        else:
            random_sample = np.random.rand(r, c)
        parameters = (
            self.max - self.min) * random_sample + self.min

        return parameters

    def set_min_or_max(self, **kwargs):
        if 'min' in kwargs:
            self.min = kwargs['min']
        if 'max' in kwargs:
            self.max = kwargs['max']

    @abstractmethod
    def shape_of_random_sample(self, size, number_of_states):
        pass


class State(ParametersGenerator):

    def shape_of_random_sample(self, size, number_of_states):
        return size, number_of_states


class MuStatesGenerator(State):
    min = 0.01
    max = 0.1


class SigmaStatesGenerator(State):
    min = 0.1
    max = 2


class TransitionCoefficientsGenerator(ParametersGenerator):
    min = 1
    max = 20

    def shape_of_random_sample(self, size, number_of_states):
        return size, number_of_states * (number_of_states - 1)
