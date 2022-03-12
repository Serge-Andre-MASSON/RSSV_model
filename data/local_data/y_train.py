from abc import ABC, abstractmethod
from typing import List
import numpy as np


class ParametersGenerator(ABC):
    min: float
    max: float

    def generate(self, size, number_of_states, **kwargs):
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


class YTrain:
    def __init__(self, size: int, number_of_states: int, *args: ParametersGenerator):
        self.generators: List[ParametersGenerator] = []
        self.size = size
        self.number_of_states = number_of_states
        for arg in args:
            if not isinstance(arg, ParametersGenerator):
                raise(ValueError('Oh!'))
            self.generators.append(arg)

    def flatten(self):
        flat = None
        for generator in self.generators:
            if flat is None:
                flat = generator.generate(self.size, self.number_of_states)
            else:
                flat: np.ndarray = np.concatenate((flat, generator.generate(
                    self.size, self.number_of_states)), axis=1)
        return flat
