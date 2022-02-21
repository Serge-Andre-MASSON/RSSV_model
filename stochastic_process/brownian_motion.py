import numpy as np


class BrownianMotion:
    """Implement a Brownian motion generator."""

    def generate_paths(self, number_of_paths: int, length_of_paths: int, step: float, random_sample: np.ndarray = None):
        """Returns an numpy array of shape (number_of_paths, length_of_paths) and where each row is a Brownian motion discretized according to step. The random_sample is mostly here for tests purposes and allows to provide random data from outside.
        """
        W = np.zeros((number_of_paths, length_of_paths))
        if random_sample is None:
            random_sample = np.random.randn(
                number_of_paths, length_of_paths - 1) * np.sqrt(step)
        W[:, 1:] = np.dot(
            random_sample * np.sqrt(step),
            np.triu(np.ones((length_of_paths - 1, length_of_paths - 1)))
        )
        return W
