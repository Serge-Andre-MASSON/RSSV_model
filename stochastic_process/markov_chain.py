import numpy as np


# TODO: Consider the case with more than two states
class ContinuousMarkovChain:
    """Implement a continous time Markov chain with two states.
    """

    def __init__(self, q_01: float, q_10: float) -> None:
        """Ask for the non_diagnoal terms of the continuous Markov chain's transition matrix.

        Args:
            q_01 (float): Transition rate from state zero to one
            q_10 (float): Transition rate from state one to zero
        """
        self.q_01 = q_01
        self.q_10 = q_10

    def generate_paths(self, number_of_paths: int, length_of_paths: int, step: float, random_sample: np.ndarray = None):
        """Generate a random matrix where each line is a randomly generated path of the Markov chain for the time step provided.

        Args:
            number_of_paths (int): Number of paths to simulate.
            length_of_paths (int): Length of eqch path.
            step (float): The time step (dt) used for the simulation.
            random_sample (np.ndarray, optional): This argument is here for test purposes and defaults to None. It can be use to provide an external random sample.

        Returns:
            _type_: A matrix containing all paths generated where each path is a row.
        """
        if random_sample is None:
            random_sample = np.random.rand(
                number_of_paths, length_of_paths - 1)
        paths = np.zeros((number_of_paths, length_of_paths))

        for i in range(number_of_paths):
            paths[i] = self.generate_one_path(
                length_of_paths, step, random_sample[i])
        return paths

    def next_value_of_X(self, x, r, step):
        """Return the next_value of X, depends on wether X is currently one or zero"""
        if x == 0:
            next_x = 1 if r < self.q_01 * step else 0
        else:
            next_x = 1 if r < self.q_10 * step else 0
        return next_x

    def generate_one_path(self, length_of_path, step, random_sample=None):
        X = np.zeros(length_of_path)

        for j in range(length_of_path - 1):
            X[j + 1] = self.next_value_of_X(X[j], random_sample[j], step)
        return X
