import numpy as np


class ContinuousMarkovChain:
    # TODO: Change this docstring.
    """Implement a continous time Markov chain.
    """

    def __init__(self, Q: np.ndarray) -> None:
        self.Q = Q

    def P(self, step):
        return np.eye(*self.Q.shape) + step * self.Q

    def g_paths(self, number_of_paths: int, length_of_paths: int, step: float, random_sample: np.ndarray = None):
        M = self.P(step)
        aggregate_P = agregate_matrix(M)
        paths = np.zeros((number_of_paths, length_of_paths))
        return paths

    def generate_paths(self, number_of_paths: int, length_of_paths: int, step: float, random_sample: np.ndarray = None):
        """Generate a random matrix where each line is a randomly generated path of the Markov chain for the time step provided.

        Args:
            number_of_paths (int): Number of paths to simulate.
            length_of_paths (int): Length of eqch path.
            step (float): The step used for the simulation.
            random_sample (np.ndarray, optional): This argument is here for test purposes and defaults to None. It can be use to provide an external random sample.

        Returns:
            _type_: A matrix containing all paths generated where each path is a row.
        """
        if random_sample is None:
            random_sample = np.random.rand(
                number_of_paths, length_of_paths - 1)

        paths = np.zeros((number_of_paths, length_of_paths), dtype=int)

        A = agregate_matrix(self.P(step))

        for j in range(length_of_paths - 1):
            X = paths[:, j]
            paths[:, j+1] = self.next_values(X, random_sample[:, j], A)

        return paths

    def next_values(self, X, random_sample, A):
        next_value_of_X = np.zeros_like(X)

        for i, x, r in zip(range(len(X)), X, random_sample):
            mask: np.ndarray = r < A[x, :]
            next_value_of_X[i] = mask.argmax(axis=0)

        return np.array(next_value_of_X)


def agregate_matrix(M):
    """Return a matrix where each columns j is the sum of columns 0 to j of the provided matrix."""
    return np.dot(M, np.triu(np.ones_like(M)))
