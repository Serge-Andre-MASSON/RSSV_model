import numpy as np


class MarkovChain:
    """Implement a continuous time markov chain with any number of states.
    """

    def __init__(self, *args) -> None:
        """Take the non diagonal terms of an intensity matrix Q, build Q and deduce the number of states from the size of Q. 
        """
        self.Q = Q_from_coefficients(args)
        self.number_of_states = len(args)

    def P(self, delta_t):
        return np.eye(len(self.Q)) + self.Q * delta_t

    def A(self, delta_t):
        P = self.P(delta_t)
        A = P @ np.triu(np.ones_like(P))
        return A

    def paths(self, random_sample: np.ndarray, delta_t):
        A = self.A(delta_t)
        r, c = random_sample.shape
        X_paths = np.zeros((r, c + 1), dtype=int)
        for j in range(c):
            current_values_of_X = X_paths[:, j]
            A_corresponding_rows = A[current_values_of_X]

            current_random_sample = random_sample[:, j]
            masked_A_corresponding_rows = mask(
                current_random_sample, A_corresponding_rows)

            X_paths[:, j + 1] = next_value(masked_A_corresponding_rows)
        return X_paths

    def next_values(self, M):
        pass


class Sigma:
    def __init__(self, *args) -> None:
        self.states = np.array(args)

    def paths(self, i):
        return self.states[i]


def agregate_matrix(M):
    """Return a matrix where each columns j is the sum of columns 0 to j of the provided matrix."""
    return np.dot(M, np.triu(np.ones_like(M)))


def Q_from_coefficients(coefficients):
    l = len(coefficients)
    n = 0
    while n**2 - n != l:
        n += 1
        if n > l:
            print(l)
            raise ValueError(
                "Le nombre de coefficients doit être de la forme n(n-1)")
    Q = np.zeros((n, n))
    l_index = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                Q[i, j] = coefficients[l_index]
                l_index += 1

    for i in range(len(Q)):
        Q[i, i] = - Q[i].sum()
    return Q


def mask(r, M):
    mask = r < np.transpose(M)
    return np.transpose(mask)


def next_value(mask):
    return np.argmax(mask, axis=1)
