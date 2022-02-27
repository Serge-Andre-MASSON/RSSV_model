import numpy as np

from stochastic_process import BrownianMotion, ContinuousMarkovChain


class RiskyAsset:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray, Q: np.ndarray):
        self.mu = mu
        self.sigma = sigma
        self.Q = Q

    def generate_paths(self, T, number_of_paths, length_of_paths, X=None):
        step = self.time_step(T, length_of_paths)

        if X is None:
            X = self.hidden_markov_chain_paths(
                number_of_paths, length_of_paths, step)

        mu_paths, sigma_paths = self.mu_and_sigma_paths(X)

        return self.asset_paths_builder(mu_paths, sigma_paths, step)

    def time_step(self, T, length_of_paths):
        return np.linspace(0, T, length_of_paths, retstep=True)[1]

    def hidden_markov_chain_paths(self, number_of_paths, length_of_paths, step):
        X = ContinuousMarkovChain(self.Q)
        X_paths = X.generate_paths(
            number_of_paths, length_of_paths - 1, step)
        return X_paths

    def mu_and_sigma_paths(self, X):
        mu_paths = self.parameter_paths_builder(self.mu)(X)
        sigma_paths = self.parameter_paths_builder(self.sigma)(X)
        return mu_paths, sigma_paths

    def parameter_paths_builder(self, parameter):
        def builder(x: int):
            return parameter[x]
        return np.vectorize(builder)

    def asset_paths_builder(self, mu_paths: np.ndarray, sigma_paths: np.ndarray, step: float):
        r, c = mu_paths.shape
        S = np.ones((r, c+1))
        dW = np.sqrt(step) * np.random.randn(r, c)
        for j in range(1, c + 1):
            S[:, j] = S[:, j - 1] * (1 + mu_paths[:, j - 1] * step +
                                     sigma_paths[:, j - 1] * dW[:, j-1])
        return S
