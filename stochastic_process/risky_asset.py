import numpy as np


class RiskyAsset:
    def __init__(self, current_price: float) -> None:
        self.current_price = current_price

    def paths(self, delta_t: float, sigma: np.ndarray, r):
        rows, cols = sigma.shape

        S_paths = self.current_price + np.zeros((rows, cols + 1))
        r_delta_t = r * delta_t
        sqrt_delta_t = np.sqrt(delta_t)
        for j in range(cols):
            S_paths[:, j+1] = S_paths[:, j] * \
                (1 + r_delta_t + sqrt_delta_t *
                 np.random.randn(rows) * sigma[:, j])

        return S_paths
