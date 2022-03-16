import numpy as np
from data.external_data.maturities import convert_maturity

from data.external_data.yahoo import Yahoo
from data.local_data.parameters_generator import ParametersGenerator
from stochastic_process.markov_chain import MarkovChain, Sigma
from stochastic_process.risky_asset import RiskyAsset


class DataGenerator:
    number_of_S_paths = 1000
    r = 0.05

    def __init__(self, asset: Yahoo, data_size, number_of_states, *args: ParametersGenerator):
        self.asset = asset
        self.data_size = data_size
        self.number_of_states = number_of_states
        self.parameters = [arg.generate(
            data_size, number_of_states) for arg in args]
        self.S_0 = asset.current_asset_price

    def generate_y_train(self):
        axis = 1 if self.data_size > 1 else 0
        return np.concatenate(self.parameters, axis=axis)

    @property
    def number_of_calls(self):
        count = 0
        for maturity in self.asset.maturities:
            for K in self.asset.call_options_data[maturity]['strike']:
                count += 1
        return count

    def generate_x_real(self):
        prices = []
        for maturity in self.asset.maturities:
            for last_price in self.asset.call_options_data[maturity]['lastPrice']:
                prices.append(last_price)
        return np.array(prices).reshape((self.number_of_calls,))

    def generate_x_train(self):
        prices = []
        for sigma, transition_coefficients in zip(*self.parameters):
            X = MarkovChain(*transition_coefficients)
            S = RiskyAsset(self.S_0)

            for maturity in self.asset.maturities:
                T = convert_maturity(maturity)
                length_of_S_paths = int(np.round(T, 3)*1000)
                delta_t = np.linspace(0, T, length_of_S_paths)[1]

                random_sample = np.random.rand(
                    self.number_of_S_paths, length_of_S_paths - 1)

                X_paths = X.paths(random_sample, delta_t)
                sigma_paths = Sigma(*sigma).paths(X_paths)

                S_paths = S.paths(delta_t, sigma_paths, self.r)
                S_T = S_paths[:, -1]

                strikes = self.asset.call_options_data[maturity]['strike'].to_numpy(
                )
                for K in strikes:
                    prices.append(np.exp(-self.r * T) *
                                  np.maximum(S_T - K, 0).mean())

        return np.array(prices).reshape((self.data_size, self.number_of_calls))
