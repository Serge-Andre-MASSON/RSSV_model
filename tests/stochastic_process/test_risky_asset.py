import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
from stochastic_process.risky_asset import RiskyAsset


@pytest.fixture
def risky_asset() -> RiskyAsset:
    Q = np.array([[-5, 2, 3],
                  [4, -7, 3],
                  [2, 1, -3]])
    mu = np.array([0.05, 0.02, 0.07])
    sigma = np.array([0.3, 0.6, 0.8])
    return RiskyAsset(mu, sigma, Q)


def test_risky_asset_path_has_good_shape(risky_asset: RiskyAsset, asset_paths_parameters):
    assert risky_asset.generate_paths(
        *asset_paths_parameters).shape == (asset_paths_parameters[1:])


@pytest.fixture
def asset_paths_parameters():
    T = 1
    number_of_paths = 50
    length_of_paths = 200

    return T, number_of_paths, length_of_paths


@pytest.fixture
def brownian_motion_paths_parameters(asset_paths_parameters):
    T, number_of_paths, length_of_paths = asset_paths_parameters
    step = np.linspace(0, T, length_of_paths)[1]

    return number_of_paths, length_of_paths - 1, step


@pytest.mark.parametrize("X_paths,expected_mu_paths,expected_sigma_paths", [
    (np.array([0, 0, 1, 1, 0, 2]),
     np.array([0.05, 0.05, 0.02, 0.02, 0.05, 0.07]),
     np.array([0.3, 0.3, 0.6, 0.6, 0.3, 0.8])),
    (np.array([[0, 0, 1, 1, 2, 0], [1, 0, 2, 1, 1, 0]]),
     np.array([[0.05, 0.05, 0.02, 0.02, 0.07, 0.05],
              [0.02, 0.05, 0.07, 0.02, 0.02, 0.05]]),
     np.array([[0.3, 0.3, 0.6, 0.6, 0.8, 0.3], [0.6, 0.3, 0.8, 0.6, 0.6, 0.3]]))
])
def test_mu_and_sigma_paths_builder(risky_asset: RiskyAsset, X_paths, expected_mu_paths, expected_sigma_paths):
    mu_paths, sigma_paths = risky_asset.mu_and_sigma_paths(X_paths)
    assert_array_almost_equal(mu_paths, expected_mu_paths)
    assert_array_almost_equal(sigma_paths, expected_sigma_paths)
