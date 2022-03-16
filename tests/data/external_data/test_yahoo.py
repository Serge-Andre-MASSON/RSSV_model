import pytest
from pandas import DataFrame
from data.external_data.yahoo import Yahoo


@pytest.fixture
def msft() -> Yahoo:
    return Yahoo('MSFT')


@pytest.mark.skip(reason="Access external data : too long!")
def test_maturities_is_a_tuple_of_str(msft: Yahoo):
    msft_maturities = msft.maturities
    assert type(msft_maturities) is tuple
    for maturity in msft_maturities:
        assert type(maturity) is str
