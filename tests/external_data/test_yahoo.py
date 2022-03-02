from pandas import DataFrame, options
import pytest
from yfinance.ticker import Ticker
from external_data.yahoo import maturities_for, options_for, yahoo_ticker_for


def test_ticker_for_a_given_string_return_a_ticker_object(asset_name):
    ticker = yahoo_ticker_for(asset_name)
    assert isinstance(ticker, Ticker)


def test_maturities_for_a_given_string_return_a_tuple_of_dates(asset_name):
    maturities = maturities_for(asset_name)
    for maturity in maturities:
        assert isinstance(maturity, str)


def test_that_options_for_return_an_empty_dict_for_an_empty_string():
    assert options_for('') == {}


def test_that_options_returning_an_empty_dict_if_the_asset_does_not_exist():
    assert options_for('A non existing asset') == {}


def test_that_options_for_returns_a_dict_containing_calls_and_puts(asset_name):
    options = options_for(asset_name)
    assert 'calls' in options
    assert 'puts' in options
    for calls_maturity, puts_maturity in zip(options['calls'], options['puts']):
        assert isinstance(options['calls'][calls_maturity], DataFrame)
        assert isinstance(options['puts'][puts_maturity], DataFrame)


@pytest.fixture
def asset_name():
    return 'MSFT'
