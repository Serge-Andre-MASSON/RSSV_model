from pandas import DataFrame
from data.external_data.yahoo import Yahoo


msft = Yahoo('MSFT')
call_options_df = msft.call_options_data
put_options_df = msft.put_options_data


def test_call_and_put_options_data_are_dataframe():
    assert isinstance(call_options_df, DataFrame)
    assert isinstance(put_options_df, DataFrame)
    for symbol in call_options_df['contractSymbol']:
        assert 'C' in symbol
    for symbol in put_options_df['contractSymbol']:
        assert 'P' in symbol
