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


# @pytest.fixture
# def call_options_data(msft: Yahoo) -> DataFrame:
#     return msft.call_options_data


# @pytest.fixture
# def put_options_data(msft: Yahoo) -> DataFrame:
#     return msft.put_options_data


# def test_call_options_data(call_options_data):
#     assert isinstance(call_options_data, DataFrame)
#     assert len(call_options_data['contractSymbol'].unique()) > 1
#     for symbol in call_options_data['contractSymbol']:
#         assert 'C' in symbol


# def test_put_options_data(put_options_data):
#     assert isinstance(put_options_data, DataFrame)
#     assert len(put_options_data['contractSymbol'].unique()) > 1
#     for symbol in put_options_data['contractSymbol']:
#         assert 'P' in symbol


# @ pytest.mark.parametrize(
#     "contract_symbol, maturity", [
#         ("MSFT220304C00200000", "2022-03-04"),
#         ("MSFT240119C00470000", "2024-01-19")
#     ]
# )
# def test_contract_symbol_to_maturities(contract_symbol, maturity):
#     assert contract_symbol_to_maturity(contract_symbol) == maturity


# def test_add_maturities():
#     df = DataFrame.from_dict(
#         data={"contractSymbol": ["MSFT220304C00200000", "MSFT240119C00470000"]})
#     df_with_maturities = add_maturities(df)
#     assert 'maturity' in df_with_maturities.columns
#     assert (df_with_maturities['maturity'] == [
#             "2022-03-04", "2024-01-19"]).all()


# def test_add_normalized_maturities():
#     df = DataFrame.from_dict(
#         data={"contractSymbol": ["MSFT220304C00200000", "MSFT220305C00470000"]})
#     df_with_maturities = add_maturities(df)
#     today = '2022-03-02'
#     df_with_normalized_maturity = add_normalized_maturities(
#         df_with_maturities, today)
#     assert 'normalizedMaturity' in df_with_normalized_maturity.columns
#     print(df_with_normalized_maturity)
#     assert (df_with_normalized_maturity['normalizedMaturity'] == [
#             2/365, 3/365]).all()


# @pytest.mark.parametrize('today, maturity_to_normalize, expected_normalized_maturity', [
#     ('2022-03-02', '2022-03-04', 2),
#     ('2022-02-27', '2022-03-04', 5),
#     ('2022-03-03', '2022-03-04', 1)])
# def test_normalize_maturity_when_today_is_given(today, maturity_to_normalize, expected_normalized_maturity):
#     normalized_maturity = normalize_maturity(maturity_to_normalize, today)
#     assert normalized_maturity == expected_normalized_maturity / 365
