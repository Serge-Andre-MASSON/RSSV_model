import pickle

from data.external_data.yahoo import Yahoo
from data.external_data.maturities import convert_maturity

with open('tests/data/external_data/ora.pickle', 'rb') as f:
    ora: Yahoo = pickle.load(f)


def test_maturity_convert_within_a_month():
    today = '2022-03-12'
    maturity = '2022-03-16'
    assert convert_maturity(maturity, today) == 4 / 365


def test_maturity_convert_within_two_months():
    today = '2022-03-12'
    maturity = '2022-04-16'
    assert convert_maturity(maturity, today) == (16 + 31 - 12) / 365
