from datetime import datetime
import pickle

import pytest

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


def test_convert_maturity_when_maturity_is_in_the_past():
    maturity = "2022-04-14"
    today = datetime(2022, 4, 13, 5, 34, 3, 657292)


def test_maturities_when_today_is_less_than_one_day_before_maturity():
    maturity = "2022-04-14"
    now = datetime(2022, 4, 13, 5, 34, 3, 657292)

    with pytest.raises(ValueError, match="Maturity is less than one day from now."):
        convert_maturity(maturity, "2022-04-14")
