from datetime import datetime
from functools import cache
import re
from typing import Callable
import pandas as pd
from yfinance.ticker import Ticker

# TODO : Réécrire et simplifier ça. Tout ce dont j'ai besoin c'est d'un array/dataframe contenant T, K, C/P, volume?, impliedVol?


class Yahoo():
    def __init__(self, asset_name: str):
        self.ticker = Ticker(asset_name)

    @property
    def current_asset_price(self):
        return self.ticker.info['currentPrice']

    @property
    def maturities(self):
        return self.ticker.options

    def options(self, option_type="call"):
        if not option_type in ["call", "put"]:
            raise ValueError("option_type must be 'call' or 'put'")

        maturities = self.maturities
        options_df: pd.DataFrame = None

        for maturity in maturities:
            if options_df is None:
                options_df = self.calls(
                    maturity) if option_type == "call" else self.puts(maturity)
                options_df['maturity'] = maturity
            else:
                options = self.calls(
                    maturity) if option_type == "call" else self.puts(maturity)
                options['maturity'] = maturity
                options_df = pd.concat(
                    [options_df, options], ignore_index=True)
        return options_df

    def calls(self, maturity):
        return self.ticker.option_chain(maturity).calls

    def puts(self, maturity):
        return self.ticker.option_chain(maturity).puts
