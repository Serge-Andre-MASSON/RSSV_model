from datetime import datetime
from functools import cache
import re
import pandas as pd
from yfinance.ticker import Ticker


class Yahoo():
    def __init__(self, asset_name: str):
        self.ticker = Ticker(asset_name)
        self.maturities = self.ticker.options
        self.current_asset_price = self.ticker.info['currentPrice']
        self.call_options_data = self.call_options()
        self.put_options_data = self.put_options()

    @cache
    def options_at(self, maturity):
        """Return a dataframe containing options data for this maturity."""
        return self.ticker.option_chain(maturity)

    def call_options(self):
        return {maturity: self.options_at(maturity).calls for maturity in self.maturities}

    def put_options(self):
        return {maturity: self.options_at(maturity).puts for maturity in self.maturities}
