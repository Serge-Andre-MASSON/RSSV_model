from yfinance.ticker import Ticker


class Yahoo():
    def __init__(self, asset_name):
        self.option_data = Ticker(asset_name).option_chain()
        self.call_options_data = self.option_data.calls
        self.put_options_data = self.option_data.puts
