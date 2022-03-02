from yfinance.ticker import Ticker


def yahoo_ticker_for(asset_name: str) -> Ticker:
    return Ticker(asset_name)


def maturities_for(asset_name: str, ticker: Ticker = None):
    ticker = yahoo_ticker_for(asset_name)
    return ticker.options


def options_for(asset_name: str) -> dict:
    if not asset_name:
        return {}

    ticker = Ticker(asset_name)
    maturities = ticker.options

    if not maturities:
        return {}

    dict = {'calls': {}, 'puts': {}}

    for maturity in maturities:
        dict['calls'][maturity] = ticker.option_chain(maturity).calls
        dict['puts'][maturity] = ticker.option_chain(maturity).puts

    return dict
