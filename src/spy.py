import yfinance as yf
import pandas as pd

def load_spy(start_date: str, end_date: str = None):
    spy = yf.download("SPY", start=start_date, end=end_date, progress=False)
    spy = spy.reset_index()
    spy = spy[["Date", "Close"]]
    spy.columns = ["datetime", "spy_close"]
    spy["datetime"] = pd.to_datetime(spy["datetime"])
    return spy