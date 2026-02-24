# src/features.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def build_word_vectorizer():
    return TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
        min_df=5,
    )

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds per-ticker technical features computed from price_now using only past/current info.
    Assumes df has columns: datetime, ticker, price_now
    Returns a COPY with new numeric feature columns.
    """
    out = df.copy()

    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out = out.dropna(subset=["datetime", "ticker", "price_now"]).copy()
    out = out.sort_values(["ticker", "datetime"]).reset_index(drop=True)

    def _per_ticker(g: pd.DataFrame) -> pd.DataFrame:
        p = g["price_now"].astype(float)

        # returns (uses current + past only)
        g["ret_1"]  = p.pct_change(1)
        g["ret_5"]  = p.pct_change(5)
        g["ret_10"] = p.pct_change(10)

        # rolling volatility of daily returns
        r1 = g["ret_1"]
        g["vol_5"]  = r1.rolling(5, min_periods=5).std()
        g["vol_10"] = r1.rolling(10, min_periods=10).std()

        # moving average distance
        ma_10 = p.rolling(10, min_periods=10).mean()
        g["ma10_dist"] = (p / ma_10) - 1.0

        return g

    out = out.groupby("ticker", group_keys=False).apply(_per_ticker)

    # Replace early NaNs (not enough history) with 0.0
    tech_cols = ["ret_1","ret_5","ret_10","vol_5","vol_10","ma10_dist"]
    out[tech_cols] = out[tech_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out