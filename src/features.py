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

    out = df.copy()

    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out = out.dropna(subset=["datetime", "ticker", "price_now"]).copy()
    out = out.sort_values(["ticker", "datetime"]).reset_index(drop=True)

    # Compute features WITHOUT groupby.apply
    out["ret_1"]  = out.groupby("ticker")["price_now"].pct_change(1)
    out["ret_5"]  = out.groupby("ticker")["price_now"].pct_change(5)
    out["ret_10"] = out.groupby("ticker")["price_now"].pct_change(10)

    out["vol_5"]  = out.groupby("ticker")["ret_1"].rolling(5).std().reset_index(level=0, drop=True)
    out["vol_10"] = out.groupby("ticker")["ret_1"].rolling(10).std().reset_index(level=0, drop=True)

    ma10 = out.groupby("ticker")["price_now"].rolling(10).mean().reset_index(level=0, drop=True)
    out["ma10_dist"] = (out["price_now"] / ma10) - 1.0

    tech_cols = ["ret_1","ret_5","ret_10","vol_5","vol_10","ma10_dist"]
    out[tech_cols] = out[tech_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out

def add_market_features(df: pd.DataFrame, spy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds market regime features computed from SPY.
    Uses only past/current info.
    """

    m = spy_df.copy().sort_values("datetime")

    p = m["spy_close"].astype(float)

    # 5-day market return
    m["mkt_ret_5"] = p.pct_change(5)

    # 10-day volatility of daily returns
    daily_ret = p.pct_change(1)
    m["mkt_vol_10"] = daily_ret.rolling(10, min_periods=10).std()

    # distance from 10-day MA
    ma10 = p.rolling(10, min_periods=10).mean()
    m["mkt_ma10_dist"] = (p / ma10) - 1.0

    m = m[["datetime", "mkt_ret_5", "mkt_vol_10", "mkt_ma10_dist"]]

    out = df.merge(m, on="datetime", how="left")

    market_cols = ["mkt_ret_5", "mkt_vol_10", "mkt_ma10_dist"]
    out[market_cols] = (
        out[market_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    return out