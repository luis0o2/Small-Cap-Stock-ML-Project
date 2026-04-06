from __future__ import annotations

import csv
import html
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import requests
from dotenv import load_dotenv
from scipy.sparse import csr_matrix, hstack

from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from src.state import app_state
from alpaca.trading.requests import MarketOrderRequest

TECH_COLS = [
    "ret_1", "ret_5", "ret_10",
    "vol_5", "vol_10", "ma10_dist",
    "mkt_ret_5", "mkt_vol_10", "mkt_ma10_dist",
]

SYMBOLS = [
    "IONQ", "SOUN", "ACHR", "LUNR", "RKLB",
    "PL", "BBAI", "QBTS", "MVST", "ASTS"
]

THRESHOLD = 0.40
SELL_THRESHOLD = 0.20
TOP_N = 2
ORDER_NOTIONAL = 250.0
MIN_CASH_BUFFER = 500.0
MAX_OPEN_POSITIONS = 3

STOP_LOSS_PCT = 0.02
TAKE_PROFIT_PCT = 0.01

AUTO_CLOSE_NEAR_END = True
MARKET_CLOSE_HOUR_UTC = 19
MARKET_CLOSE_MINUTE_UTC = 50

POLL_SECONDS = 10
LOOKBACK_DAYS = 40
NEWS_LIMIT = 10
ALERT_DELTA = 0.05

LOG_FILE = "trade_log.csv"


env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise ValueError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in .env")

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

model = joblib.load("model.pkl")
vectorizer = joblib.load("tfidf.pkl")
scaler = joblib.load("scaler.pkl")

http_session = requests.Session()
http_session.headers.update({
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": SECRET_KEY,
})


KEYWORDS_MAP = {
    "IONQ": ["ionq", "quantum"],
    "SOUN": ["soundhound", "soun", "voice ai"],
    "ACHR": ["archer", "achr", "evtol"],
    "LUNR": ["intuitive machines", "lunr", "moon", "lunar"],
    "RKLB": ["rocket lab", "rklb", "launch", "satellite"],
    "PL": ["planet labs", "pl", "earth data", "satellite imagery"],
    "BBAI": ["bigbear", "bbai", "defense ai"],
    "QBTS": ["d-wave", "qbts", "quantum"],
    "MVST": ["microvast", "mvst", "battery"],
    "ASTS": ["ast spacemobile", "asts", "satellite cellular"],
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def log_event(action: str, symbol: str, long_prob=None, price=None, note: str = "") -> None:
    file_exists = Path(LOG_FILE).exists()
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp_utc", "action", "symbol", "long_prob", "price", "note"])
        writer.writerow([
            utc_now_iso(),
            action,
            symbol,
            "" if long_prob is None else float(long_prob),
            "" if price is None else float(price),
            note,
        ])


def emit_event(event_type: str, **payload: Any) -> None:
    event = {
        "type": event_type,
        "timestamp_utc": utc_now_iso(),
        **payload,
    }
    app_state.add_event(event)
    print(event)


def market_is_open() -> bool:
    return bool(trading_client.get_clock().is_open)


def should_force_close_now() -> bool:
    now = utc_now()
    return (
        AUTO_CLOSE_NEAR_END
        and now.hour == MARKET_CLOSE_HOUR_UTC
        and now.minute >= MARKET_CLOSE_MINUTE_UTC
    )


def get_open_positions_snapshot() -> list[dict[str, Any]]:
    positions_out: list[dict[str, Any]] = []
    try:
        for pos in trading_client.get_all_positions():
            positions_out.append({
                "symbol": pos.symbol,
                "qty": str(pos.qty),
                "side": "long" if float(pos.qty) > 0 else "short",
                "avg_entry_price": str(getattr(pos, "avg_entry_price", "")),
                "market_value": str(getattr(pos, "market_value", "")),
                "unrealized_pl": str(getattr(pos, "unrealized_pl", "")),
            })
    except Exception as e:
        emit_event("error", symbol="SYSTEM", message=f"positions_snapshot_failed: {e}")
    return positions_out


def get_position_map() -> dict[str, dict[str, float | int]]:
    positions: dict[str, dict[str, float | int]] = {}
    for pos in trading_client.get_all_positions():
        positions[pos.symbol] = {
            "qty": int(float(pos.qty)),
            "avg_entry_price": float(pos.avg_entry_price),
            "market_value": float(getattr(pos, "market_value", 0) or 0),
            "unrealized_pl": float(getattr(pos, "unrealized_pl", 0) or 0),
        }
    return positions


def open_position_count(position_map: dict[str, dict[str, float | int]]) -> int:
    return len(position_map)


def get_recent_closes(symbol: str, lookback_days: int = LOOKBACK_DAYS) -> pd.Series:
    end_dt = utc_now()
    start_dt = end_dt - timedelta(days=lookback_days)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start_dt,
        end=end_dt,
        feed=DataFeed.IEX,
    )

    bars = data_client.get_stock_bars(request).df

    if bars.empty:
        raise ValueError(f"No bar data returned for {symbol}")

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level=0)

    closes = bars["close"].astype(float).dropna().sort_index()

    if len(closes) < 11:
        raise ValueError(f"Not enough daily bars for {symbol}. Needed at least 11, got {len(closes)}")

    return closes


def build_numeric_features(
    symbol: str,
    symbol_closes: pd.Series,
    market_closes: pd.Series,
) -> pd.DataFrame:
    closes = symbol_closes

    ret_1 = closes.iloc[-1] / closes.iloc[-2] - 1
    ret_5 = closes.iloc[-1] / closes.iloc[-6] - 1
    ret_10 = closes.iloc[-1] / closes.iloc[-11] - 1

    daily_rets = closes.pct_change().dropna()
    vol_5 = daily_rets.iloc[-5:].std()
    vol_10 = daily_rets.iloc[-10:].std()

    ma10 = closes.iloc[-10:].mean()
    ma10_dist = closes.iloc[-1] / ma10 - 1

    mkt_daily_rets = market_closes.pct_change().dropna()
    mkt_ret_5 = market_closes.iloc[-1] / market_closes.iloc[-6] - 1
    mkt_vol_10 = mkt_daily_rets.iloc[-10:].std()
    mkt_ma10 = market_closes.iloc[-10:].mean()
    mkt_ma10_dist = market_closes.iloc[-1] / mkt_ma10 - 1

    return pd.DataFrame([{
        "ret_1": ret_1,
        "ret_5": ret_5,
        "ret_10": ret_10,
        "vol_5": vol_5,
        "vol_10": vol_10,
        "ma10_dist": ma10_dist,
        "mkt_ret_5": mkt_ret_5,
        "mkt_vol_10": mkt_vol_10,
        "mkt_ma10_dist": mkt_ma10_dist,
    }])


def get_recent_headlines_text(symbol: str, limit: int = NEWS_LIMIT) -> str:
    url = "https://data.alpaca.markets/v1beta1/news"
    params = {
        "symbols": symbol,
        "limit": limit,
    }

    response = http_session.get(url, params=params, timeout=15)
    response.raise_for_status()

    data = response.json()
    news_items = data.get("news", [])

    if not news_items:
        raise ValueError(f"No news returned for {symbol}")

    keywords = KEYWORDS_MAP.get(symbol.upper(), [symbol.lower()])

    usable: list[str] = []
    fallback: list[str] = []

    for item in news_items:
        headline = html.unescape(item.get("headline", "").strip())
        summary = html.unescape(item.get("summary", "").strip())
        text = f"{headline} {summary}".strip()

        if not text:
            continue

        fallback.append(text)
        lowered = text.lower()

        if any(keyword in lowered for keyword in keywords):
            usable.append(text)

    chosen = usable[:5] if usable else fallback[:5]
    return " ".join(chosen)


def get_latest_price(symbol: str) -> float:
    request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
    quote = data_client.get_stock_latest_quote(request)
    return float(quote[symbol].ask_price)

def get_buying_power() -> float:
    account = trading_client.get_account()
    return float(account.buying_power)

def score_symbol(symbol: str, market_closes: pd.Series) -> dict[str, Any] | None:
    symbol_closes = get_recent_closes(symbol)
    raw_numeric_df = build_numeric_features(symbol, symbol_closes, market_closes)
    headline_text = get_recent_headlines_text(symbol, limit=NEWS_LIMIT)

    if not headline_text:
        return None

    x_text = vectorizer.transform([headline_text])
    x_numeric = scaler.transform(raw_numeric_df[TECH_COLS])
    x = hstack([x_text, csr_matrix(x_numeric)]).tocsr()

    pred = model.predict_proba(x)
    long_prob = float(pred[0][1])
    price = get_latest_price(symbol)

    return {
        "symbol": symbol,
        "headline_text": headline_text,
        "long_prob": long_prob,
        "features": raw_numeric_df.iloc[0].to_dict(),
        "price": price,
    }


def place_buy_order(symbol: str, notional: float = ORDER_NOTIONAL) -> None:
    order = MarketOrderRequest(
        symbol=symbol,
        notional=notional,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )
    trading_client.submit_order(order_data=order)


def place_sell_order(symbol: str, qty: int) -> None:
    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )
    trading_client.submit_order(order_data=order)


def should_sell(current_price: float, long_prob: float, avg_entry_price: float) -> str | None:
    if long_prob < SELL_THRESHOLD:
        return "signal_drop"

    if current_price <= avg_entry_price * (1 - STOP_LOSS_PCT):
        return "stop_loss"

    if current_price >= avg_entry_price * (1 + TAKE_PROFIT_PCT):
        return "take_profit"

    return None


class TradingService:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_probs: dict[str, float] = {}
        self._last_headlines: dict[str, str] = {}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        app_state.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        emit_event("service_started", symbol="SYSTEM", message="Trading service started")

    def stop(self) -> None:
        self._stop_event.set()
        app_state.running = False
        emit_event("service_stopped", symbol="SYSTEM", message="Trading service stopped")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._run_cycle()
            except Exception as e:
                app_state.last_error = str(e)
                emit_event("error", symbol="SYSTEM", message=str(e))

            emit_event("sleeping", symbol="SYSTEM", seconds=POLL_SECONDS)
            time.sleep(POLL_SECONDS)

    def _run_cycle(self) -> None:
        is_open = market_is_open()
        app_state.market_open = is_open
        app_state.last_cycle_utc = utc_now_iso()
        app_state.positions = get_open_positions_snapshot()

        emit_event("heartbeat", symbol="SYSTEM", message="cycle_started")

        if not is_open:
            emit_event("market_closed", symbol="SYSTEM", message="Market closed")
            return

        position_map = get_position_map()
        app_state.positions = get_open_positions_snapshot()

        if should_force_close_now():
            for symbol, pos in position_map.items():
                qty = int(pos["qty"])
                if qty <= 0:
                    continue

                current_price = get_latest_price(symbol)
                place_sell_order(symbol, qty)
                log_event("sell", symbol, price=current_price, note="end_of_day")
                emit_event(
                    "sell_submitted",
                    symbol=symbol,
                    price=current_price,
                    qty=qty,
                    reason="end_of_day",
                )
            return

        market_closes = get_recent_closes("SPY")
        signals: list[dict[str, Any]] = []

        for symbol in SYMBOLS:
            try:
                result = score_symbol(symbol, market_closes=market_closes)
                if result is None:
                    continue

                prob = result["long_prob"]
                old_prob = self._last_probs.get(symbol)
                old_headline = self._last_headlines.get(symbol)

                app_state.latest_scores[symbol] = result
                signals.append(result)
                log_event("score", symbol, long_prob=prob, price=result["price"])

                if old_prob is None or abs(prob - old_prob) >= ALERT_DELTA:
                    emit_event(
                        "score_update",
                        symbol=symbol,
                        long_prob=prob,
                        price=result["price"],
                        headline=result["headline_text"][:280],
                    )

                if old_prob is not None and old_prob < THRESHOLD <= prob:
                    emit_event(
                        "threshold_cross_up",
                        symbol=symbol,
                        long_prob=prob,
                        price=result["price"],
                    )

                if old_headline != result["headline_text"]:
                    emit_event(
                        "headline_update",
                        symbol=symbol,
                        headline=result["headline_text"][:280],
                    )

                self._last_probs[symbol] = prob
                self._last_headlines[symbol] = result["headline_text"]

            except Exception as e:
                log_event("error", symbol, note=str(e))
                emit_event("error", symbol=symbol, message=str(e))

        signals.sort(key=lambda x: x["long_prob"], reverse=True)
        top_candidates = [item for item in signals if item["long_prob"] > THRESHOLD][:TOP_N]

        sold_symbols: set[str] = set()

        for symbol, pos in position_map.items():
            matching = next((s for s in signals if s["symbol"] == symbol), None)
            if matching is None:
                continue

            current_price = float(matching["price"])
            long_prob = float(matching["long_prob"])
            avg_entry_price = float(pos["avg_entry_price"])
            qty = int(pos["qty"])

            sell_reason = should_sell(current_price, long_prob, avg_entry_price)
            if sell_reason:
                place_sell_order(symbol, qty)
                sold_symbols.add(symbol)
                log_event("sell", symbol, long_prob=long_prob, price=current_price, note=sell_reason)
                emit_event(
                    "sell_submitted",
                    symbol=symbol,
                    long_prob=long_prob,
                    price=current_price,
                    qty=qty,
                    reason=sell_reason,
                )

        app_state.top_candidates = top_candidates
        emit_event(
            "cycle_complete",
            symbol="SYSTEM",
            scored=len(signals),
            top_candidates=[
                {"symbol": item["symbol"], "long_prob": item["long_prob"]}
                for item in top_candidates
            ],
        )

        current_open_count = open_position_count(position_map)

        for item in top_candidates:
            symbol = item["symbol"]
            prob = item["long_prob"]
            price = item["price"]

            if symbol in sold_symbols:
                emit_event(
                    "buy_skipped",
                    symbol=symbol,
                    long_prob=prob,
                    price=price,
                    reason="sold_this_cycle",
                )
                continue

            if symbol in position_map:
                log_event("skip_already_holding", symbol, long_prob=prob, price=price)
                emit_event(
                    "buy_skipped",
                    symbol=symbol,
                    long_prob=prob,
                    price=price,
                    reason="already_holding",
                )
                continue

            if current_open_count >= MAX_OPEN_POSITIONS:
                log_event("skip_max_positions", symbol, long_prob=prob, price=price)
                emit_event(
                    "buy_skipped",
                    symbol=symbol,
                    long_prob=prob,
                    price=price,
                    reason="max_open_positions_reached",
                )
                continue

            buying_power = get_buying_power()
            if buying_power < ORDER_NOTIONAL + MIN_CASH_BUFFER:
                log_event("skip_low_buying_power", symbol, long_prob=prob, price=price)
                emit_event(
                    "buy_skipped",
                    symbol=symbol,
                    long_prob=prob,
                    price=price,
                    reason="low_buying_power",
                )
                continue

            place_buy_order(symbol, notional=ORDER_NOTIONAL)
            current_open_count += 1

            log_event("buy", symbol, long_prob=prob, price=price)
            emit_event(
                "buy_submitted",
                symbol=symbol,
                long_prob=prob,
                price=price,
                notional=ORDER_NOTIONAL,
            )

            current_open_count += 1
            log_event("buy", symbol, long_prob=prob, price=price)
            emit_event(
                "buy_submitted",
                symbol=symbol,
                long_prob=prob,
                price=price,
                qty=ORDER_QTY,
            )


service = TradingService()


def main() -> None:
    print("Trading service starting...")
    service.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        service.stop()
        print("Trading service stopped.")


if __name__ == "__main__":
    main()