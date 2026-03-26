import csv
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

LOG_FILE = "trade_log.csv"

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)


def log_event(action: str, symbol: str, qty=None, note="") -> None:
    file_exists = Path(LOG_FILE).exists()
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp_utc", "action", "symbol", "long_prob", "price", "note"])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            action,
            symbol,
            "",
            "",
            f"qty={qty} {note}".strip(),
        ])


def close_all_positions() -> None:
    positions = trading_client.get_all_positions()

    if not positions:
        print("No open positions to close.")
        return

    for position in positions:
        symbol = position.symbol
        qty = abs(int(float(position.qty)))

        if qty == 0:
            continue

        side = OrderSide.SELL if float(position.qty) > 0 else OrderSide.BUY

        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
        )

        trading_client.submit_order(order_data=order)
        print(f"Closing {symbol} | qty={qty} | side={side}")
        log_event("close", symbol, qty=qty, note=f"side={side}")


if __name__ == "__main__":
    close_all_positions()