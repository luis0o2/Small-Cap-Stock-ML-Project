from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Any


class AppState:
    def __init__(self) -> None:
        self.lock = Lock()
        self.running = False
        self.market_open = False
        self.last_cycle_utc = None
        self.last_error = None

        self.latest_scores: dict[str, dict[str, Any]] = {}
        self.top_candidates: list[dict[str, Any]] = []
        self.positions: list[dict[str, Any]] = []
        self.events = deque(maxlen=500)

    def add_event(self, event: dict[str, Any]) -> None:
        with self.lock:
            self.events.append(event)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "running": self.running,
                "market_open": self.market_open,
                "last_cycle_utc": self.last_cycle_utc,
                "last_error": self.last_error,
                "latest_scores": dict(self.latest_scores),
                "top_candidates": list(self.top_candidates),
                "positions": list(self.positions),
                "recent_events": list(self.events),
            }


app_state = AppState()