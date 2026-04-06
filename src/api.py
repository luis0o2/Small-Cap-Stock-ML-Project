from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from src.live_trading import service
from src.state import app_state


app = FastAPI(title="Stock ML Backend", version="1.0.0")


class ConnectionManager:
    def __init__(self) -> None:
        self.connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.connections:
            self.connections.remove(websocket)

    async def broadcast(self, message: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        for connection in self.connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                dead.append(connection)

        for connection in dead:
            self.disconnect(connection)


manager = ConnectionManager()


@app.on_event("startup")
async def startup_event() -> None:
    service.start()
    asyncio.create_task(broadcast_loop())


async def broadcast_loop() -> None:
    last_sent_count = 0

    while True:
        snapshot = app_state.snapshot()
        events = snapshot["recent_events"]

        if len(events) != last_sent_count:
            new_events = events[last_sent_count:]
            for event in new_events:
                await manager.broadcast(event)
            last_sent_count = len(events)

        await asyncio.sleep(1)


@app.get("/status")
def get_status():
    return app_state.snapshot()


@app.get("/scores")
def get_scores():
    snapshot = app_state.snapshot()
    return {
        "latest_scores": snapshot["latest_scores"],
        "top_candidates": snapshot["top_candidates"],
    }


@app.get("/positions")
def get_positions():
    return {"positions": app_state.snapshot()["positions"]}


@app.post("/start")
def start_service():
    service.start()
    return {"ok": True, "message": "service started"}


@app.post("/stop")
def stop_service():
    service.stop()
    return {"ok": True, "message": "service stopped"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await websocket.send_json({
            "type": "hello",
            "message": "connected",
            "snapshot": app_state.snapshot(),
        })

        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

        @app.get("/summary")
        def get_summary():
            snapshot = app_state.snapshot()
            return {
                "running": snapshot["running"],
                "market_open": snapshot["market_open"],
                "last_cycle_utc": snapshot["last_cycle_utc"],
                "top_candidates": snapshot["top_candidates"],
                "positions": snapshot["positions"],
                "last_error": snapshot["last_error"],
            }