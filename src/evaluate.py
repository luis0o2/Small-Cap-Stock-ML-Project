#EVALUATE.PY
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Force a consistent label order everywhere so metrics are comparable.
# Project labels: -1 = SHORT, 0 = NO_TRADE, 1 = LONG
LABEL_ORDER = [-1, 0, 1]
LABEL_NAMES = ["SHORT(-1)", "NO_TRADE(0)", "LONG(1)"]


def evaluate_classifier(y_true, y_pred, *, title: str = "Validation"):
    """
    Print a confusion matrix + classification report in a consistent way.

    y_true: array-like of true labels in {-1,0,1}
    y_pred: array-like of predicted labels in {-1,0,1}
    """
    # Convert to numpy arrays to avoid pandas index/type surprises
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    # Confusion matrix: rows=true, cols=pred, in fixed label order [-1,0,1]
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)

    print(f"\n=== {title}: Confusion Matrix (rows=true, cols=pred) labels={LABEL_ORDER} ===")
    print(cm)

    # Classification report shows precision/recall/f1 per class.
    # zero_division=0 prevents warnings when a class has no predicted samples.
    print(f"\n=== {title}: Classification Report ===")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=LABEL_ORDER,
            target_names=LABEL_NAMES,
            zero_division=0,
        )
    )

    # Return cm in case you want to use it elsewhere (optional)
    return cm


def actions_from_proba(probs, p_long, p_short, delta=0.10):
    """
    Convert probabilities into trading actions {-1, 0, 1}
    using BOTH absolute confidence and a margin vs NO_TRADE.

    probs columns must be ordered as:
    [P(SHORT), P(NO_TRADE), P(LONG)]
    """
    probs = np.asarray(probs, dtype=float)

    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError(f"Expected probs shape (N,3). Got {probs.shape}")

    p_short_col = probs[:, 0]
    p_no_col    = probs[:, 1]
    p_long_col  = probs[:, 2]

    winner = np.argmax(probs, axis=1)  # 0=SHORT, 1=NO, 2=LONG
    actions = np.zeros(len(probs), dtype=int)

    long_margin  = p_long_col - p_no_col
    short_margin = p_short_col - p_no_col

    long_mask = (
        (winner == 2) &
        (p_long_col >= p_long) &
        (long_margin >= delta)
    )

    short_mask = (
        (winner == 0) &
        (p_short_col >= p_short) &
        (short_margin >= delta)
    )

    actions[long_mask] = 1
    actions[short_mask] = -1
    return actions


def summarize_backtest(actions, future_returns, cost_bps: float = 3.0):
    """
    Simple backtest summary for 1-step forward returns.

    actions: array of {-1,0,1}
    future_returns: array of returns aligned with actions (same length)
                   Example: +0.01 means +1%, -0.02 means -2%
    cost_bps: transaction cost in basis points per trade (action != 0)
              3 bps = 0.0003 in decimal return

    pnl = action * future_return - cost_if_trade
    """
    actions = np.asarray(actions, dtype=int)
    future_returns = np.asarray(future_returns, dtype=float)

    if len(actions) != len(future_returns):
        raise ValueError("actions and future_returns must have the same length.")

    # Convert basis points to decimal (100 bps = 1% = 0.01)
    cost = cost_bps / 10_000.0

    # Trade cost applies only when we trade (action != 0)
    traded = actions != 0

    pnl = actions * future_returns - traded * cost

    # Only keep pnl values for actual trades
    trades = pnl[traded]
    num_trades = int(traded.sum())

    if num_trades == 0:
        return {
            "num_trades": 0,
            "win_rate": 0.0,
            "avg_trade": 0.0,
            "profit_factor": 0.0,
            "total_return": 0.0,
        }

    # ✅ FIX: filter using trades itself (same length), not the full-length 'traded' mask
    wins = trades[trades > 0].sum()
    losses = -trades[trades < 0].sum()  # make losses positive for PF calculation

    # Profit factor = total wins / total losses
    if losses == 0 and wins > 0:
        profit_factor = float("inf")
    elif losses > 0:
        profit_factor = float(wins / losses)
    else:
        profit_factor = 0.0

    win_rate = float((trades > 0).mean())
    avg_trade = float(trades.mean())
    total_return = float(trades.sum())

    return {
        "num_trades": num_trades,
        "win_rate": win_rate,
        "avg_trade": avg_trade,
        "profit_factor": profit_factor,
        "total_return": total_return,
    }
    
def side_breakdown(actions, future_returns, cost_bps: float = 3.0):
    actions = np.asarray(actions, dtype=int)
    r = np.asarray(future_returns, dtype=float)
    cost = cost_bps / 10_000.0

    out = {}
    for side, name in [(1, "LONG"), (-1, "SHORT")]:
        m = actions == side
        if m.sum() == 0:
            out[name] = {"trades": 0, "avg": 0.0, "total": 0.0, "win_rate": 0.0}
            continue
        pnl = side * r[m] - cost
        out[name] = {
            "trades": int(m.sum()),
            "avg": float(pnl.mean()),
            "total": float(pnl.sum()),
            "win_rate": float((pnl > 0).mean()),
        }
    return out


def threshold_sweep(
    probs,
    future_returns=None,
    thresholds=(0.50, 0.55, 0.60, 0.65),
    delta: float = 0.10,
    cost_bps: float = 3.0,
    min_trades: int = 20,   # <-- add
    title: str = "Threshold sweep",
):
    probs = np.asarray(probs, dtype=float)
    print(f"\n=== {title} ===")

    best = None  # (pL, pS, stats)

    for pL in thresholds:
        for pS in thresholds:
            actions = actions_from_proba(probs, p_long=pL, p_short=pS, delta=delta)
            trades = int((actions != 0).sum())

            if future_returns is None:
                print(f"pL={pL:.2f} pS={pS:.2f} trades={trades:4d}")
                continue

            stats = summarize_backtest(actions, future_returns, cost_bps=cost_bps)

            print(
                f"pL={pL:.2f} pS={pS:.2f} trades={stats['num_trades']:4d} "
                f"avg_trade={stats['avg_trade']:+.5f} pf={stats['profit_factor']:.3f} "
                f"total_return={stats['total_return']:+.3f}"
            )

            # only consider configs with enough trades
            if stats["num_trades"] >= min_trades:
                if best is None or stats["avg_trade"] > best[2]["avg_trade"]:
                    best = (pL, pS, stats)

    if best is not None and future_returns is not None:
        pL, pS, stats = best
        print(f"\nBest by avg_trade (min_trades={min_trades}): pL={pL:.2f}, pS={pS:.2f} -> {stats}")

    return best
