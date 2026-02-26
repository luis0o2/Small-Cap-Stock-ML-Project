#EVALUATE.PY
import numpy as np
import pandas as pd

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

def actions_from_proba(probs, p_long, delta=None, p_short=None):
    # probs shape: (n, 2)
    p_not_long = probs[:, 0]
    p_long_prob = probs[:, 1]

    actions = np.zeros(len(probs))

    long_mask = (p_long_prob >= p_long) & ((p_long_prob - p_not_long) > delta)
    actions[long_mask] = 1

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
    
    


def long_only_baseline(future_returns, cost_bps: float = 3.0):
    r = np.asarray(future_returns, dtype=float)
    cost = cost_bps / 10_000.0
    pnl = r - cost  # always long, always pay cost

    wins = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()

    pf = float("inf") if (losses == 0 and wins > 0) else (float(wins / losses) if losses > 0 else 0.0)

    return {
        "num_trades": int(len(r)),
        "win_rate": float((pnl > 0).mean()),
        "avg_trade": float(pnl.mean()),
        "profit_factor": pf,
        "total_return": float(pnl.sum()),
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
    p_long_thresholds=(0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65),
    delta: float = 0.10,
    cost_bps: float = 3.0,
    min_trades: int = 80,
    title: str = "Long-only threshold sweep",
):
    probs = np.asarray(probs, dtype=float)
    print(f"\n=== {title} ===")

    best = None  # (pL, stats)

    for pL in p_long_thresholds:
        actions = actions_from_proba(probs, p_long=pL, delta=delta)
        trades = int((actions != 0).sum())

        if future_returns is None:
            print(f"pL={pL:.2f} trades={trades:4d}")
            continue

        stats = summarize_backtest(actions, future_returns, cost_bps=cost_bps)

        print(
            f"pL={pL:.2f} trades={stats['num_trades']:4d} "
            f"avg_trade={stats['avg_trade']:+.5f} pf={stats['profit_factor']:.3f} "
            f"total_return={stats['total_return']:+.3f}"
        )

        if stats["num_trades"] >= min_trades:
            if best is None or stats["avg_trade"] > best[1]["avg_trade"]:
                best = (pL, stats)
    

    if best is not None and future_returns is not None:
        pL, stats = best
        print(f"\nBest by avg_trade (min_trades={min_trades}): pL={pL:.2f} -> {stats}")

    return best

def cross_sectional_backtest(
    df: pd.DataFrame,
    prob_col: str,
    return_col: str,
    top_frac: float = 0.15,
    max_position_size: float = 0.10,   # 10% capital per stock
    slippage: float = 0.01,            # 1% slippage
    cost_bps: float = 30.0,            # 30 bps transaction cost
):
    """
    Realistic cross-sectional long-only simulator.

    - Caps position size per stock
    - Applies slippage + transaction cost
    - Compounds capital realistically
    """

    strategy_daily = []
    baseline_daily = []

    capital = 1.0
    equity_curve = []

    cost = cost_bps / 10_000.0

    for date, group in df.groupby("datetime"):

        if len(group) < 5:
            continue

        k = max(1, int(top_frac * len(group)))

        top = group.sort_values(prob_col, ascending=False).head(k)

        # Equal weight within top picks, but capped at max_position_size
        weight_per_stock = min(max_position_size, 1.0 / k)

        daily_portfolio_return = 0.0

        for r in top[return_col]:
            # Apply slippage + cost
            r_adj = r - slippage - cost
            daily_portfolio_return += weight_per_stock * r_adj

        # Capital update
        capital *= (1.0 + daily_portfolio_return)
        equity_curve.append(capital)

        strategy_daily.append(daily_portfolio_return)
        baseline_daily.append(group[return_col].mean())

    strategy_daily = np.array(strategy_daily)
    baseline_daily = np.array(baseline_daily)
    equity_curve = np.array(equity_curve)

    # === Metrics ===
    total_return = equity_curve[-1] - 1.0 if len(equity_curve) > 0 else 0.0

    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    max_dd = drawdown.min() if len(drawdown) > 0 else 0.0

    sharpe = (
        strategy_daily.mean() / strategy_daily.std()
        if strategy_daily.std() > 0 else 0.0
    )

    return {
        "strategy_avg": strategy_daily.mean(),
        "strategy_std": strategy_daily.std(),
        "baseline_avg": baseline_daily.mean(),
        "excess": strategy_daily.mean() - baseline_daily.mean(),
        "positive_rate": (strategy_daily > baseline_daily).mean(),
        "num_days": len(strategy_daily),
        "total_return": total_return,
        "max_drawdown": max_dd,
        "sharpe_daily": sharpe,
    }