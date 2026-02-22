# src/main.py
import numpy as np

from src.data import (
    load_dataset,
    drop_unusable_rows,
    pick_return_col,
    time_train_val_test_split,
)
from src.features import build_word_vectorizer
from src.model import train_xgb
from src.evaluate import (
    evaluate_classifier,
    threshold_sweep,
    actions_from_proba,
    summarize_backtest,
    long_only_baseline,
    side_breakdown,
)

PATH = "data/cleaned/train_dataset.csv"
COST_BPS = 3.0

def main():
    # 1) Load + clean
    df = load_dataset(PATH)
    df = drop_unusable_rows(df, require_label=True, require_return=True)

    # 2) Time-aware 3-way split
    train_df, val_df, test_df = time_train_val_test_split(df, train_frac=0.70, val_frac=0.15)

    # 3) Text/labels
    X_train_text = train_df["headline"].astype(str)
    y_train = train_df["label"].to_numpy(dtype=int)

    X_val_text = val_df["headline"].astype(str)
    y_val = val_df["label"].to_numpy(dtype=int)

    X_test_text = test_df["headline"].astype(str)
    y_test = test_df["label"].to_numpy(dtype=int)

    # 4) TF-IDF (fit train only)
    vectorizer = build_word_vectorizer()
    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)
    X_test = vectorizer.transform(X_test_text)

    # 5) Train
    model = train_xgb(X_train, y_train, X_val=X_val, y_val=y_val, use_class_weights=True)

    # 6) VAL classifier eval + diagnostics
    val_preds = model.predict(X_val)
    val_probs = model.predict_proba(X_val)

    val_winner = np.argmax(val_probs, axis=1)
    n_long_winner = int((val_winner == 2).sum())
    print("\nVAL: winner==LONG count:", n_long_winner)

    margins = val_probs[:, 2] - val_probs[:, 1]  # p_long - p_no
    print("VAL: margin quantiles (all):", np.quantile(margins, [0, .25, .5, .75, .9, .95, .99]))
    if n_long_winner > 0:
        print(
            "VAL: margin quantiles (winner==LONG):",
            np.quantile(margins[val_winner == 2], [0, .25, .5, .75, .9, .95, .99]),
        )

    evaluate_classifier(y_val, val_preds, title="XGBoost VAL")

    # 7) Return columns
    ret_col_val = pick_return_col(val_df)
    ret_col_test = pick_return_col(test_df)
    if ret_col_val is None or ret_col_test is None:
        print("\n[Trading eval skipped] No return column found.")
        return

    val_returns = val_df[ret_col_val].to_numpy(dtype=float)
    test_returns = test_df[ret_col_test].to_numpy(dtype=float)

    # 8) Portfolio-growth selection on VAL (long-only)
    # Objective: maximize total_return / N (avg return per opportunity)
    p_long_grid = (0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30)
    delta_grid = (0.00, 0.05, 0.10, 0.15)

    # Coverage floor instead of a fixed min_trades that can make selection impossible
    MIN_COVERAGE = 0.02  # 2%
    min_trades = max(1, int(MIN_COVERAGE * len(val_returns)))

    best_overall = None  # (delta, pL, stats, val_avg_per_opportunity)

    for d in delta_grid:
        best = threshold_sweep(
            val_probs,
            future_returns=val_returns,
            p_long_thresholds=p_long_grid,
            delta=d,
            cost_bps=COST_BPS,
            min_trades=min_trades,
            title=f"VAL long-only sweep (delta={d:.2f}, min_trades={min_trades})",
        )
        if best is None:
            continue

        pL, stats = best
        val_avg_per_opportunity = stats["total_return"] / len(val_returns)

        if best_overall is None or val_avg_per_opportunity > best_overall[3]:
            best_overall = (d, pL, stats, val_avg_per_opportunity)

    if best_overall is None:
        print("No config met min_trades on VAL even after loosening.")
        return

    best_delta, best_pL, best_stats, best_val_avg = best_overall
    print("\n=== BEST OVERALL ON VAL (portfolio growth objective, long-only) ===")
    print(f"delta={best_delta:.2f} pL={best_pL:.2f}")
    print("VAL avg_return_per_opportunity:", best_val_avg)
    print("VAL stats:", best_stats)

    # 9) TEST evaluation (one shot)
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)
    evaluate_classifier(y_test, test_preds, title="XGBoost TEST (pure classifier)")

    test_actions = actions_from_proba(test_probs, p_long=best_pL, delta=best_delta)
    test_stats = summarize_backtest(test_actions, test_returns, cost_bps=COST_BPS)
    test_avg_per_opportunity = test_stats["total_return"] / len(test_returns)
    test_coverage = test_stats["num_trades"] / len(test_returns)

    print("\n=== FINAL Trading-style TEST report (thresholds chosen on VAL) ===")
    print(f"Using pL={best_pL:.2f}, delta={best_delta:.2f}")
    print(test_stats)
    print("coverage:", test_coverage)
    print("TEST avg_return_per_opportunity:", test_avg_per_opportunity)

    print("\nTEST long-only baseline:")
    baseline = long_only_baseline(test_returns, cost_bps=COST_BPS)
    print(baseline)
    print("Baseline avg_return_per_opportunity:", baseline["total_return"] / len(test_returns))

    print("\nTEST side breakdown:")
    print(side_breakdown(test_actions, test_returns, cost_bps=COST_BPS))


if __name__ == "__main__":
    main()