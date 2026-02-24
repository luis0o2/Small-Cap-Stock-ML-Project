# src/main.py
import numpy as np

from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler

from src.data import (
    load_dataset,
    drop_unusable_rows,
    pick_return_col,
    time_train_val_test_split,
)
from src.features import build_word_vectorizer, add_technical_features
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

TECH_COLS = ["ret_1", "ret_5", "ret_10", "vol_5", "vol_10", "ma10_dist"]


def main():
    # 1) Load + clean
    df = load_dataset(PATH)
    df = drop_unusable_rows(df, require_label=True, require_return=True)

    # 2) Add technical features (must happen before splitting so rolling windows work)
    df = add_technical_features(df)

    # 3) Time-aware 3-way split
    train_df, val_df, test_df = time_train_val_test_split(df, train_frac=0.70, val_frac=0.15)

    # 4) Labels
    y_train = train_df["label"].to_numpy(dtype=int)
    y_val = val_df["label"].to_numpy(dtype=int)
    y_test = test_df["label"].to_numpy(dtype=int)

    # 5) TF-IDF (fit train only)
    vectorizer = build_word_vectorizer()
    X_train_txt = vectorizer.fit_transform(train_df["headline"].astype(str))
    X_val_txt = vectorizer.transform(val_df["headline"].astype(str))
    X_test_txt = vectorizer.transform(test_df["headline"].astype(str))

    # 6) Tech numeric features + scaling (fit train only)
    X_train_tech = train_df[TECH_COLS].to_numpy(dtype=float)
    X_val_tech = val_df[TECH_COLS].to_numpy(dtype=float)
    X_test_tech = test_df[TECH_COLS].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_train_tech = scaler.fit_transform(X_train_tech)
    X_val_tech = scaler.transform(X_val_tech)
    X_test_tech = scaler.transform(X_test_tech)

    X_train_tech_sp = csr_matrix(X_train_tech)
    X_val_tech_sp = csr_matrix(X_val_tech)
    X_test_tech_sp = csr_matrix(X_test_tech)

    # 7) Combine text + tech
    X_train = hstack([X_train_txt, X_train_tech_sp]).tocsr()
    X_val = hstack([X_val_txt, X_val_tech_sp]).tocsr()
    X_test = hstack([X_test_txt, X_test_tech_sp]).tocsr()

    print("X_train shape:", X_train.shape, "| TFIDF:", X_train_txt.shape[1], "| TECH:", X_train_tech_sp.shape[1])

    # 8) Train
    model = train_xgb(X_train, y_train, X_val=X_val, y_val=y_val, use_class_weights=True)

    # 9) VAL classifier eval + diagnostics
    val_preds = model.predict(X_val)
    val_probs = model.predict_proba(X_val)

    val_winner = np.argmax(val_probs, axis=1)
    n_long_winner = int((val_winner == 2).sum())
    print("\nVAL: winner==LONG count:", n_long_winner)

    margins = val_probs[:, 2] - val_probs[:, 1]  # p_long - p_no
    print("VAL: margin quantiles (all):", np.quantile(margins, [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
    if n_long_winner > 0:
        print(
            "VAL: margin quantiles (winner==LONG):",
            np.quantile(margins[val_winner == 2], [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]),
        )

    evaluate_classifier(y_val, val_preds, title="XGBoost VAL")

    # 10) Return columns
    ret_col_val = pick_return_col(val_df)
    ret_col_test = pick_return_col(test_df)
    if ret_col_val is None or ret_col_test is None:
        print("\n[Trading eval skipped] No return column found.")
        return

    val_returns = val_df[ret_col_val].to_numpy(dtype=float)
    test_returns = test_df[ret_col_test].to_numpy(dtype=float)

    # 11) Portfolio-growth selection on VAL (long-only)
    p_long_grid = (0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.35, 0.40, 0.45)
    delta_grid = (0.00, 0.05, 0.10, 0.15)

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

    # 12) TEST evaluation (one shot)
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