import numpy as np

from src.data import (
    load_dataset,
    drop_unusable_rows,
    pick_return_col,
    time_train_val_test_split,
)
from src.features import build_word_vectorizer
from src.model import train_xgb
from src.evaluate import evaluate_classifier, threshold_sweep, actions_from_proba, summarize_backtest

PATH = "data/cleaned/train_dataset.csv"

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
    X_val   = vectorizer.transform(X_val_text)
    X_test  = vectorizer.transform(X_test_text)

    # 5) Train (use VAL as eval_set for training monitoring only)
    model = train_xgb(X_train, y_train, X_val=X_val, y_val=y_val, use_class_weights=True)

    # 6) Classifier eval (VAL)
    val_preds = model.predict(X_val)
    val_probs = model.predict_proba(X_val)
    evaluate_classifier(y_val, val_preds, title="XGBoost VAL")

    # 7) Pick return column
    ret_col_val = pick_return_col(val_df)
    ret_col_test = pick_return_col(test_df)
    if ret_col_val is None or ret_col_test is None:
        print("\n[Trading eval skipped] No return column found.")
        return

    val_returns  = val_df[ret_col_val].to_numpy(dtype=float)
    test_returns = test_df[ret_col_test].to_numpy(dtype=float)
    
    best_overall = None  # (delta, pL, pS, stats)

    # 8) Threshold selection on VAL ONLY
    for d in (0.00, 0.05, 0.10, 0.15, 0.20):
        best = threshold_sweep(
            val_probs,
            future_returns=val_returns,
            thresholds=(0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65),
            delta=d,
            cost_bps=3.0,
            min_trades=80,
            title=f"VAL threshold sweep (delta={d:.2f})",
        )
        if best is None:
            continue
        pL, pS, stats = best

        if best_overall is None or stats["avg_trade"] > best_overall[3]["avg_trade"]:
            best_overall = (d, pL, pS, stats)


    if best_overall is None:
        print("No config met min_trades on VAL.")
        return

    best_delta, best_pL, best_pS, best_stats = best_overall
    print("\n=== BEST OVERALL ON VAL ===")
    print(f"delta={best_delta:.2f} pL={best_pL:.2f} pS={best_pS:.2f} -> {best_stats}")


    # 9) FINAL report ON TEST (one shot)
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)
    evaluate_classifier(y_test, test_preds, title="XGBoost TEST (pure classifier)")

    test_actions = actions_from_proba(test_probs, p_long=best_pL, p_short=best_pS, delta=best_delta)
    test_stats = summarize_backtest(test_actions, test_returns, cost_bps=3.0)

    print("\n=== FINAL Trading-style TEST report (thresholds chosen on VAL) ===")
    print(f"Using pL={best_pL:.2f}, pS={best_pS:.2f}, delta={best_delta:.2f}")
    print(test_stats)
    
    from src.evaluate import long_only_baseline
    print("\nTEST long-only baseline:")
    print(long_only_baseline(test_returns, cost_bps=3.0))


    from src.evaluate import side_breakdown
    print("\nTEST side breakdown:")
    print(side_breakdown(test_actions, test_returns, cost_bps=3.0))

if __name__ == "__main__":
    main()
