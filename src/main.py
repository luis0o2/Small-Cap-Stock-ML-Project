import numpy as np

from src.data import load_dataset, drop_unusable_rows, time_train_val_split, pick_return_col
from src.features import build_word_vectorizer
from src.model import train_xgb
from src.evaluate import evaluate_classifier, threshold_sweep


PATH = "data/cleaned/train_dataset.csv"


def main():
    # 1) Load + minimal cleaning/validation
    df = load_dataset(PATH)

    # require_label=True because we're training a supervised model
    # require_return=True because we want to run the backtest-style sweep
    df = drop_unusable_rows(df, require_label=True, require_return=True)

    # 2) Time-aware split (NO SHUFFLE)
    train_df, val_df = time_train_val_split(df, train_frac=0.8)

    # 3) Separate inputs (X) and targets (y)
    # X is text, y is your labels (-1/0/1)
    X_train_text = train_df["headline"]
    y_train = train_df["label"].to_numpy(dtype=int)

    X_val_text = val_df["headline"]
    y_val = val_df["label"].to_numpy(dtype=int)

    # 4) Build TF-IDF features
    # IMPORTANT: fit on TRAIN only, transform on VAL
    vectorizer = build_word_vectorizer()
    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)

    # 5) Train XGBoost model
    # This returns your wrapper that outputs labels in {-1,0,1}
    model = train_xgb(X_train, y_train, X_val=X_val, y_val=y_val, use_class_weights=True)

    # 6) Predict on validation
    preds = model.predict(X_val)        # -> -1/0/1
    probs = model.predict_proba(X_val)  # -> shape (N,3), columns aligned to [-1,0,1]

    # 7) Classification evaluation
    evaluate_classifier(y_val, preds, title="XGBoost Val")

    # 8) Threshold sweep + backtest-style metrics
    # Pick whichever return column exists: "future_return" or "fwd_ret"
    ret_col = pick_return_col(val_df)
    if ret_col is None:
        print("\n[Threshold sweep skipped] No return column found.")
        return

    future_returns = val_df[ret_col].to_numpy(dtype=float)

    threshold_sweep(probs, future_returns=future_returns,
                thresholds=(0.50,0.55,0.60,0.65),
                delta=0.20,
                title="Threshold sweep (delta=0.20)")



if __name__ == "__main__":
    main()