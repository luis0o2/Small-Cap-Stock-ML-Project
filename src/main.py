# src/main.py

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler

from src.spy import load_spy
from src.data import (
    load_dataset,
    drop_unusable_rows,
    time_train_val_test_split,
)
from src.features import (
    build_word_vectorizer,
    add_technical_features,
    add_market_features,
)
from src.model import train_xgb
from src.evaluate import evaluate_classifier
from src.walkforward import run_walk_forward


PATH = "data/cleaned/train_dataset.csv"
COST_BPS = 30.0

TECH_COLS = [
    "ret_1", "ret_5", "ret_10",
    "vol_5", "vol_10", "ma10_dist",
    "mkt_ret_5", "mkt_vol_10", "mkt_ma10_dist"
]


def main():

    # ==========================================================
    # 1) LOAD + CLEAN DATA
    # ==========================================================
    df = load_dataset(PATH)
    df = drop_unusable_rows(df, require_label=True, require_return=True)

    # Convert to binary: 1 = LONG, 0 = NOT_LONG
    df["label"] = (df["label"] == 1).astype(int)

    # Clip extreme returns to reduce tail distortion
    CLIP_LEVEL = 0.30
    #df["future_return"] = df["future_return"].clip(-CLIP_LEVEL, CLIP_LEVEL)
    
    print("\nCOLUMNS IN DATASET:")
    print(df.columns)
    # ==========================================================
    # 2) ADD TECHNICAL FEATURES
    # ==========================================================
    df = add_technical_features(df)

    # ==========================================================
    # 3) ADD MARKET FEATURES (SPY)
    # ==========================================================
    start_date = df["datetime"].min().strftime("%Y-%m-%d")
    end_date = df["datetime"].max().strftime("%Y-%m-%d")

    spy_df = load_spy(start_date, end_date)
    df = add_market_features(df, spy_df)

    # ==========================================================
    # 4) TIME-AWARE SPLIT (FOR DIAGNOSTICS ONLY)
    # ==========================================================
    train_df, val_df, test_df = time_train_val_test_split(
        df,
        train_frac=0.70,
        val_frac=0.15
    )

    y_train = train_df["label"].to_numpy(dtype=int)
    y_val = val_df["label"].to_numpy(dtype=int)

    # ==========================================================
    # 5) FEATURE ENGINEERING
    # ==========================================================
    vectorizer = build_word_vectorizer()

    X_train_txt = vectorizer.fit_transform(train_df["headline"].astype(str))
    X_val_txt = vectorizer.transform(val_df["headline"].astype(str))

    scaler = StandardScaler()

    X_train_tech = scaler.fit_transform(train_df[TECH_COLS])
    X_val_tech = scaler.transform(val_df[TECH_COLS])

    X_train = hstack([X_train_txt, csr_matrix(X_train_tech)]).tocsr()
    X_val = hstack([X_val_txt, csr_matrix(X_val_tech)]).tocsr()

    print("X_train shape:", X_train.shape,
          "| TFIDF:", X_train_txt.shape[1],
          "| NUMERIC:", len(TECH_COLS))

    # ==========================================================
    # 6) TRAIN MODEL
    # ==========================================================
    model = train_xgb(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        use_class_weights=True
    )
    import joblib
    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "tfidf.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("Model, vectorizer, and scaler saved!")

    # ==========================================================
    # 7) VALIDATION DIAGNOSTICS (CLASSIFICATION ONLY)
    # ==========================================================
    val_preds = model.predict(X_val)
    val_probs = model.predict_proba(X_val)

    evaluate_classifier(y_val, val_preds, title="XGBoost VAL")

    print("Mean predicted LONG prob (VAL):", val_probs[:, 1].mean())
    print("Std predicted LONG prob (VAL):", val_probs[:, 1].std())

    # ==========================================================
    # 8) WALK-FORWARD CROSS-SECTIONAL TEST
    # ==========================================================
    print("\nRUNNING WALK-FORWARD CROSS-SECTIONAL TEST...")

    df_sorted = df.sort_values("datetime").reset_index(drop=True)

    run_walk_forward(
        df_sorted,
        cost_bps=COST_BPS,
        train_size=4500,
        val_size=1500,
        test_size=800,
        step_size=800,
        top_frac=0.05,  # long top 15% each day
    )

if __name__ == "__main__":
    main()