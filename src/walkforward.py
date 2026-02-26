# src/walkforward.py

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler

from src.features import build_word_vectorizer
from src.model import train_xgb
from src.data import pick_return_col
from src.evaluate import cross_sectional_backtest

TECH_COLS = [
    "ret_1", "ret_5", "ret_10",
    "vol_5", "vol_10", "ma10_dist",
    "mkt_ret_5", "mkt_vol_10", "mkt_ma10_dist"
]


def run_walk_forward(
    df,
    cost_bps=5.0,
    train_size=4000,
    val_size=1000,
    test_size=800,
    step_size=800,
    top_frac=0.15,
):

    df = df.sort_values("datetime").reset_index(drop=True)
    N = len(df)

    results = []
    window = 1
    start = 0

    while start + train_size + val_size + test_size <= N:

        print(f"\n===== WALK-FORWARD WINDOW {window} =====")

        train_df = df.iloc[start : start + train_size]
        val_df   = df.iloc[start + train_size : start + train_size + val_size]
        test_df  = df.iloc[start + train_size + val_size :
                           start + train_size + val_size + test_size]
        print("Columns in test_df:", test_df.columns)
        ret_col = pick_return_col(test_df)
        if ret_col is None:
            print("[skip] No return column found.")
            break

        # ---- Universe diagnostic ----
        avg_universe = test_df.groupby("datetime").size().mean()
        print("Avg rows per day (TEST):", round(avg_universe, 2))
        y_train = train_df["label"].to_numpy(dtype=int)
        y_val   = val_df["label"].to_numpy(dtype=int)

        # ===== FEATURES =====
        vec = build_word_vectorizer()

        X_train_txt = vec.fit_transform(train_df["headline"].astype(str))
        X_val_txt   = vec.transform(val_df["headline"].astype(str))
        X_test_txt  = vec.transform(test_df["headline"].astype(str))

        scaler = StandardScaler()

        X_train_tech = scaler.fit_transform(train_df[TECH_COLS])
        X_val_tech   = scaler.transform(val_df[TECH_COLS])
        X_test_tech  = scaler.transform(test_df[TECH_COLS])

        X_train = hstack([X_train_txt, csr_matrix(X_train_tech)]).tocsr()
        X_val   = hstack([X_val_txt,   csr_matrix(X_val_tech)]).tocsr()
        X_test  = hstack([X_test_txt,  csr_matrix(X_test_tech)]).tocsr()

        # ===== TRAIN =====
        model = train_xgb(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            use_class_weights=True
        )

        # ===== VALIDATION =====
        val_probs = model.predict_proba(X_val)[:, 1]
        val_df_local = val_df.copy()
        val_df_local["pred"] = val_probs
        # 🔥 Aggregate to ONE row per ticker per day
        val_df_local = (
            val_df_local
            .groupby(["datetime", "ticker"], as_index=False)
            .agg({
                "pred": "mean",              # average signal across headlines
                ret_col: "first"             # return is identical anyway
            })
        )

        val_stats = cross_sectional_backtest(
            val_df_local,
            prob_col="pred",
            return_col=ret_col,
            top_frac=top_frac,
        )

        print("VAL stats:", val_stats)

        # ===== TEST =====
        test_probs = model.predict_proba(X_test)[:, 1]
        test_df_local = test_df.copy()
        test_df_local["pred"] = test_probs
        

        # 🔥 Aggregate to ONE row per ticker per day
        test_df_local = (
            test_df_local
            .groupby(["datetime", "ticker"], as_index=False)
            .agg({
                "pred": "mean",              # average signal across headlines
                ret_col: "first"             # return is identical anyway
            })
        )
        print("\n--- SAMPLE TEST ROWS ---")
        print(test_df_local[["datetime", "pred", ret_col]].head(20))
        print("--- END SAMPLE ---\n")
        
        test_stats = cross_sectional_backtest(
            test_df_local,
            prob_col="pred",
            return_col=ret_col,
            top_frac=top_frac,
        )

        print("TEST stats:", test_stats)
        print("Total return:", round(test_stats["total_return"], 4))
        print("Max drawdown:", round(test_stats["max_drawdown"], 4))
        print("Daily Sharpe:", round(test_stats["sharpe_daily"], 3))
        print("Days traded:", test_stats["num_days"])
        print("Strategy daily std:", round(test_stats["strategy_std"], 6))

        results.append(test_stats)

        start += step_size
        window += 1

    print("\n===== WALK-FORWARD SUMMARY =====")

    if not results:
        print("No windows produced.")
        return results

    strategy_avgs = np.array([r["strategy_avg"] for r in results])
    strategy_stds = np.array([r["strategy_std"] for r in results])
    excesses = np.array([r["excess"] for r in results])

    print("windows:", len(results))
    print("mean strategy avg:", strategy_avgs.mean())
    print("mean strategy std:", strategy_stds.mean())
    print("mean excess:", excesses.mean())
    print("positive excess rate:", (excesses > 0).mean())

    return results