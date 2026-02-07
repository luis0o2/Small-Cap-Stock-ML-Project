import pandas as pd


# Columns we expect to exist (label/return names can be customized below)
REQUIRED_TEXT_COL = "headline"
DEFAULT_LABEL_COL = "label"
RETURN_COL_CANDIDATES = ["future_return", "fwd_ret"]


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load the cleaned dataset and do minimal validation/cleaning.
    This module should NOT do feature engineering or modeling.
    """
    df = pd.read_csv(csv_path)

    if REQUIRED_TEXT_COL not in df.columns:
        raise ValueError(f"Missing required column: '{REQUIRED_TEXT_COL}'")

    # Make headlines safe
    df = df.copy()
    df[REQUIRED_TEXT_COL] = df[REQUIRED_TEXT_COL].astype(str).fillna("")

    # If label exists, ensure it's numeric-ish (we won't force it if user wants retrieval-only)
    if DEFAULT_LABEL_COL in df.columns:
        df[DEFAULT_LABEL_COL] = pd.to_numeric(df[DEFAULT_LABEL_COL], errors="coerce")

    # Keep index clean and predictable
    df = df.reset_index(drop=True)
    return df


def pick_return_col(df: pd.DataFrame) -> str | None:
    """
    Choose which return column to use for backtesting.
    Returns the column name if found, otherwise None.
    """
    for c in RETURN_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def drop_unusable_rows(
    df: pd.DataFrame,
    label_col: str = DEFAULT_LABEL_COL,
    require_label: bool = True,
    require_return: bool = False,
) -> pd.DataFrame:
    """
    Remove rows that can't be used given what you plan to run.
    - If training a model, require_label=True
    - If running backtests, require_return=True
    """
    df = df.copy()

    subset = [REQUIRED_TEXT_COL]

    if require_label:
        if label_col not in df.columns:
            raise ValueError(f"Missing label column: '{label_col}'")
        subset.append(label_col)

    if require_return:
        ret_col = pick_return_col(df)
        if ret_col is None:
            raise ValueError(f"No return column found. Tried: {RETURN_COL_CANDIDATES}")
        subset.append(ret_col)

    df = df.dropna(subset=subset).reset_index(drop=True)

    # If labels are required, ensure they're ints (and in expected set)
    if require_label:
        df[label_col] = df[label_col].astype(int)
        bad = set(df[label_col].unique()) - {-1, 0, 1}
        if bad:
            raise ValueError(f"Unexpected label values found: {sorted(bad)} (expected only -1,0,1)")

    return df


def time_train_val_split(df: pd.DataFrame, train_frac: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-aware split (no shuffle). Train is the earliest part, val is the most recent part.
    """
    if not (0.5 < train_frac < 0.95):
        raise ValueError("train_frac should usually be between 0.5 and 0.95 for time splits.")

    n = len(df)
    if n < 100:
        raise ValueError("Dataset too small for a meaningful split.")

    split_idx = int(n * train_frac)
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    val_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    return train_df, val_df