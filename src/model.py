#MODELS.PY
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
from xgboost import XGBClassifier

try:
# Optional: only used if you enable class balancing
    from sklearn.utils.class_weight import compute_class_weight
except Exception:
    compute_class_weight = None




@dataclass
class TradingXGBModel:
    model: XGBClassifier

    def predict(self, X) -> np.ndarray:
        return (self.model.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

def make_sample_weight_balanced(y_train_xgb: np.ndarray) -> np.ndarray:
    """
    Compute balanced sample weights for classes {0,1,2}.
    """
    if compute_class_weight is None:
        raise ImportError("sklearn is required for compute_class_weight")

    classes_idx = np.array([0, 1], dtype=int)
    class_w = compute_class_weight(
        class_weight="balanced",
        classes=classes_idx,
        y=y_train_xgb
    )
    w_map = {c: w for c, w in zip(classes_idx, class_w)}
    return np.array([w_map[c] for c in y_train_xgb], dtype=float)


def build_xgb_classifier(random_state: int = 88) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",

        # 🔽 Stronger regularization
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,

        subsample=0.7,
        colsample_bytree=0.7,

        min_child_weight=6,
        reg_lambda=3.0,
        reg_alpha=1.0,

        gamma=1.0,   # require stronger splits

        tree_method="hist",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=random_state,
    )
    
def train_xgb(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    use_class_weights: bool = True,
    random_state: int = 88,
) -> TradingXGBModel:

    # Binary labels: 0 / 1
    y_train_xgb = np.asarray(y_train).astype(int)

    sample_weight = None
    if use_class_weights:
        sample_weight = make_sample_weight_balanced(y_train_xgb)

    model = build_xgb_classifier(random_state=random_state)

    eval_set = None
    if X_val is not None and y_val is not None:
        y_val_xgb = np.asarray(y_val).astype(int)
        eval_set = [(X_val, y_val_xgb)]

    model.fit(
        X_train,
        y_train_xgb,
        sample_weight=sample_weight,
        eval_set=eval_set,
        verbose=False,
    )

    return model