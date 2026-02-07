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

# ---- Label mapping: project labels {-1,0,1} <-> XGBoost labels {0,1,2} ----
LABEL_TO_XGB: Dict[int, int] = {-1: 0, 0: 1, 1: 2}
XGB_TO_LABEL: Dict[int, int] = {0: -1, 1: 0, 2: 1}


# This is the canonical class order your project will use everywhere
PROJECT_CLASSES = np.array([-1, 0, 1], dtype=int)

@dataclass
class TradingXGBModel:
    model: XGBClassifier
    
    @property
    def classes_(self) -> np.ndarray:
        return PROJECT_CLASSES
    
    def predict(self, X) -> np.ndarray:
        pred_idx = self.model.predict(X)
        pred = np.vectorize(XGB_TO_LABEL.get)(pred_idx)
        return pred.astype(int)
    
    def predict_proba(self, X) -> np.ndarray:
        probs = self.model.predict_proba(X)
        if probs.shape[1] != 3:
            raise ValueError(f"Expected 3-class probabilities, got shape={probs.shape}")
        return probs   

def make_sample_weight_balanced(y_train_xgb: np.ndarray) -> np.ndarray:
    """
    Compute balanced sample weights for classes {0,1,2}.
    """
    if compute_class_weight is None:
        raise ImportError("sklearn is required for compute_class_weight")

    classes_idx = np.array([0, 1, 2], dtype=int)
    class_w = compute_class_weight(
        class_weight="balanced",
        classes=classes_idx,
        y=y_train_xgb
    )
    w_map = {c: w for c, w in zip(classes_idx, class_w)}
    return np.array([w_map[c] for c in y_train_xgb], dtype=float)

def map_labels_to_xgb(y: np.ndarray) -> np.ndarray:
    """
    Map labels from {-1,0,1} -> {0,1,2} for XGBoost.
    """
    y = np.asarray(y).astype(int)
    bad = set(np.unique(y)) - {-1, 0, 1}
    if bad:
        raise ValueError(f"Unexpected labels: {sorted(bad)} (expected only -1,0,1)")
    return np.vectorize(LABEL_TO_XGB.get)(y).astype(int)


def build_xgb_classifier(random_state: int = 88) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=800,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="mlogloss",
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
    
    y_train_xgb = map_labels_to_xgb(np.asarray(y_train))


    sample_weight = None
    if use_class_weights:
        sample_weight = make_sample_weight_balanced(y_train_xgb)


    model = build_xgb_classifier(random_state=random_state) 


    eval_set: Optional[list] = None
    if X_val is not None and y_val is not None:
        y_val_xgb = map_labels_to_xgb(np.asarray(y_val))
        eval_set = [(X_val, y_val_xgb)]


    model.fit(
        X_train,
        y_train_xgb,
        sample_weight=sample_weight, # set use_class_weights=False to disable
        eval_set=eval_set if eval_set else None,
        verbose=False,
    )


    return TradingXGBModel(model=model)