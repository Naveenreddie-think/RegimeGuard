"""LightGBM multi-class signal model - approved Phase 3 design (item #1): gradient-
boosted trees over a neural net, given ~3,900 total rows (far too few for a neural
net's extra capacity to pay off on tabular data), plus native feature-importance
explainability matching proposal §4.3's "kept honest and explainable" requirement.

No feature scaling here, unlike Phase 2's JM/HMM pipeline - tree splits are invariant
to monotonic transforms, so there's no scaler to fit per-fold. The point-in-time
discipline that mattered for Phase 2 (fit preprocessing on training data only)
carries over as "fit the model itself only on the fold's training data," which
walk_forward.py already guarantees.
"""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd

LABEL_TO_CLASS = {-1: 0, 0: 1, 1: 2}  # LightGBM needs contiguous class ids from 0
CLASS_TO_LABEL = {v: k for k, v in LABEL_TO_CLASS.items()}

LGBM_PARAMS = dict(
    objective="multiclass",
    max_depth=4,
    num_leaves=15,
    min_child_samples=40,
    learning_rate=0.05,
    n_estimators=500,
    class_weight="balanced",  # flat is the minority class (~21% overall)
    random_state=0,
    verbosity=-1,
)

EARLY_STOPPING_VALID_FRAC = 0.15
EARLY_STOPPING_ROUNDS = 30
INTERNAL_PURGE_DAYS = 1  # same H=1 purge, applied between the internal early-
                         # stopping validation slice and the training-proper slice
                         # before it - not the full 240-day embargo, since this
                         # internal slice is a training-time heuristic only and
                         # never contributes to a reported metric (the true held-out
                         # test fold is what's reported, protected by the outer
                         # purge+embargo in walk_forward.py).


def fit_fold_model(X_train: pd.DataFrame, y_train: pd.Series) -> lgb.LGBMClassifier:
    n = len(X_train)
    n_valid = max(int(n * EARLY_STOPPING_VALID_FRAC), 50)
    split = n - n_valid - INTERNAL_PURGE_DAYS
    assert split > 50, f"fold training set too small ({n} rows) for internal early-stopping split"

    X_tr, y_tr = X_train.iloc[:split], y_train.iloc[:split]
    X_val, y_val = X_train.iloc[split + INTERNAL_PURGE_DAYS:], y_train.iloc[split + INTERNAL_PURGE_DAYS:]

    y_tr_enc = y_tr.map(LABEL_TO_CLASS)
    y_val_enc = y_val.map(LABEL_TO_CLASS)

    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_tr, y_tr_enc,
        eval_X=X_val, eval_y=y_val_enc,
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model


def predict_labels(model: lgb.LGBMClassifier, X: pd.DataFrame) -> pd.Series:
    pred_enc = model.predict(X)
    return pd.Series(pred_enc, index=X.index).map(CLASS_TO_LABEL).astype(float)
