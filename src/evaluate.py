"""Evaluation helpers: held-out metrics and cross-validated scoring.

This module owns the *how well* of modelling.  It is intentionally stateless:
callers pass in predictions / pipelines and get back plain dicts of scores.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

N_CV_FOLDS = 5

CV_SCORING = {
    "accuracy": "accuracy",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def evaluate_on_test(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_prob: pd.Series,
) -> dict[str, float]:
    """Compute held-out test metrics: accuracy, recall, F1 and ROC-AUC."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def run_cross_validation(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> dict[str, float]:
    """Return mean CV scores across ``N_CV_FOLDS`` stratified folds."""

    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=random_state)
    try:
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=CV_SCORING,
            error_score="raise",
        )
    except Exception:
        logger.warning("CV failed for pipeline — returning zeros.")
        return {k: 0.0 for k in CV_SCORING}

    return {
        metric: float(np.mean(scores[f"test_{metric}"]))
        for metric in CV_SCORING
    }
