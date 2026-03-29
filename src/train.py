"""Training orchestration: split, fit, cross-validate and compare models.

This module owns the *how* of modelling.  It brings together the model
definitions from ``model`` and the scoring functions from ``evaluate``,
runs the full training loop and returns a ``TrainArtifacts`` bundle.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.evaluate import evaluate_on_test, run_cross_validation
from src.model import TrainArtifacts, build_candidates, build_preprocessor

logger = logging.getLogger(__name__)


def train_and_compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainArtifacts:
    """Split, preprocess, train, cross-validate and evaluate all candidates.

    The returned ``TrainArtifacts`` contains both held-out test metrics and
    cross-validation metrics so downstream reporting can show both.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info(
        "Train/test split: %d train, %d test (%.0f%% test, stratified).",
        len(X_train), len(X_test), test_size * 100,
    )

    preprocessor, numeric_features, categorical_features = build_preprocessor(X)
    candidates = build_candidates(preprocessor, random_state)

    test_rows: list[dict] = []
    cv_rows: list[dict] = []
    probabilities: dict[str, pd.Series] = {}
    predictions: dict[str, pd.Series] = {}

    for name, pipeline in candidates.items():
        logger.info("Training %-25s ...", name)
        pipeline.fit(X_train, y_train)

        y_prob = pd.Series(
            pipeline.predict_proba(X_test)[:, 1],
            index=X_test.index,
        )
        y_pred = pd.Series(pipeline.predict(X_test), index=X_test.index)

        test_scores = evaluate_on_test(y_test, y_pred, y_prob)
        test_rows.append({"model": name, **test_scores})
        probabilities[name] = y_prob
        predictions[name] = y_pred

        cv_scores = run_cross_validation(pipeline, X_train, y_train, random_state)
        cv_rows.append({"model": name, **{f"cv_{k}": v for k, v in cv_scores.items()}})

        logger.info(
            "  test  recall=%.3f  f1=%.3f  auc=%.3f",
            test_scores["recall"], test_scores["f1"], test_scores["roc_auc"],
        )
        logger.info(
            "  cv    recall=%.3f  f1=%.3f  auc=%.3f",
            cv_scores["recall"], cv_scores["f1"], cv_scores["roc_auc"],
        )

    metrics = (
        pd.DataFrame(test_rows)
        .sort_values(by=["recall", "f1", "roc_auc"], ascending=False)
        .reset_index(drop=True)
    )
    cv_metrics = pd.DataFrame(cv_rows)

    fitted_preprocessor = candidates["logistic_regression"].named_steps["preprocessor"]
    try:
        raw_names = fitted_preprocessor.get_feature_names_out()
        feature_names = [n.replace("num__", "").replace("cat__", "") for n in raw_names]
    except Exception:
        feature_names = []

    return TrainArtifacts(
        models=candidates,
        metrics=metrics,
        cv_metrics=cv_metrics,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        probabilities=probabilities,
        predictions=predictions,
        feature_names_after_preprocessing=feature_names,
    )
