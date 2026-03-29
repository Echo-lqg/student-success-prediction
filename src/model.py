"""Model definitions and preprocessing pipeline.

This module owns the *what* of modelling: which preprocessor to use, which
classifiers to compare, and the data-class that carries all training
artefacts downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class TrainArtifacts:
    """Everything downstream stages need: fitted models, splits and scores."""

    models: dict[str, Pipeline]
    metrics: pd.DataFrame
    cv_metrics: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    probabilities: dict[str, pd.Series]
    predictions: dict[str, pd.Series]
    feature_names_after_preprocessing: list[str] = field(default_factory=list)


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Create a column transformer that scales numerics and one-hot-encodes
    categoricals.  Returns the transformer and the two column lists so callers
    can inspect them."""

    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor, numeric_features, categorical_features


def build_candidates(
    preprocessor: ColumnTransformer,
    random_state: int,
) -> dict[str, Pipeline]:
    """Return an ordered dict of (name -> sklearn Pipeline) to evaluate.

    Models
    ------
    - **majority_baseline** : trivial most-frequent classifier.
    - **logistic_regression** : interpretable linear model with balanced
      class weights — favours recall on the minority class.
    - **random_forest** : non-linear ensemble with balanced class weights.
    """

    return {
        "majority_baseline": Pipeline([
            ("preprocessor", preprocessor),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]),
        "logistic_regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=random_state,
            )),
        ]),
        "random_forest": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),
    }
