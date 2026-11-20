"""Graders: each fits on a training frame and predicts a 0-5 score per row."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, FeatureBuilder


class MeanGrader:
    """Predicts the training mean for everyone: the floor any model must beat."""

    name = "mean (floor)"

    def fit(self, df):
        self.mean = float(df["score"].mean())
        return self

    def predict(self, df) -> np.ndarray:
        return np.full(len(df), self.mean)


class FeatureGrader:
    """A regressor over FeatureBuilder's features, optionally restricted to some of them."""

    def __init__(self, name: str, regressor, use: list[str] | None = None) -> None:
        self.name, self.regressor = name, regressor
        self.cols = [FEATURE_NAMES.index(f) for f in (use or FEATURE_NAMES)]

    def fit(self, df):
        self.fb = FeatureBuilder()
        self.regressor.fit(self.fb.fit_transform(df)[:, self.cols], df["score"])
        return self

    def features(self, df) -> np.ndarray:
        return self.fb.transform(df["question"], df["reference_answer"], df["student_answer"])[:, self.cols]

    def predict(self, df) -> np.ndarray:
        return np.clip(self.regressor.predict(self.features(df)), 0, 5)


def all_graders() -> list:
    """The comparison table's rows, simplest first."""
    return [
        MeanGrader(),
        FeatureGrader("TF-IDF cosine only", make_pipeline(StandardScaler(), Ridge(1.0)), ["word_tfidf_cos"]),
        FeatureGrader("All features + Ridge", make_pipeline(StandardScaler(), Ridge(1.0))),
        FeatureGrader("All features + Gradient Boosting",
                      HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3,
                                                    min_samples_leaf=20, random_state=0)),
    ]


def best_grader() -> FeatureGrader:
    """The grader the app uses."""
    return all_graders()[-1]
