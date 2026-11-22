"""
SM03 — the surrogate model.

A random forest is the right workhorse here: it trains in under a second on 40
labelled molecules, handles the sparse fingerprint block without scaling, and
gives per-compound epistemic uncertainty for free as the spread of the
individual trees' votes. That matters because the acquisition policies need a
usable uncertainty signal from the very first round, when only 40 labels exist
and anything heavier would be fitting noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier


@dataclass
class Posterior:
    """What the policy sees for each candidate before any reveal."""

    cids: list[str]
    probability: np.ndarray  # P(active), forest mean
    uncertainty: np.ndarray  # std of per-tree probability, epistemic
    entropy: np.ndarray  # predictive entropy, aleatoric + epistemic

    def as_dicts(self) -> tuple[dict[str, float], dict[str, float]]:
        return (
            {c: float(p) for c, p in zip(self.cids, self.probability)},
            {c: float(u) for c, u in zip(self.cids, self.uncertainty)},
        )


class ForestSurrogate:
    """Random forest with tree-disagreement uncertainty.

    Handles the degenerate early rounds explicitly: if the labelled set is
    single-class (quite possible among 40 hash-selected molecules), the forest
    cannot be fit, so we fall back to a flat prior with maximum uncertainty.
    That keeps round 1 exploring instead of crashing.
    """

    def __init__(self, n_estimators: int = 500, random_state: int = 0, **kwargs):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.kwargs = {
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
            "n_jobs": -1,
            **kwargs,
        }
        self._forest: RandomForestClassifier | None = None
        self._degenerate_class: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ForestSurrogate":
        y = np.asarray(y).astype(int)
        classes = np.unique(y)
        if classes.size < 2:
            self._forest = None
            self._degenerate_class = int(classes[0])
            return self
        self._degenerate_class = None
        self._forest = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            **self.kwargs,
        ).fit(X, y)
        return self

    def posterior(self, cids: list[str], X: np.ndarray) -> Posterior:
        n = X.shape[0]

        if self._forest is None:
            # Single-class or unfit: no information, maximum exploration.
            prob = np.full(n, 0.5, dtype=np.float64)
            unc = np.full(n, 0.5, dtype=np.float64)
            ent = np.ones(n, dtype=np.float64)
            return Posterior(list(cids), prob, unc, ent)

        active_col = int(np.where(self._forest.classes_ == 1)[0][0])
        per_tree = np.stack(
            [tree.predict_proba(X)[:, active_col] for tree in self._forest.estimators_]
        )
        prob = per_tree.mean(axis=0)
        unc = per_tree.std(axis=0)

        eps = 1e-12
        p = np.clip(prob, eps, 1 - eps)
        ent = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

        return Posterior(list(cids), prob, unc, ent)
