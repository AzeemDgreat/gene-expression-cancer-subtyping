"""Preprocessing: a data scale check and an unsupervised variance filter."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def scale_report(X, integer_tol=1e-6):
    """Describe the scale of an expression matrix.

    Differential expression packages such as DESeq2 need raw integer counts. This report
    states what the values look like, so the README can say it plainly instead of assuming.
    """
    values = np.asarray(X, dtype=np.float64)
    non_negative = bool((values >= 0).all())
    sample = values[: min(len(values), 200)].ravel()
    integer_like = bool(np.all(np.abs(sample - np.round(sample)) < integer_tol))
    maximum = float(values.max())
    if integer_like:
        verdict = "integer values: consistent with raw counts"
    elif maximum < 50:
        verdict = "non-integer, small range: consistent with log-scale normalized data"
    else:
        verdict = "non-integer, large range: consistent with linear-scale normalized data"
    return {
        "n_samples": int(values.shape[0]),
        "n_genes": int(values.shape[1]),
        "min": float(values.min()),
        "max": maximum,
        "mean": float(values.mean()),
        "non_negative": non_negative,
        "integer_like": integer_like,
        "fraction_zero": float((values == 0).mean()),
        "verdict": verdict,
    }


class TopVarianceSelector(BaseEstimator, TransformerMixin):
    """Keep the ``k`` highest-variance genes. Unsupervised, fitted on the training data only."""

    def __init__(self, k=5000):
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        variances = X.var(axis=0)
        order = np.argsort(-variances, kind="stable")
        self.support_ = np.sort(order[: min(self.k, X.shape[1])])
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]
