"""Differential expression with rank-based tests and Benjamini-Hochberg correction.

The expression values are normalized, not raw counts, so count-based methods such as DESeq2
do not apply. Rank-based tests make no distributional assumption about the values.

``mean_diff`` is the difference of group means on the data's own scale. It equals a log2 fold
change only if the data are log2-transformed, which this code cannot know.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.stats.multitest import multipletests


def kruskal_by_gene(X, y):
    """Kruskal-Wallis H test per gene across all classes. Returns ``(statistic, p_value)``."""
    X, y = np.asarray(X), np.asarray(y)
    groups = [X[y == c] for c in np.unique(y)]
    if len(groups) < 2:
        raise ValueError("need at least two classes")
    with np.errstate(invalid="ignore", divide="ignore"):  # constant genes divide 0 by 0
        statistic, p_value = stats.kruskal(*groups, axis=0)
    # A constant gene gives NaN; treat it as uninformative.
    return np.nan_to_num(statistic, nan=0.0), np.nan_to_num(p_value, nan=1.0)


def benjamini_hochberg(p_values):
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    return multipletests(np.asarray(p_values), method="fdr_bh")[1]


def kruskal_table(X, y, gene_names=None):
    """Genome-wide Kruskal-Wallis results, sorted by q-value."""
    statistic, p_value = kruskal_by_gene(X, y)
    names = gene_names if gene_names is not None else [f"gene_{i}" for i in range(len(p_value))]
    table = pd.DataFrame(
        {"gene": list(names), "statistic": statistic, "p_value": p_value,
         "q_value": benjamini_hochberg(p_value)}
    )
    return table.sort_values(["q_value", "statistic"], ascending=[True, False]).reset_index(
        drop=True
    )


def one_vs_rest_table(X, y, gene_names=None):
    """Mann-Whitney U test per gene for each class against all other samples.

    Returns one long table with columns class, gene, mean_diff, p_value, q_value. The
    q-value is corrected across all genes within each class comparison.
    """
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y)
    names = list(gene_names) if gene_names is not None else [
        f"gene_{i}" for i in range(X.shape[1])
    ]
    frames = []
    for cls in np.unique(y):
        inside, outside = X[y == cls], X[y != cls]
        _, p_value = stats.mannwhitneyu(inside, outside, axis=0)
        p_value = np.nan_to_num(p_value, nan=1.0)
        frames.append(
            pd.DataFrame(
                {
                    "class": cls,
                    "gene": names,
                    "mean_diff": inside.mean(axis=0) - outside.mean(axis=0),
                    "p_value": p_value,
                    "q_value": benjamini_hochberg(p_value),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def top_genes_per_class(table, n=20):
    """Top ``n`` genes per class by q-value, ties broken by absolute mean difference."""
    ranked = table.assign(abs_diff=table["mean_diff"].abs()).sort_values(
        ["class", "q_value", "abs_diff"], ascending=[True, True, False]
    )
    return ranked.groupby("class", sort=True).head(n).drop(columns="abs_diff").reset_index(
        drop=True
    )


class KruskalSelector(BaseEstimator, TransformerMixin):
    """Keep the ``k`` genes with the strongest Kruskal-Wallis signal in the training data.

    Used inside a scikit-learn Pipeline so it is refitted on each cross-validation training
    fold. Selecting genes on the full dataset before cross-validating would leak the test
    labels into the features and inflate the scores.
    """

    def __init__(self, k=500):
        self.k = k

    def fit(self, X, y):
        statistic, p_value = kruskal_by_gene(X, y)
        order = np.lexsort((-statistic, p_value))  # p ascending, then statistic descending
        self.pvalues_ = p_value
        self.support_ = np.sort(order[: min(self.k, len(order))])
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]
