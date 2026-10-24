"""Load and validate the UCI PANCAN RNA-Seq dataset, plus a synthetic stand-in for tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_SHAPE = (801, 20531)
EXPECTED_CLASSES = ("BRCA", "COAD", "KIRC", "LUAD", "PRAD")


def load_dataset(data_dir, validate_shape=True):
    """Read ``data.csv`` and ``labels.csv`` from ``data_dir``.

    Returns ``(X, y)``: a samples-by-genes float32 DataFrame and a Series of class labels
    aligned on the sample ids. Raises ``ValueError`` if the files are inconsistent.
    """
    data_dir = Path(data_dir)
    data_path, labels_path = data_dir / "data.csv", data_dir / "labels.csv"
    for path in (data_path, labels_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run scripts/download_data.py first.")

    X = pd.read_csv(data_path, index_col=0).astype(np.float32)
    labels = pd.read_csv(labels_path, index_col=0)
    if "Class" not in labels.columns:
        raise ValueError("labels.csv must contain a 'Class' column")
    y = labels["Class"].astype(str)

    X, y = validate_dataset(X, y)
    if validate_shape and X.shape != EXPECTED_SHAPE:
        raise ValueError(f"expected shape {EXPECTED_SHAPE}, got {X.shape}")
    return X, y


def validate_dataset(X, y):
    """Check alignment, missing values and class sizes. Returns ``(X, y)`` aligned."""
    if not X.index.equals(y.index):
        common = X.index.intersection(y.index)
        if len(common) != len(X) or len(common) != len(y):
            raise ValueError("sample ids in the expression matrix and the labels do not match")
        y = y.loc[X.index]
    if X.isna().any().any():
        raise ValueError("expression matrix contains missing values")
    if y.isna().any():
        raise ValueError("labels contain missing values")
    counts = y.value_counts()
    if counts.min() < 5:
        raise ValueError(f"every class needs at least 5 samples for 5-fold CV, got {dict(counts)}")
    return X, y


def make_synthetic_dataset(
    n_samples=250, n_genes=600, n_informative=40, n_classes=5, noise=1.0, seed=0
):
    """Small log-scale-like dataset with class-specific shifts in some genes.

    Used by the tests and by ``run_pipeline.py --synthetic`` as a fast dry run. Results on
    this data say nothing about real tumours.
    """
    rng = np.random.default_rng(seed)
    classes = list(EXPECTED_CLASSES[:n_classes]) if n_classes <= 5 else [
        f"C{i}" for i in range(n_classes)
    ]
    y = np.array([classes[i % n_classes] for i in range(n_samples)])
    gene_means = rng.uniform(2.0, 9.0, size=n_genes)
    X = gene_means + rng.normal(0.0, noise, size=(n_samples, n_genes))
    informative = rng.choice(n_genes, size=n_informative, replace=False)
    for name in classes:
        shift = rng.normal(0.0, 1.5, size=n_informative)
        X[np.ix_(y == name, informative)] += shift
    X = np.clip(X, 0.0, None).astype(np.float32)
    index = [f"sample_{i}" for i in range(n_samples)]
    columns = [f"gene_{j}" for j in range(n_genes)]
    return pd.DataFrame(X, index=index, columns=columns), pd.Series(y, index=index, name="Class")
