import numpy as np
import pandas as pd
import pytest

from gexpr.data import load_dataset, make_synthetic_dataset, validate_dataset


def _write(tmp_path, X, y):
    X.to_csv(tmp_path / "data.csv")
    y.to_frame("Class").to_csv(tmp_path / "labels.csv")


def test_synthetic_shape_and_classes():
    X, y = make_synthetic_dataset(n_samples=100, n_genes=50, n_informative=5, seed=1)
    assert X.shape == (100, 50)
    assert set(y) == {"BRCA", "COAD", "KIRC", "LUAD", "PRAD"}
    assert (X.to_numpy() >= 0).all()
    assert X.index.equals(y.index)


def test_synthetic_is_reproducible():
    a, _ = make_synthetic_dataset(seed=3)
    b, _ = make_synthetic_dataset(seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_load_dataset_roundtrip(tmp_path):
    X, y = make_synthetic_dataset(n_samples=50, n_genes=20, n_informative=3)
    _write(tmp_path, X, y)
    X2, y2 = load_dataset(tmp_path, validate_shape=False)
    assert X2.shape == X.shape
    assert list(y2) == list(y)
    assert X2.dtypes.eq(np.float32).all()


def test_load_dataset_enforces_expected_shape(tmp_path):
    X, y = make_synthetic_dataset(n_samples=50, n_genes=20, n_informative=3)
    _write(tmp_path, X, y)
    with pytest.raises(ValueError, match="expected shape"):
        load_dataset(tmp_path)


def test_missing_files_give_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_dataset(tmp_path)


def test_validate_rejects_misaligned_ids():
    X, y = make_synthetic_dataset(n_samples=50, n_genes=20, n_informative=3)
    with pytest.raises(ValueError, match="do not match"):
        validate_dataset(X, y.rename(index=lambda s: s + "x"))


def test_validate_reorders_labels_to_match():
    X, y = make_synthetic_dataset(n_samples=50, n_genes=20, n_informative=3)
    _, y_shuffled = validate_dataset(X, y.sample(frac=1, random_state=0))
    assert y_shuffled.index.equals(X.index)


def test_validate_rejects_missing_values():
    X, y = make_synthetic_dataset(n_samples=50, n_genes=20, n_informative=3)
    X.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        validate_dataset(X, y)


def test_validate_rejects_tiny_classes():
    X, y = make_synthetic_dataset(n_samples=50, n_genes=20, n_informative=3)
    y = y.copy()
    y.iloc[:46] = "BRCA"
    with pytest.raises(ValueError, match="at least 5"):
        validate_dataset(X, y)
