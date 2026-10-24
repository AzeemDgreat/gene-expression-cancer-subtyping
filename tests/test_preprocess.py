import numpy as np

from gexpr.preprocess import TopVarianceSelector, scale_report


def test_scale_report_detects_integer_counts():
    counts = np.random.default_rng(0).poisson(20, size=(30, 10))
    report = scale_report(counts)
    assert report["integer_like"] is True
    assert "raw counts" in report["verdict"]


def test_scale_report_detects_log_scale_floats():
    values = np.random.default_rng(0).uniform(0, 12, size=(30, 10))
    report = scale_report(values)
    assert report["integer_like"] is False
    assert "log-scale" in report["verdict"]
    assert report["n_samples"] == 30 and report["n_genes"] == 10


def test_scale_report_flags_large_range_floats():
    values = np.random.default_rng(0).uniform(0, 5000, size=(30, 10))
    assert "linear-scale" in scale_report(values)["verdict"]


def test_top_variance_selector_keeps_highest_variance_columns():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 6)) * np.array([0.1, 5.0, 0.2, 4.0, 0.3, 0.1])
    selector = TopVarianceSelector(k=2).fit(X)
    assert list(selector.support_) == [1, 3]
    assert selector.transform(X).shape == (100, 2)


def test_top_variance_selector_k_larger_than_genes():
    X = np.random.default_rng(0).normal(size=(20, 4))
    assert TopVarianceSelector(k=100).fit(X).transform(X).shape == (20, 4)
