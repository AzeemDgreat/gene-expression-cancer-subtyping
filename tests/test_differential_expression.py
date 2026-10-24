import numpy as np
import pytest
from scipy import stats

from gexpr.differential_expression import (
    KruskalSelector,
    benjamini_hochberg,
    kruskal_by_gene,
    kruskal_table,
    one_vs_rest_table,
    top_genes_per_class,
)


@pytest.fixture
def planted():
    """Noise genes plus gene_3, which is clearly higher in class 'B'."""
    rng = np.random.default_rng(0)
    y = np.array(["A", "B", "C"] * 40)
    X = rng.normal(size=(120, 50))
    X[y == "B", 3] += 3.0
    names = [f"gene_{i}" for i in range(50)]
    return X, y, names


def test_kruskal_matches_scipy_per_gene(planted):
    X, y, _ = planted
    statistic, p_value = kruskal_by_gene(X, y)
    for gene in (0, 3, 17):
        h, p = stats.kruskal(*[X[y == c, gene] for c in np.unique(y)])
        assert statistic[gene] == pytest.approx(h)
        assert p_value[gene] == pytest.approx(p)


def test_constant_gene_is_uninformative():
    X = np.ones((30, 2))
    X[:, 1] = np.random.default_rng(0).normal(size=30)
    y = np.array(["A", "B", "C"] * 10)
    statistic, p_value = kruskal_by_gene(X, y)
    assert statistic[0] == 0.0 and p_value[0] == 1.0


def test_benjamini_hochberg_matches_hand_calculation():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    # Sorted p: .005,.01,.03,.04 -> raw adj: .02,.02,.04,.04 -> already monotone
    expected = np.array([0.02, 0.04, 0.04, 0.02])
    np.testing.assert_allclose(benjamini_hochberg(p), expected)


def test_benjamini_hochberg_never_below_raw_p():
    p = np.random.default_rng(0).uniform(size=200)
    q = benjamini_hochberg(p)
    assert (q >= p - 1e-12).all() and (q <= 1).all()


def test_kruskal_table_ranks_planted_gene_first(planted):
    X, y, names = planted
    table = kruskal_table(X, y, names)
    assert table.iloc[0]["gene"] == "gene_3"
    assert table.iloc[0]["q_value"] < 1e-6
    assert list(table.columns) == ["gene", "statistic", "p_value", "q_value"]


def test_one_vs_rest_finds_planted_gene_for_the_right_class(planted):
    X, y, names = planted
    table = one_vs_rest_table(X, y, names)
    row = table[(table["class"] == "B") & (table["gene"] == "gene_3")].iloc[0]
    assert row["mean_diff"] > 2.0
    assert row["q_value"] < 1e-6
    other = table[(table["class"] == "A") & (table["gene"] == "gene_3")].iloc[0]
    assert other["mean_diff"] < 0  # B is high, so A is lower than the rest
    assert len(table) == 3 * 50


def test_top_genes_per_class_shape_and_order(planted):
    X, y, names = planted
    top = top_genes_per_class(one_vs_rest_table(X, y, names), n=5)
    assert len(top) == 15
    assert set(top["class"]) == {"A", "B", "C"}
    best_b = top[top["class"] == "B"].iloc[0]
    assert best_b["gene"] == "gene_3"


def test_selector_keeps_informative_gene_and_k_columns(planted):
    X, y, _ = planted
    selector = KruskalSelector(k=5).fit(X, y)
    assert 3 in selector.support_
    assert selector.transform(X).shape == (120, 5)


def test_selector_uses_only_the_labels_it_was_fitted_on(planted):
    X, y, _ = planted
    selector = KruskalSelector(k=1).fit(X[:90], y[:90])
    unrelated_labels = np.random.default_rng(1).permutation(y)
    # Transform never looks at labels, so shuffling them cannot change the output.
    np.testing.assert_array_equal(selector.transform(X[90:]), selector.transform(X[90:]))
    assert selector.support_.tolist() == KruskalSelector(k=1).fit(X[:90], y[:90]).support_.tolist()
    assert not np.array_equal(
        KruskalSelector(k=1).fit(X[:90], unrelated_labels[:90]).pvalues_, selector.pvalues_
    )
