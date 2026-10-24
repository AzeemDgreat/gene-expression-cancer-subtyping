import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score

from gexpr.data import make_synthetic_dataset
from gexpr.models import TorchMLPClassifier, build_models


@pytest.fixture
def separable():
    rng = np.random.default_rng(0)
    y = np.array(["a", "b", "c"] * 50)
    X = rng.normal(size=(150, 10))
    X[y == "b", 0] += 4.0
    X[y == "c", 1] += 4.0
    return X, y


def test_mlp_learns_separable_data(separable):
    X, y = separable
    model = TorchMLPClassifier(epochs=40, random_state=0).fit(X, y)
    assert (model.predict(X) == y).mean() > 0.9
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_mlp_probabilities_sum_to_one(separable):
    X, y = separable
    proba = TorchMLPClassifier(epochs=5, random_state=0).fit(X, y).predict_proba(X)
    assert proba.shape == (150, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)


def test_mlp_is_reproducible_with_a_seed(separable):
    X, y = separable
    a = TorchMLPClassifier(epochs=5, random_state=7).fit(X, y).predict_proba(X)
    b = TorchMLPClassifier(epochs=5, random_state=7).fit(X, y).predict_proba(X)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_mlp_works_with_sklearn_clone(separable):
    X, y = separable
    cloned = clone(TorchMLPClassifier(epochs=3, hidden=(8,)))
    assert cloned.hidden == (8,) and cloned.fit(X, y).classes_.tolist() == ["a", "b", "c"]


def test_build_models_names():
    assert set(build_models()) == {
        "logreg_l2", "logreg_l1", "svm_linear", "random_forest", "mlp_torch"
    }


def test_every_model_fits_and_predicts_on_synthetic_data():
    X, y = make_synthetic_dataset(n_samples=150, n_genes=300, n_informative=30, seed=0)
    for name, model in build_models(prefilter=200, n_genes=40, random_state=0).items():
        fitted = clone(model).fit(X.to_numpy(), y.to_numpy())
        accuracy = (fitted.predict(X.to_numpy()) == y.to_numpy()).mean()
        assert accuracy > 0.5, name


def test_selection_inside_pipeline_does_not_leak():
    """Pure noise with random labels must score near chance (0.2 for 5 classes).

    If genes were chosen on the full dataset before cross-validation, 40 of 1500 noise genes
    would be picked using the labels and accuracy would be far above chance.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 1500))
    y = np.array(["BRCA", "COAD", "KIRC", "LUAD", "PRAD"] * 20)
    pipeline = build_models(prefilter=1000, n_genes=40, random_state=0)["logreg_l2"]
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    accuracy = cross_val_score(pipeline, X, y, cv=cv).mean()
    assert accuracy < 0.4
