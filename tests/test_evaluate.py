import numpy as np
import pandas as pd
import pytest

from gexpr import evaluate
from gexpr.data import make_synthetic_dataset
from gexpr.models import build_models

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def cv_result():
    X, y = make_synthetic_dataset(n_samples=150, n_genes=300, n_informative=30, seed=0)
    models = build_models(prefilter=200, n_genes=40, random_state=0)
    models = {k: models[k] for k in ("logreg_l2", "svm_linear")}
    folds, oof = evaluate.cross_validate_models(X, y, models, n_splits=3, verbose=False)
    return X, y, folds, oof


def test_cross_validation_output_shapes(cv_result):
    X, y, folds, oof = cv_result
    assert len(folds) == 2 * 3
    assert set(folds["model"]) == {"logreg_l2", "svm_linear"}
    assert set(folds["fold"]) == {0, 1, 2}
    assert oof.shape == (len(y), 2)
    assert oof.index.equals(y.index)
    assert not oof.isna().any().any()


def test_oof_predictions_cover_every_sample_once(cv_result):
    _, y, folds, oof = cv_result
    accuracy = (oof["logreg_l2"].to_numpy() == y.to_numpy()).mean()
    assert accuracy > 0.6  # informative synthetic genes are learnable
    fold_mean = folds[folds["model"] == "logreg_l2"]["accuracy"].mean()
    assert fold_mean == pytest.approx(accuracy, abs=0.05)


def test_summarize(cv_result):
    _, _, folds, _ = cv_result
    summary = evaluate.summarize(folds)
    assert list(summary["model"]) == ["logreg_l2", "svm_linear"]
    assert {"accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"} <= set(summary)


def test_per_class_report_and_confusion(cv_result):
    _, y, _, oof = cv_result
    report = evaluate.per_class_report(y.to_numpy(), oof["logreg_l2"].to_numpy())
    assert {"BRCA", "COAD", "KIRC", "LUAD", "PRAD"} <= set(report["label"])
    cm, labels = evaluate.confusion_table(y.to_numpy(), oof["logreg_l2"].to_numpy())
    assert cm.sum() == len(y) and cm.shape == (5, 5) and len(labels) == 5


def test_figures_are_written(cv_result, tmp_path):
    X, y, folds, oof = cv_result
    summary = evaluate.summarize(folds)
    cm, labels = evaluate.confusion_table(y.to_numpy(), oof["logreg_l2"].to_numpy())
    evaluate.plot_model_comparison(summary, tmp_path / "models.png")
    evaluate.plot_confusion(cm, labels, tmp_path / "cm.png")
    evaluate.plot_pca_small_multiples(X.to_numpy(), y.to_numpy(), tmp_path / "pca.png")
    for name in ("models.png", "cm.png", "pca.png"):
        data = (tmp_path / name).read_bytes()
        assert data.startswith(PNG_MAGIC) and len(data) > 5_000


def test_summary_handles_single_fold_std():
    folds = pd.DataFrame(
        {"model": ["m"], "fold": [0], "accuracy": [0.9], "macro_f1": [0.8]}
    )
    summary = evaluate.summarize(folds)
    assert np.isnan(summary.loc[0, "accuracy_std"])
