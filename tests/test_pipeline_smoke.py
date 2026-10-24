import importlib.util
import json
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_pipeline.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_pipeline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synthetic_end_to_end_run_writes_all_outputs(tmp_path):
    module = _load_script()
    code = module.main(["--synthetic", "--results-dir", str(tmp_path), "--n-splits", "3"])
    assert code == 0

    expected = [
        "scale_report.json", "de_kruskal_top.csv", "de_one_vs_rest_top.csv",
        "cv_metrics_per_fold.csv", "cv_summary.csv", "oof_predictions.csv",
        "per_class_report.csv", "run_info.json",
        "figures/model_comparison.png", "figures/confusion_best_model.png",
        "figures/pca_by_class.png",
    ]
    for name in expected:
        assert (tmp_path / name).exists(), name

    info = json.loads((tmp_path / "run_info.json").read_text())
    assert info["synthetic"] is True
    assert info["best_model_by_accuracy"] in {
        "logreg_l2", "logreg_l1", "svm_linear", "random_forest", "mlp_torch"
    }
    summary = pd.read_csv(tmp_path / "cv_summary.csv")
    assert len(summary) == 5
    assert (summary["accuracy_mean"] > 0.5).all()
    report = json.loads((tmp_path / "scale_report.json").read_text())
    assert "verdict" in report


def test_skip_mlp_flag(tmp_path):
    module = _load_script()
    module.main(["--synthetic", "--results-dir", str(tmp_path), "--n-splits", "3", "--skip-mlp"])
    summary = pd.read_csv(tmp_path / "cv_summary.csv")
    assert "mlp_torch" not in set(summary["model"])
    assert len(summary) == 4
