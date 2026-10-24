"""Run the full analysis: scale check, differential expression, model comparison, figures.

    python scripts/download_data.py
    python scripts/run_pipeline.py

Use ``--synthetic`` for a fast dry run on generated data (results are meaningless biologically).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
import torch  # noqa: E402

from gexpr import evaluate  # noqa: E402
from gexpr.data import load_dataset, make_synthetic_dataset  # noqa: E402
from gexpr.differential_expression import (  # noqa: E402
    kruskal_table,
    one_vs_rest_table,
    top_genes_per_class,
)
from gexpr.models import build_models  # noqa: E402
from gexpr.preprocess import scale_report  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--prefilter", type=int, default=5000, help="top-variance genes kept first")
    p.add_argument("--n-genes", type=int, default=500, help="top Kruskal-Wallis genes per fold")
    p.add_argument("--top-table", type=int, default=100, help="rows kept in gene tables")
    p.add_argument("--skip-mlp", action="store_true", help="skip the PyTorch model")
    p.add_argument("--synthetic", action="store_true", help="use generated data (dry run)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.time()
    out = Path(args.results_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    # 1. Data and scale check
    if args.synthetic:
        print("Using SYNTHETIC data (dry run).")
        X, y = make_synthetic_dataset(seed=args.seed)
        args.prefilter, args.n_genes = min(args.prefilter, 400), min(args.n_genes, 60)
    else:
        X, y = load_dataset(args.data_dir)
    print(f"Data: {X.shape[0]} samples x {X.shape[1]} genes; classes {y.value_counts().to_dict()}")
    report = scale_report(X)
    (out / "scale_report.json").write_text(json.dumps(report, indent=2))
    print("Scale check:", report["verdict"])

    # 2. Exploratory differential expression on the full dataset (descriptive only)
    print("Differential expression ...", flush=True)
    kw = kruskal_table(X.to_numpy(), y.to_numpy(), X.columns)
    kw.head(args.top_table).to_csv(out / "de_kruskal_top.csv", index=False)
    ovr = one_vs_rest_table(X.to_numpy(), y.to_numpy(), X.columns)
    top_genes_per_class(ovr, n=25).to_csv(out / "de_one_vs_rest_top.csv", index=False)
    n_sig = int((kw["q_value"] < 0.05).sum())
    print(f"  {n_sig} of {len(kw)} genes differ across classes (Kruskal-Wallis, BH q < 0.05)")

    # 3. Leakage-safe cross-validation
    print(f"{args.n_splits}-fold cross-validation ...", flush=True)
    models = build_models(args.prefilter, args.n_genes, args.seed)
    if args.skip_mlp:
        models.pop("mlp_torch")
    folds, oof = evaluate.cross_validate_models(X, y, models, args.n_splits, args.seed)
    summary = evaluate.summarize(folds)
    folds.to_csv(out / "cv_metrics_per_fold.csv", index=False)
    summary.to_csv(out / "cv_summary.csv", index=False)
    oof.assign(true=y.to_numpy()).to_csv(out / "oof_predictions.csv")

    reports = []
    for name in models:
        rep = evaluate.per_class_report(y.to_numpy(), oof[name].to_numpy())
        reports.append(rep.assign(model=name))
    pd.concat(reports).to_csv(out / "per_class_report.csv", index=False)

    best = summary.sort_values("accuracy_mean").iloc[-1]["model"]

    # 4. Figures
    print("Figures ...", flush=True)
    fig_dir = out / "figures"
    evaluate.plot_model_comparison(summary, fig_dir / "model_comparison.png")
    for name in models:
        cm, labels = evaluate.confusion_table(y.to_numpy(), oof[name].to_numpy())
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(out / f"confusion_{name}.csv")
        if name == best:
            evaluate.plot_confusion(
                cm, labels, fig_dir / "confusion_best_model.png",
                title=f"Confusion matrix: {name} (out-of-fold)",
            )
    evaluate.plot_pca_small_multiples(X.to_numpy(), y.to_numpy(), fig_dir / "pca_by_class.png")

    # 5. Run record
    info = {
        "synthetic": bool(args.synthetic),
        "seed": args.seed,
        "n_splits": args.n_splits,
        "prefilter_genes": args.prefilter,
        "selected_genes_per_fold": args.n_genes,
        "n_samples": int(X.shape[0]),
        "n_genes": int(X.shape[1]),
        "class_counts": {k: int(v) for k, v in y.value_counts().sort_index().items()},
        "genes_q_lt_0_05_kruskal": n_sig,
        "best_model_by_accuracy": best,
        "runtime_seconds": round(time.time() - started, 1),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2))

    print("\nSummary (mean over folds):")
    print(summary.round(4).to_string(index=False))
    print(f"\nDone in {info['runtime_seconds']}s. Results in {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
