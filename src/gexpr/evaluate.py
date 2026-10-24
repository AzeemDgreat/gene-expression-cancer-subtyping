"""Leakage-safe cross-validation, metrics and figures.

Every model is a Pipeline, so gene filtering and scaling are refitted on each training fold.
Figures use one accent hue plus gray: model identity is carried by the axis labels, and the
five cancer types are shown as small multiples (one class highlighted per panel) rather than
five colours in one scatter plot.
"""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# Design tokens (light surface)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
ACCENT = "#2a78d6"
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


# --------------------------------------------------------------------------- cross-validation
def cross_validate_models(X, y, models, n_splits=5, random_state=42, verbose=True):
    """Stratified k-fold CV for every model.

    Returns ``(fold_metrics, oof_predictions)``: a long DataFrame with one row per
    model and fold, and a DataFrame of out-of-fold predictions (one column per model).
    """
    X_arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
    index = y.index if hasattr(y, "index") else np.arange(len(y_arr))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(splitter.split(X_arr, y_arr))

    rows = []
    oof = {name: np.empty(len(y_arr), dtype=object) for name in models}
    for name, model in models.items():
        start = time.time()
        for fold, (train, test) in enumerate(splits):
            fitted = clone(model).fit(X_arr[train], y_arr[train])
            predicted = fitted.predict(X_arr[test])
            oof[name][test] = predicted
            rows.append(
                {
                    "model": name,
                    "fold": fold,
                    "accuracy": accuracy_score(y_arr[test], predicted),
                    "macro_f1": f1_score(y_arr[test], predicted, average="macro"),
                }
            )
        if verbose:
            scores = [r["accuracy"] for r in rows if r["model"] == name]
            print(
                f"  {name:14s} accuracy {np.mean(scores):.4f}  ({time.time() - start:.0f}s)",
                flush=True,
            )
    return pd.DataFrame(rows), pd.DataFrame(oof, index=index)


def summarize(fold_metrics):
    """Mean and standard deviation of each metric across folds, one row per model."""
    grouped = fold_metrics.groupby("model", sort=False)[["accuracy", "macro_f1"]]
    summary = grouped.agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def per_class_report(y_true, y_pred):
    """Precision, recall and F1 per class as a DataFrame."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return pd.DataFrame(report).T.reset_index().rename(columns={"index": "label"})


def confusion_table(y_true, y_pred):
    labels = sorted(set(np.asarray(y_true)) | set(np.asarray(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm, labels


# --------------------------------------------------------------------------- figures
def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9, length=3, color=AXIS)


def _save(fig, path):
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(summary, path, metric="macro_f1"):
    """Horizontal bars of mean +/- sd across folds. One hue, zero baseline, direct labels."""
    data = summary.sort_values(f"{metric}_mean")
    fig, ax = plt.subplots(figsize=(7, 0.7 * len(data) + 1.4), facecolor=SURFACE)
    _style(ax)
    y = np.arange(len(data))
    mean, sd = data[f"{metric}_mean"].to_numpy(), data[f"{metric}_std"].fillna(0).to_numpy()
    ax.barh(y, mean, height=0.5, color=ACCENT, zorder=3)
    ax.errorbar(mean, y, xerr=sd, fmt="none", ecolor=INK_2, elinewidth=1.2, capsize=3, zorder=4)
    for yi, m, s in zip(y, mean, sd, strict=True):
        ax.text(min(m + s + 0.03, 1.16), yi, f"{m:.3f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, data["model"])
    ax.set_xlim(0, 1.2)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    label = "Macro-F1" if metric == "macro_f1" else metric.capitalize()
    ax.set_xlabel(f"{label}, mean +/- sd across CV folds", color=INK_2, fontsize=9)
    ax.set_title("Model comparison", loc="left", color=INK, fontsize=12, fontweight="bold")
    _save(fig, path)


def plot_confusion(cm, labels, path, title="Confusion matrix (out-of-fold predictions)"):
    """Counts on a one-hue sequential ramp; the true class is on the rows."""
    cmap = LinearSegmentedColormap.from_list("blue_seq", SEQUENTIAL)
    fig, ax = plt.subplots(figsize=(5.2, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=max(cm.max(), 1))
    ax.set_xticks(range(len(labels)), labels, color=INK_2, fontsize=9)
    ax.set_yticks(range(len(labels)), labels, color=INK_2, fontsize=9)
    ax.set_xlabel("Predicted", color=INK_2, fontsize=9)
    ax.set_ylabel("True", color=INK_2, fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    threshold = cm.max() * 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, int(cm[i, j]), ha="center", va="center", fontsize=10,
                color="#ffffff" if cm[i, j] > threshold else INK,
            )
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    _save(fig, path)


def plot_pca_small_multiples(X, y, path, n_genes=2000, random_state=0):
    """PCA of the highest-variance genes, one panel per class with that class highlighted.

    This is an unsupervised view for exploration only; nothing from it feeds the models.
    """
    values = np.asarray(X, dtype=np.float64)
    top = np.argsort(-values.var(axis=0))[: min(n_genes, values.shape[1])]
    scaled = StandardScaler().fit_transform(values[:, top])
    pca = PCA(n_components=2, random_state=random_state)
    pcs = pca.fit_transform(scaled)
    classes = sorted(set(np.asarray(y)))
    y_arr = np.asarray(y)

    fig, axes = plt.subplots(
        1, len(classes), figsize=(2.9 * len(classes), 3.3), sharex=True, sharey=True,
        facecolor=SURFACE,
    )
    axes = np.atleast_1d(axes)
    for ax, cls in zip(axes, classes, strict=True):
        _style(ax)
        mask = y_arr == cls
        ax.scatter(pcs[~mask, 0], pcs[~mask, 1], s=9, color=AXIS, alpha=0.7, linewidths=0)
        ax.scatter(pcs[mask, 0], pcs[mask, 1], s=11, color=ACCENT, linewidths=0)
        ax.set_title(f"{cls} (n={int(mask.sum())})", loc="left", color=INK, fontsize=10)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})", color=INK_2, fontsize=8)
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})", color=INK_2, fontsize=8)
    fig.suptitle(
        "PCA of the top-variance genes, one class highlighted per panel",
        x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold", y=1.04,
    )
    _save(fig, path)
