"""The four classifier families, each wrapped in a leakage-safe preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import sklearn
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn

from gexpr.differential_expression import KruskalSelector
from gexpr.preprocess import TopVarianceSelector


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """Small fully connected network trained with Adam, scikit-learn compatible."""

    def __init__(
        self,
        hidden=(128, 32),
        dropout=0.3,
        lr=1e-3,
        weight_decay=1e-3,
        epochs=60,
        batch_size=64,
        random_state=0,
    ):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def _build(self, n_in, n_out):
        layers, width = [], n_in
        for h in self.hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(self.dropout)]
            width = h
        layers.append(nn.Linear(width, n_out))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        self.classes_, y_idx = np.unique(np.asarray(y), return_inverse=True)
        X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        y_t = torch.as_tensor(y_idx, dtype=torch.long)

        self.model_ = self._build(X_t.shape[1], len(self.classes_))
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.CrossEntropyLoss()
        self.loss_history_ = []
        n = len(X_t)
        for _ in range(self.epochs):
            self.model_.train()
            order = rng.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                batch = order[start : start + self.batch_size]
                optimizer.zero_grad()
                loss = loss_fn(self.model_(X_t[batch]), y_t[batch])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch)
            self.loss_history_.append(epoch_loss / n)
        return self

    def predict_proba(self, X):
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.as_tensor(np.asarray(X), dtype=torch.float32))
            return torch.softmax(logits, dim=1).numpy()

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def _logistic_regression(l1, random_state):
    """Multinomial logistic regression with an L1 or L2 penalty.

    scikit-learn 1.8 deprecated ``penalty`` in favour of ``l1_ratio`` (0 = L2, 1 = L1), so
    pick the spelling that the installed version understands. The L1 model uses ``saga``,
    which supports multiclass problems and L1 together.
    """
    major, minor = (int(part) for part in sklearn.__version__.split(".")[:2])
    common = {"C": 0.1, "max_iter": 3000, "random_state": random_state}
    if not l1:
        return LogisticRegression(**common)  # the default penalty is L2
    if (major, minor) >= (1, 8):
        return LogisticRegression(l1_ratio=1.0, solver="saga", tol=1e-3, **common)
    return LogisticRegression(penalty="l1", solver="saga", tol=1e-3, **common)


def build_models(prefilter=5000, n_genes=500, random_state=42):
    """Return ``{name: Pipeline}``.

    Every pipeline fits its gene filtering and scaling on the training data only:
    variance prefilter -> Kruskal-Wallis top genes -> standardise -> classifier.
    """

    def pipeline(classifier):
        return Pipeline(
            [
                ("variance", TopVarianceSelector(k=prefilter)),
                ("de_select", KruskalSelector(k=n_genes)),
                ("scale", StandardScaler()),
                ("clf", classifier),
            ]
        )

    return {
        "logreg_l2": pipeline(_logistic_regression(l1=False, random_state=random_state)),
        "logreg_l1": pipeline(_logistic_regression(l1=True, random_state=random_state)),
        "svm_linear": pipeline(SVC(kernel="linear", C=0.01, random_state=random_state)),
        "random_forest": pipeline(
            RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=random_state)
        ),
        "mlp_torch": pipeline(TorchMLPClassifier(random_state=random_state)),
    }
