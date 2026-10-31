"""Classical baselines that read the same quantum state through finite measurement shots.

* ``string``: one product-Pauli setting (the cluster-state string order) with a single learned threshold.
* ``local-LR``: logistic regression on four shot-estimated local/non-local Pauli features that come from
  three measurement settings; the shot budget is split equally over the settings.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURES = ("mean X", "mean XX", "mean ZXZ (even sites)", "string order")
_H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)


def _eigenvalues(n: int) -> np.ndarray:
    idx = np.arange(2**n)
    return 1 - 2 * ((idx[None, :] >> (n - 1 - np.arange(n))[:, None]) & 1)


def _basis_probs(psi: np.ndarray, n: int, x_qubits: list[int]) -> np.ndarray:
    t = psi.reshape([2] * n)
    for q in x_qubits:
        t = np.moveaxis(np.tensordot(_H, t, axes=([1], [q])), 0, q)
    return (np.abs(t) ** 2).reshape(-1)


class LocalFeatureTables:
    """Outcome distributions and per-outcome feature values for the three measurement settings."""

    def __init__(self, psi: np.ndarray, n: int) -> None:
        e = _eigenvalues(n)
        xx = [q for q in range(n)]
        even = list(range(0, n, 2))
        string_x = list(range(1, n - 2, 2))
        self.probs = [
            _basis_probs(psi, n, xx),
            _basis_probs(psi, n, even),
            _basis_probs(psi, n, string_x),
        ]
        self.values = [
            np.stack([e.mean(0), np.mean([e[q] * e[(q + 1) % n] for q in range(n)], axis=0)]),
            np.mean([e[(q - 1) % n] * e[q] * e[(q + 1) % n] for q in even], axis=0)[None],
            (e[0] * e[n - 2] * np.prod([e[q] for q in string_x], axis=0))[None],
        ]

    def exact(self) -> np.ndarray:
        """Exact (infinite-shot) feature vector."""
        return np.concatenate([p @ v.T for p, v in zip(self.probs, self.values)])

    def sample(self, rng: np.random.Generator, shots: int, reps: int) -> np.ndarray:
        """``reps`` independent shot-estimated feature vectors, shots split over the settings."""
        per = max(1, shots // 3)
        cols = []
        for p, v in zip(self.probs, self.values):
            counts = rng.multinomial(per, p / p.sum(), size=reps)
            cols.append(counts @ v.T / per)
        return np.concatenate(cols, axis=1)


class LocalFeatureClassifier:
    """Standardised logistic regression on the exact features of the labelled training states."""

    def __init__(self, c: float = 10.0) -> None:
        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=c, max_iter=1000)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "LocalFeatureClassifier":
        self.model.fit(self.scaler.fit_transform(features), labels > 0)
        return self

    def predict_positive(self, features: np.ndarray) -> np.ndarray:
        """Boolean prediction ``True`` for the SPT class."""
        return self.model.predict(self.scaler.transform(features)).astype(bool)
