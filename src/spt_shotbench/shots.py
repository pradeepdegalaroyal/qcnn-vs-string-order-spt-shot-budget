"""Finite-shot decision statistics for single-observable classifiers (exact, no sampling)."""

from __future__ import annotations

import numpy as np
from scipy.stats import binom


def prob_correct(mean: float | np.ndarray, threshold: float, shots: int, label: int) -> np.ndarray:
    """Probability that ``shots`` +/-1 outcomes of an observable with expectation ``mean`` give the right class.

    The class is "+1" when the sample mean exceeds ``threshold``; an exact tie counts as half a hit.
    """
    mean = np.clip(np.asarray(mean, dtype=float), -1.0, 1.0)
    p_plus = (1 + mean) / 2
    cut = shots * (1 + threshold) / 2
    gt = binom.sf(np.floor(cut), shots, p_plus)
    tie = binom.pmf(cut, shots, p_plus) if float(cut).is_integer() else 0.0
    p_above = gt + 0.5 * tie
    return p_above if label > 0 else 1.0 - p_above


def midpoint_threshold(values: np.ndarray, labels: np.ndarray) -> float:
    """Midpoint between the class means of a one-dimensional score."""
    return float(0.5 * (values[labels > 0].mean() + values[labels < 0].mean()))


def distance_bin(delta: np.ndarray) -> np.ndarray:
    """Name of the distance-to-critical-point bin of each test state."""
    return np.where(delta < 0.15, "near", np.where(delta < 0.4, "mid", "far"))
