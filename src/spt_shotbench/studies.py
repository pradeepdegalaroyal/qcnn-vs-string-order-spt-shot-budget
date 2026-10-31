"""The four studies: shot-budget accuracy, noise grid, MPS truncation and compiled cost."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .baselines import LocalFeatureClassifier, LocalFeatureTables
from .hamiltonian import CRITICAL_H1, load_states, string_operator, string_order, test_and_pool_fields
from .mps import half_chain_entropy, truncate_state
from .qcnn import QCNN, as_tensor, count_compiled_cx
from .shots import distance_bin, midpoint_threshold, prob_correct
from .train import qcnn_outputs, train_qcnn

INF = float("inf")


@dataclass
class Config:
    """Experiment sizes; ``quick`` is a smoke-test configuration."""

    n_small: int = 8
    n_large: int = 16
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    seeds_large: tuple[int, ...] = (0, 1, 2)
    n_train: int = 8
    n_train_sweep: tuple[int, ...] = (2, 4, 16)
    steps: int = 150
    steps_large: int = 40
    restarts: int = 2
    shots: tuple[int, ...] = (10, 30, 100, 300, 1000, 3000)
    lr_reps: int = 40
    noise_shots: int = 100
    p2_grid: tuple[float, ...] = (0.0, 0.002, 0.005, 0.01, 0.02, 0.05)
    r_grid: tuple[float, ...] = (0.0, 0.005, 0.01, 0.02, 0.05)
    chis: tuple[int, ...] = (2, 4, 8, 16, 32)
    delta_test: tuple[float, ...] = tuple(0.025 + 0.05 * np.arange(16))
    delta_pool: tuple[float, ...] = tuple(0.3 + 0.1 * np.arange(8))
    extra: dict = field(default_factory=dict)

    @classmethod
    def quick(cls) -> "Config":
        return cls(
            n_large=8, seeds=(0, 1), seeds_large=(0,), n_train=4, n_train_sweep=(2,), steps=30, steps_large=10,
            restarts=1, shots=(30, 300), lr_reps=5, p2_grid=(0.0, 0.02), r_grid=(0.0, 0.02),
            chis=(2, 4, 8), delta_test=tuple(0.025 + 0.1 * np.arange(8)), delta_pool=(0.3, 0.6, 1.0),
        )


@dataclass
class Data:
    """Ground states of the test grid and of the labelled pool for one chain length."""

    n: int
    test_h: np.ndarray
    pool_h: np.ndarray
    test_x: np.ndarray
    pool_x: np.ndarray
    test_y: np.ndarray
    pool_y: np.ndarray
    test_delta: np.ndarray
    test_bin: np.ndarray
    test_string: np.ndarray
    pool_string: np.ndarray


def build_data(n: int, cfg: Config) -> Data:
    """Compute exact ground states for the test fields and the labelled pool."""
    test_h, pool_h = test_and_pool_fields(np.array(cfg.delta_test), np.array(cfg.delta_pool))
    test_x, pool_x = load_states(n, test_h), load_states(n, pool_h)
    delta = np.abs(test_h - CRITICAL_H1)
    return Data(
        n, test_h, pool_h, test_x, pool_x,
        np.where(test_h < CRITICAL_H1, 1, -1), np.where(pool_h < CRITICAL_H1, 1, -1),
        delta, distance_bin(delta),
        np.array([string_order(p, n) for p in test_x]), np.array([string_order(p, n) for p in pool_x]),
    )


def train_indices(data: Data, seed: int, n_train: int) -> np.ndarray:
    """Stratified draw of ``n_train`` labelled pool states (half per phase)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(data.pool_y > 0)[0], np.where(data.pool_y < 0)[0]
    half = n_train // 2
    return np.concatenate([rng.choice(pos, half, replace=False), rng.choice(neg, half, replace=False)])


def _bin_masks(data: Data) -> dict[str, np.ndarray]:
    b = data.test_bin
    return {
        "near": b == "near", "mid": b == "mid", "far": b == "far",
        "crit": b != "far", "all": np.ones_like(b, dtype=bool),
    }


def _rows_from_correct(base: dict, shots: float, correct: np.ndarray, data: Data) -> list[dict]:
    return [
        {**base, "S": shots, "bin": name, "acc": float(correct[mask].mean())}
        for name, mask in _bin_masks(data).items()
    ]


def _single_observable_rows(base, data, mean_test, thr, shots_list) -> list[dict]:
    rows = _rows_from_correct(
        base, INF, ((mean_test > thr) == (data.test_y > 0)).astype(float), data
    )
    for s in shots_list:
        pc = np.array([prob_correct(m, thr, s, int(y)) for m, y in zip(mean_test, data.test_y)])
        rows += _rows_from_correct(base, s, pc, data)
    return rows


def main_study(
    cfg: Config, data: Data, seeds, n_train: int, variants=("qcnn", "qcnn-no-pool", "string", "local-LR"),
    steps: int | None = None, init_from: dict | None = None, dtype=torch.complex128,
) -> tuple[list[dict], list[dict], dict]:
    """Train on a few labelled states and measure accuracy versus shots on the test grid."""
    rows: list[dict] = []
    outs: list[dict] = []
    models: dict = {}
    for seed in seeds:
        tr = train_indices(data, seed, n_train)
        xs, ys, hs = data.pool_x[tr], data.pool_y[tr], data.pool_h[tr]
        for variant in variants:
            base = {"N": data.n, "seed": seed, "n_train": n_train, "method": variant}
            if variant in ("qcnn", "qcnn-no-pool"):
                model = train_qcnn(
                    xs, ys, data.n, use_pool=variant == "qcnn", seed=seed, restarts=cfg.restarts,
                    steps=steps or cfg.steps, dtype=dtype,
                    init_from=None if init_from is None else init_from[seed],
                )
                models[seed, variant] = model
                m_tr, m_te = qcnn_outputs(model, xs), qcnn_outputs(model, data.test_x)
                thr = midpoint_threshold(m_tr, ys)
                rows += _single_observable_rows(base, data, m_te, thr, cfg.shots)
                outs += _output_rows(base, data, m_te, thr)
            elif variant == "string":
                thr = midpoint_threshold(data.pool_string[tr], ys)
                rows += _single_observable_rows(base, data, data.test_string, thr, cfg.shots)
                outs += _output_rows(base, data, data.test_string, thr)
            elif variant == "local-LR":
                rows += _local_lr_rows(cfg, base, data, tr, seed)
    return rows, outs, models


def _output_rows(base: dict, data: Data, values: np.ndarray, thr: float) -> list[dict]:
    return [
        {**base, "h1": float(h), "delta": float(d), "label": int(y), "value": float(v), "threshold": thr}
        for h, d, y, v in zip(data.test_h, data.test_delta, data.test_y, values)
    ]


def _local_lr_rows(cfg: Config, base: dict, data: Data, tr: np.ndarray, seed: int) -> list[dict]:
    rng = np.random.default_rng(10_000 + seed)
    pool_tabs = [LocalFeatureTables(data.pool_x[i], data.n) for i in tr]
    clf = LocalFeatureClassifier().fit(np.stack([t.exact() for t in pool_tabs]), data.pool_y[tr])
    test_tabs = [LocalFeatureTables(p, data.n) for p in data.test_x]
    truth = data.test_y > 0
    rows = _rows_from_correct(
        base, INF, (clf.predict_positive(np.stack([t.exact() for t in test_tabs])) == truth).astype(float), data
    )
    for s in cfg.shots:
        correct = np.array([
            (clf.predict_positive(t.sample(rng, s, cfg.lr_reps)) == yy).mean()
            for t, yy in zip(test_tabs, truth)
        ])
        rows += _rows_from_correct(base, s, correct, data)
    return rows


def noise_grid(cfg: Config, data: Data, models: dict, seeds) -> list[dict]:
    """QCNN (gate depolarising ``p2`` + readout flip ``r``) versus the string estimator (readout flip only)."""
    weight = len(string_operator(data.n))
    rows = []
    for seed in seeds:
        model = models[seed, "qcnn"]
        tr = train_indices(data, seed, cfg.n_train)
        ys = data.pool_y[tr]
        for p2 in cfg.p2_grid:
            m_tr = qcnn_outputs(model, data.pool_x[tr], p2)
            m_te = qcnn_outputs(model, data.test_x, p2)
            for r in cfg.r_grid:
                a_q, a_s = 1 - 2 * r, (1 - 2 * r) ** weight
                thr_q = midpoint_threshold(a_q * m_tr, ys)
                thr_s = midpoint_threshold(a_s * data.pool_string[tr], ys)
                for name, mask in _bin_masks(data).items():
                    if name not in ("crit", "all"):
                        continue
                    pq = np.array([prob_correct(a_q * m, thr_q, cfg.noise_shots, int(y))
                                   for m, y in zip(m_te[mask], data.test_y[mask])])
                    ps = np.array([prob_correct(a_s * m, thr_s, cfg.noise_shots, int(y))
                                   for m, y in zip(data.test_string[mask], data.test_y[mask])])
                    rows.append({"seed": seed, "p2": p2, "r": r, "bin": name, "S": cfg.noise_shots,
                                 "acc_qcnn": float(pq.mean()), "acc_string": float(ps.mean())})
    return rows


def mps_study(cfg: Config, data: Data, models: dict, seeds) -> list[dict]:
    """QCNN readout and string order evaluated on MPS-compressed inputs of bond dimension ``chi``."""
    rows = []
    entropy = [half_chain_entropy(p, data.n) for p in data.test_x]
    for seed in seeds:
        model = models[seed, "qcnn"]
        tr = train_indices(data, seed, cfg.n_train)
        thr = midpoint_threshold(qcnn_outputs(model, data.pool_x[tr]), data.pool_y[tr])
        exact = qcnn_outputs(model, data.test_x)
        for chi in cfg.chis:
            trunc = np.stack([truncate_state(p, data.n, chi) for p in data.test_x])
            out = qcnn_outputs(model, trunc)
            st = np.array([string_order(p, data.n) for p in trunc])
            for i in range(len(trunc)):
                rows.append({
                    "N": data.n, "seed": seed, "chi": chi, "h1": float(data.test_h[i]),
                    "delta": float(data.test_delta[i]), "bin": str(data.test_bin[i]),
                    "label": int(data.test_y[i]), "entropy_bits": entropy[i],
                    "out_exact": float(exact[i]), "out_chi": float(out[i]),
                    "string_exact": float(data.test_string[i]), "string_chi": float(st[i]),
                    "correct_exact": bool((exact[i] > thr) == (data.test_y[i] > 0)),
                    "correct_chi": bool((out[i] > thr) == (data.test_y[i] > 0)),
                })
    return rows


def compiled_cost(models: dict) -> list[dict]:
    """CX count and depth of the trained circuits, on a line and with all-to-all connectivity."""
    out = []
    for (seed, variant), model in sorted(models.items()):
        if variant != "qcnn":
            continue
        for line in (True, False):
            out.append({"seed": seed, "connectivity": "line" if line else "all-to-all",
                        **count_compiled_cx(model, line=line)})
    return out
