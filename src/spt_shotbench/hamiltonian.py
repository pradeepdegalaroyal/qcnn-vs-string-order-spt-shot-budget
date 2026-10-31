"""Cluster-Ising chain with periodic boundaries: exact ground states and Pauli expectations.

H = -J sum_i Z_{i-1} X_i Z_{i+1} - h1 sum_i X_i - h2 sum_i X_i X_{i+1}

Qubit ``q`` is axis ``q`` of the state reshaped to ``(2,) * N`` (bit position ``N-1-q`` of the index).
For ``h2 = 0`` the symmetry-protected-topological (SPT) phase ends at ``h1 = J``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

CRITICAL_H1 = 1.0


def _bits(n: int) -> np.ndarray:
    """Return an (n, 2**n) array of bit values; row q is the bit of qubit q."""
    idx = np.arange(2**n)
    return (idx[None, :] >> (n - 1 - np.arange(n))[:, None]) & 1


def _flip_index(n: int, qubits: tuple[int, ...]) -> np.ndarray:
    mask = 0
    for q in qubits:
        mask ^= 1 << (n - 1 - q)
    return np.arange(2**n) ^ mask


def ground_state(n: int, h1: float, h2: float = 0.0, j: float = 1.0) -> tuple[np.ndarray, float]:
    """Return the real, normalised ground state and its energy of the periodic chain."""
    z = 1 - 2 * _bits(n)
    flips = [_flip_index(n, (q,)) for q in range(n)]
    flips2 = [_flip_index(n, (q, (q + 1) % n)) for q in range(n)]
    zz = [z[(q - 1) % n] * z[(q + 1) % n] for q in range(n)]

    def matvec(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v).reshape(-1)
        out = np.zeros_like(v, dtype=float)
        for q in range(n):
            out -= j * zz[q] * v[flips[q]]
            out -= h1 * v[flips[q]]
            if h2 != 0.0:
                out -= h2 * v[flips2[q]]
        return out

    dim = 2**n
    if dim <= 1024:
        mat = np.stack([matvec(e) for e in np.eye(dim)], axis=1)
        vals, vecs = np.linalg.eigh(mat)
        energy, psi = float(vals[0]), vecs[:, 0]
    else:
        op = LinearOperator((dim, dim), matvec=matvec, dtype=float)
        v0 = np.random.default_rng(0).standard_normal(dim)
        vals, vecs = eigsh(op, k=1, which="SA", v0=v0, tol=1e-10)
        energy, psi = float(vals[0]), vecs[:, 0]
    psi = psi / np.linalg.norm(psi)
    if psi[np.argmax(np.abs(psi))] < 0:
        psi = -psi
    return psi, energy


def pauli_expectation(psi: np.ndarray, n: int, ops: dict[int, str]) -> float:
    """Expectation of a product of Pauli X and Z operators given as ``{qubit: 'X'|'Z'}``."""
    z = 1 - 2 * _bits(n)
    sign = np.ones(2**n)
    x_qubits = []
    for q, op in ops.items():
        if op == "Z":
            sign = sign * z[q]
        elif op == "X":
            x_qubits.append(q)
        else:
            raise ValueError(f"unsupported Pauli {op!r}")
    flipped = psi[_flip_index(n, tuple(x_qubits))]
    return float(np.sum(psi * sign * flipped))


def string_operator(n: int) -> dict[int, str]:
    """Cluster-state string order Z_0 X_1 X_3 ... X_{n-3} Z_{n-2} (product of ZXZ stabilisers)."""
    ops = {q: "X" for q in range(1, n - 2, 2)}
    ops[0] = "Z"
    ops[n - 2] = "Z"
    return ops


def string_order(psi: np.ndarray, n: int) -> float:
    """Expectation value of the string order parameter."""
    return pauli_expectation(psi, n, string_operator(n))


def test_and_pool_fields(
    delta_test: np.ndarray, delta_pool: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return mirrored field grids ``1 -/+ delta`` for the test and the labelled pool."""
    mirror = lambda d: np.sort(np.concatenate([CRITICAL_H1 - d, CRITICAL_H1 + d]))  # noqa: E731
    return mirror(delta_test), mirror(delta_pool)


def load_states(n: int, fields: np.ndarray, cache_dir: str | Path = "data/raw") -> np.ndarray:
    """Ground states for a list of ``h1`` values (rows), cached on disk for large ``n``."""
    fields = np.asarray(fields, dtype=float)
    path = None
    if n >= 12:
        key = hashlib.md5(np.round(fields, 9).tobytes()).hexdigest()[:10]
        path = Path(cache_dir) / f"states_N{n}_{key}.npy"
        if path.exists():
            return np.load(path)
    states = np.stack([ground_state(n, float(h))[0] for h in fields])
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, states)
    return states
