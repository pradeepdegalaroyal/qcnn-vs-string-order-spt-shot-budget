"""Matrix-product-state (MPS) compression of a dense state by sequential truncated singular value decomposition."""

from __future__ import annotations

import numpy as np


def truncate_state(psi: np.ndarray, n: int, chi: int) -> np.ndarray:
    """Return the dense, renormalised state after cutting every bond (open chain) to dimension ``chi``."""
    rest = psi.reshape(1, -1)
    tensors = []
    for _ in range(n - 1):
        rows = rest.shape[0]
        u, s, vh = np.linalg.svd(rest.reshape(rows * 2, -1), full_matrices=False)
        k = min(chi, len(s))
        tensors.append(u[:, :k].reshape(rows, 2, k))
        rest = s[:k, None] * vh[:k]
    tensors.append(rest.reshape(rest.shape[0], 2, 1))
    cur = tensors[0].reshape(2, -1)
    for t in tensors[1:]:
        cur = np.tensordot(cur, t, axes=([-1], [0])).reshape(-1, t.shape[-1])
    out = cur.reshape(-1)
    return out / np.linalg.norm(out)


def half_chain_entropy(psi: np.ndarray, n: int) -> float:
    """Von Neumann entropy (in bits) across the middle cut of the open chain."""
    s = np.linalg.svd(psi.reshape(2 ** (n // 2), -1), compute_uv=False) ** 2
    s = s[s > 1e-14]
    return float(-(s * np.log2(s)).sum())
