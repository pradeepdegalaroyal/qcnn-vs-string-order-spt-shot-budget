"""Training of the QCNN on exact expectation values of a few labelled ground states."""

from __future__ import annotations

import numpy as np
import torch

from .qcnn import QCNN, as_tensor


def transfer_init(model: QCNN, small: QCNN) -> None:
    """Initialise a larger QCNN from a trained smaller one (extra layers copy the last layer)."""
    with torch.no_grad():
        for layer in range(model.layers):
            src = min(layer, small.layers - 1)
            model.conv[layer].copy_(small.conv[src].to(model.rdtype))
            if model.use_pool:
                model.pool[layer].copy_(small.pool[src].to(model.rdtype))


def train_qcnn(
    states: np.ndarray,
    labels: np.ndarray,
    n_qubits: int,
    use_pool: bool = True,
    seed: int = 0,
    restarts: int = 2,
    steps: int = 120,
    lr: float = 0.05,
    dtype: torch.dtype = torch.complex128,
    init_from: QCNN | None = None,
) -> QCNN:
    """Minimise the mean-square error between ``<Z_out>`` and the +/-1 phase label with Adam.

    Returns the best of ``restarts`` random initialisations (lowest final training loss); with
    ``init_from`` a single run starts from the transferred parameters of a smaller QCNN.
    """
    x = as_tensor(states, dtype)
    y = torch.as_tensor(labels, dtype=torch.float64 if dtype == torch.complex128 else torch.float32)
    best: QCNN | None = None
    best_loss = float("inf")
    for r in range(1 if init_from is not None else restarts):
        model = QCNN(n_qubits, use_pool=use_pool, seed=1000 * seed + r, dtype=dtype)
        if init_from is not None:
            transfer_init(model, init_from)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
        loss = torch.tensor(float("inf"))
        for _ in range(steps):
            opt.zero_grad()
            loss = torch.mean((model(x) - y) ** 2)
            loss.backward()
            opt.step()
        final = float(torch.mean((model(x) - y) ** 2).detach())
        if final < best_loss:
            best, best_loss = model, final
    assert best is not None
    best.final_loss = best_loss  # type: ignore[attr-defined]
    return best


@torch.no_grad()
def qcnn_outputs(model: QCNN, states: np.ndarray, p2: float = 0.0) -> np.ndarray:
    """``<Z_out>`` for each state, optionally with depolarising gate noise ``p2`` (density matrix)."""
    x = as_tensor(states, model.cdtype)
    chunks = []
    for i in range(0, len(x), 16):
        part = x[i : i + 16]
        out = model.noisy_forward(part, p2) if p2 > 0 else model(part)
        chunks.append(out.double().numpy())
    return np.concatenate(chunks)
