"""Quantum convolutional neural network (QCNN) on a statevector / density-matrix simulator.

Layer ``l`` applies one shared two-qubit convolution unitary on all neighbouring pairs of the active
qubits (even bonds, then odd bonds), then a pooling gate (controlled pair of single-qubit rotations)
from each even active qubit onto its odd neighbour, which stays active. The last active qubit is read
out in the Z basis. Parameters: 15 (convolution) + 4 (pooling) per layer, i.e. O(log N) in total.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import torch
from torch import nn

_I = torch.eye(2, dtype=torch.complex128)
_X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
_Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
_Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
_PAULI_2Q = torch.stack(
    [torch.kron(a, b) for a, b in itertools.product([_I, _X, _Y, _Z], repeat=2)][1:]
)  # 15 generators


def schedule(n: int) -> list[tuple[str, int, int, int]]:
    """Gate list ``(kind, layer, qubit_a, qubit_b)``; the readout qubit is ``readout(n)``."""
    gates: list[tuple[str, int, int, int]] = []
    active = list(range(n))
    layer = 0
    while len(active) > 1:
        for start in (0, 1):
            for k in range(start, len(active) - 1, 2):
                gates.append(("conv", layer, active[k], active[k + 1]))
        for k in range(0, len(active) - 1, 2):
            gates.append(("pool", layer, active[k], active[k + 1]))
        active = active[1::2]
        layer += 1
    return gates


def readout(n: int) -> int:
    """Index of the qubit that is measured at the end of the QCNN."""
    active = list(range(n))
    while len(active) > 1:
        active = active[1::2]
    return active[0]


def apply_gate(psi: torch.Tensor, u: torch.Tensor, qa: int, qb: int) -> torch.Tensor:
    """Apply a 4x4 unitary to axes ``qa+1`` and ``qb+1`` (axis 0 is the batch)."""
    out = torch.tensordot(u.reshape(2, 2, 2, 2), psi, dims=([2, 3], [qa + 1, qb + 1]))
    return torch.movedim(out, (0, 1), (qa + 1, qb + 1))


def _depolarise_qubit(rho: torch.Tensor, q: int, n: int) -> torch.Tensor:
    """Completely depolarise qubit ``q`` of a density tensor with axes (batch, rows.., cols..)."""
    ra, ca = q + 1, n + q + 1
    tr = rho.narrow(ra, 0, 1).narrow(ca, 0, 1) + rho.narrow(ra, 1, 1).narrow(ca, 1, 1)
    out = torch.zeros_like(rho)
    out.narrow(ra, 0, 1).narrow(ca, 0, 1).copy_(tr / 2)
    out.narrow(ra, 1, 1).narrow(ca, 1, 1).copy_(tr / 2)
    return out


class QCNN(nn.Module):
    """Translation-invariant QCNN with ``log2(n)`` layers."""

    def __init__(
        self,
        n_qubits: int,
        use_pool: bool = True,
        seed: int = 0,
        dtype: torch.dtype = torch.complex128,
    ) -> None:
        super().__init__()
        layers = int(round(math.log2(n_qubits)))
        if 2**layers != n_qubits:
            raise ValueError("n_qubits must be a power of two")
        self.n, self.layers, self.use_pool, self.cdtype = n_qubits, layers, use_pool, dtype
        self.rdtype = torch.float64 if dtype == torch.complex128 else torch.float32
        gen = torch.Generator().manual_seed(seed)
        self.conv = nn.Parameter(0.5 * torch.randn(layers, 15, generator=gen, dtype=self.rdtype))
        pool = 0.5 * torch.randn(layers, 4, generator=gen, dtype=self.rdtype)
        self.pool = nn.Parameter(pool if use_pool else torch.zeros_like(pool), requires_grad=use_pool)
        self.gates = schedule(n_qubits)
        self.out_qubit = readout(n_qubits)
        self.paulis = _PAULI_2Q.to(dtype)

    @property
    def n_params(self) -> int:
        """Number of trainable parameters."""
        return self.conv.numel() + (self.pool.numel() if self.use_pool else 0)

    def _unitaries(self) -> dict[tuple[str, int], torch.Tensor]:
        c = self.cdtype
        mats: dict[tuple[str, int], torch.Tensor] = {}
        for layer in range(self.layers):
            gen = torch.einsum("k,kab->ab", self.conv[layer].to(c), self.paulis)
            mats["conv", layer] = torch.linalg.matrix_exp(-0.5j * gen)
            p = self.pool[layer]
            blocks = []
            for a, b in ((p[0], p[1]), (p[2], p[3])):
                rz = torch.diag(torch.exp(-0.5j * torch.stack([a, -a]).to(c)))
                cb, sb = torch.cos(b / 2).to(c), torch.sin(b / 2).to(c)
                rx = torch.stack([torch.stack([cb, -1j * sb]), torch.stack([-1j * sb, cb])])
                blocks.append(rz @ rx)
            v = torch.zeros(4, 4, dtype=c)
            v[:2, :2], v[2:, 2:] = blocks[0], blocks[1]
            mats["pool", layer] = v
        return mats

    def _z_expectation(self, probs: torch.Tensor) -> torch.Tensor:
        dims = tuple(d for d in range(1, self.n + 1) if d != self.out_qubit + 1)
        marg = probs.sum(dim=dims)
        return marg[:, 0] - marg[:, 1]

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Noiseless ``<Z>`` of the readout qubit for a batch of pure states ``(B, 2**n)``."""
        psi = states.to(self.cdtype).reshape(-1, *([2] * self.n))
        mats = self._unitaries()
        for kind, layer, qa, qb in self.gates:
            psi = apply_gate(psi, mats[kind, layer], qa, qb)
        return self._z_expectation(psi.abs() ** 2)

    @torch.no_grad()
    def noisy_forward(self, states: torch.Tensor, p2: float) -> torch.Tensor:
        """``<Z>`` when every two-qubit gate is followed by a depolarising channel of strength ``p2``.

        With probability ``p2`` the pair state is replaced by the maximally mixed state.
        """
        n = self.n
        psi = states.to(self.cdtype)
        batch = psi.shape[0]
        ket = psi.reshape(batch, *([2] * n), *([1] * n))
        bra = psi.conj().reshape(batch, *([1] * n), *([2] * n))
        rho = ket * bra
        mats = self._unitaries()
        for kind, layer, qa, qb in self.gates:
            u = mats[kind, layer]
            rho = apply_gate(rho, u, qa, qb)
            rho = apply_gate(rho, u.conj(), n + qa, n + qb)
            if p2 > 0:
                mixed = _depolarise_qubit(_depolarise_qubit(rho, qa, n), qb, n)
                rho = (1 - p2) * rho + p2 * mixed
        dim = 2**n
        diag = torch.diagonal(rho.reshape(batch, dim, dim), dim1=1, dim2=2).real
        return self._z_expectation(diag.reshape(batch, *([2] * n)))

    def to_qiskit(self):
        """Return the circuit as a Qiskit ``QuantumCircuit`` of two-qubit unitary gates."""
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import UnitaryGate

        qc = QuantumCircuit(self.n)
        with torch.no_grad():
            mats = {k: v.numpy() for k, v in self._unitaries().items()}
        for kind, layer, qa, qb in self.gates:
            # Qiskit is little-endian: the first listed qubit is the least significant one.
            u = mats[kind, layer].astype(np.complex128)
            left, _, right = np.linalg.svd(u)  # remove single-precision round-off
            u = (left @ right).reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)
            qc.append(UnitaryGate(u), [qa, qb])
        return qc


def count_compiled_cx(model: QCNN, line: bool = True) -> dict[str, int]:
    """Transpile to {cx, rz, sx, x} (optionally onto a line) and return CX count and depth."""
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap

    cmap = CouplingMap.from_line(model.n) if line else None
    qc = transpile(
        model.to_qiskit(),
        basis_gates=["cx", "rz", "sx", "x"],
        coupling_map=cmap,
        optimization_level=1,
        seed_transpiler=0,
    )
    ops = qc.count_ops()
    return {
        "n_qubits": model.n,
        "logical_gates": len(model.gates),
        "cx": int(ops.get("cx", 0)),
        "depth": int(qc.depth()),
    }


def as_tensor(states: np.ndarray, dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    """Convert real/complex numpy states to a torch tensor."""
    return torch.as_tensor(np.asarray(states), dtype=dtype)
