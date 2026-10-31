import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spt_shotbench.hamiltonian import ground_state, pauli_expectation, string_order  # noqa: E402
from spt_shotbench.baselines import LocalFeatureTables  # noqa: E402
from spt_shotbench.mps import truncate_state  # noqa: E402
from spt_shotbench.qcnn import QCNN, as_tensor, schedule  # noqa: E402
from spt_shotbench.shots import prob_correct  # noqa: E402


def test_cluster_state_energy_and_string_order():
    psi, e = ground_state(8, 0.0)
    assert abs(e + 8.0) < 1e-9
    assert abs(string_order(psi, 8) - 1.0) < 1e-9


def test_paramagnet_limit_has_no_string_order():
    psi, _ = ground_state(8, 50.0)
    assert abs(string_order(psi, 8)) < 0.05
    assert pauli_expectation(psi, 8, {0: "X"}) > 0.95


def test_qcnn_schedule_and_parameter_count():
    assert len(schedule(8)) == 18
    model = QCNN(8)
    assert model.n_params == 3 * 19
    assert QCNN(8, use_pool=False).n_params == 3 * 15


def test_qcnn_output_bounded_and_noise_free_density_matrix_agrees():
    states = as_tensor(np.stack([ground_state(8, 0.4)[0], ground_state(8, 1.6)[0]]))
    model = QCNN(8, seed=3)
    pure = model(states).detach()
    assert torch.all(pure.abs() <= 1 + 1e-9)
    assert torch.allclose(pure, model.noisy_forward(states, 0.0), atol=1e-9)


def test_depolarising_noise_shrinks_the_signal():
    states = as_tensor(np.stack([ground_state(8, 0.2)[0]]))
    model = QCNN(8, seed=5)
    clean = abs(float(model(states)))
    noisy = abs(float(model.noisy_forward(states, 0.3)))
    assert noisy < clean


def test_qiskit_circuit_matches_statevector_simulation():
    from qiskit.quantum_info import Statevector

    model = QCNN(4, seed=2)
    psi = ground_state(4, 0.7)[0]
    expected = float(model(as_tensor(psi[None])))
    # Qiskit uses little-endian qubit order, so reverse the axes of the input state.
    qiskit_in = psi.reshape([2] * 4).transpose(3, 2, 1, 0).reshape(-1)
    out = Statevector(qiskit_in).evolve(model.to_qiskit())
    expectation = out.expectation_value(__import__("qiskit.quantum_info", fromlist=["Pauli"]).Pauli("Z"), [model.out_qubit]).real
    assert abs(expected - expectation) < 1e-8


def test_prob_correct_limits():
    assert prob_correct(0.0, 0.0, 10, 1) == 0.5 or abs(prob_correct(0.0, 0.0, 10, 1) - 0.5) < 1e-12
    assert prob_correct(0.9, 0.0, 1000, 1) > 0.999
    assert prob_correct(0.9, 0.0, 1000, -1) < 0.001
    assert abs(prob_correct(0.3, 0.1, 25, 1) + prob_correct(0.3, 0.1, 25, -1) - 1) < 1e-12


def test_mps_truncation_is_exact_for_large_chi_and_lossy_for_small():
    psi = ground_state(8, 0.95)[0]
    assert abs(abs(truncate_state(psi, 8, 16) @ psi) - 1) < 1e-9
    assert abs(truncate_state(psi, 8, 1) @ psi) < 0.999


def test_baseline_feature_tables_match_pauli_expectations():
    psi = ground_state(8, 0.5)[0]
    exact = LocalFeatureTables(psi, 8).exact()
    assert abs(exact[0] - np.mean([pauli_expectation(psi, 8, {q: "X"}) for q in range(8)])) < 1e-9
    assert abs(exact[3] - string_order(psi, 8)) < 1e-9
