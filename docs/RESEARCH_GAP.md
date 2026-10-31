# Research gap: what does a Quantum Convolutional Neural Network (QCNN) buy per state copy?

## State of the art (from the supplied literature)

Cong, Choi and Lukin (2019, doi:10.1038/s41567-019-0648-8) introduced the Quantum Convolutional Neural
Network (QCNN): a circuit with O(log N) parameters that recognises a one-dimensional
symmetry-protected-topological (SPT) phase and was reported to surpass existing approaches. Orús (2019,
doi:10.1038/s42254-019-0086-7) reviews the classical counterpart for such states, tensor networks and
in particular Matrix Product States (MPS), whose cost is set by the entanglement across a cut. Gyongyosi and
Imre (2019, doi:10.1016/j.cosrev.2018.11.002) list the open problems of near-term quantum computing, among
them noise and the cost of reading results out. A recent simulation study of quantum key distribution
networks (Cvitić and Gavrilović, 2027, doi:10.1007/978-3-032-40818-1_7) finds that the efficiency of the
single-photon detector, i.e. the readout, is the dominant factor for the usable rate. The remaining papers in the
brief concern materials and devices (see the table) and do not touch learning on quantum states.

## Related work table

| Paper | Year | Role | What it does | What it does NOT cover |
|---|---|---|---|---|
| Cong, Choi, Lukin, doi:10.1038/s41567-019-0648-8 | 2019 | landmark | QCNN with O(log N) parameters for SPT recognition and error-correction design | Shot cost of the readout, comparison with a measurement-matched classical estimator, gate and readout noise, MPS cost |
| Orús, doi:10.1038/s42254-019-0086-7 | 2019 | review | Overview of tensor networks, entanglement and classification of phases | Head-to-head resource accounting against learned quantum circuits |
| Gyongyosi and Imre, doi:10.1016/j.cosrev.2018.11.002 | 2019 | review | Survey of quantum computing technology and open problems | Any concrete benchmark; stays qualitative |
| Cvitić and Gavrilović, doi:10.1007/978-3-032-40818-1_7 | 2027 | recent | Simulation of detector characteristics versus sifted key rate in quantum key distribution networks | Learning tasks; shows only that detection efficiency dominates |
| Chen et al., doi:10.1016/j.optcom.2026.133887 | 2027 | recent | Master-equation simulation of magnon blockade with Kerr and optomechanical nonlinearities | Many-body phase recognition or machine learning |
| Wu et al., doi:10.1088/2631-7990/aeabdc | 2027 | recent | Review of atomic-level manufacturing | Quantum algorithms; relevant only as long-term device context |
| Tran et al., doi:10.1038/nnano.2015.242 | 2016 | landmark | Room-temperature single-photon emission from hexagonal boron nitride | Readout or learning on emitted states |
| Kodigala et al., doi:10.1038/nature20799 | 2017 | landmark | Lasing from photonic bound states in the continuum | Quantum information processing tasks |
| Tsao et al., doi:10.1002/aelm.201600501 | 2018 | landmark | Ultrawide-bandgap semiconductor opportunities | Quantum algorithms |
| Lachance-Quirion et al., doi:10.7567/1882-0786/ab248d | 2019 | review | Hybrid quantum systems based on magnons | Learning or classical simulability |
| Liu and Hersam, doi:10.1038/s41578-019-0136-x | 2019 | review | Two-dimensional materials as platforms for quantum information | Algorithmic benchmarks |
| García de Arquer et al., doi:10.1126/science.aaz8541 | 2021 | review | Semiconductor quantum dot technology and challenges | Algorithmic benchmarks |

## The gap

The QCNN is a landmark proposal, and tensor networks are the accepted classical tool for the same phases, but
no paper in the brief puts the two on one ledger. Three things are missing: (i) the number of **state copies
(shots)** each approach needs near the phase transition, where the signal is smallest; (ii) how that cost changes
once **gate noise and readout noise** are charged, which is where a deep circuit and a long Pauli string
differ; (iii) how large an **MPS bond dimension** a classical simulation of the QCNN input needs near
criticality. The brief's open-problem sentences ask for exactly this kind of cost-aware, simulation-based
validation ("simulation tools ... allow the validation of theoretical models" without expensive infrastructure).

**Novelty in one sentence:** a shot-, noise- and bond-dimension-matched comparison of a trained QCNN with the
cluster-state string order and a four-feature learned Pauli-measurement classifier, resolved by distance
from the critical point.

## Testable hypotheses and success metrics

Setting: periodic cluster-Ising chain, field h1 swept through the critical value h1 = 1; N = 8 and 16 qubits
(exact simulation on a laptop, no hardware); 8 labelled training states; 5 seeds (3 at N = 16).

* **H1 (shot efficiency).** For test states with |h1 - 1| < 0.4, the trained QCNN reaches 90% accuracy with
  fewer state copies than both the string-order threshold and the learned feature classifier.
  Metric: smallest copy number S* in {10, 30, 100, 300, 1000, 3000} with seed-mean accuracy >= 0.9.
  H1 is falsified if either baseline has S* <= the QCNN's.
* **H2 (noise).** The QCNN's advantage over the string estimator shrinks with two-qubit gate error and grows
  with readout error (the string touches 5 qubits at N = 8, the QCNN reads one). Metric: accuracy difference
  at 100 copies on a grid of gate error and readout error; H2 is falsified if the sign pattern is not
  monotone in these two axes.
* **H3 (classical cost).** The MPS bond dimension chi needed to reproduce the QCNN readout grows towards the
  critical point. Metric: mean absolute readout error versus chi per distance bin; H3 is falsified if the
  error in the near bin is not larger than in the far bin at fixed chi.

Scale caveat: all results are exact-simulation results for N <= 16 qubits. Nothing here is a claim of
quantum advantage; the question is the cost ledger, not a speed-up.
