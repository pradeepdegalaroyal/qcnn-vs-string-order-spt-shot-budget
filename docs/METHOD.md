# Method

## Task and data

Periodic spin chain of N qubits (axis q of the state, q = 0 .. N-1):

H = - sum_i Z(i-1) X(i) Z(i+1) - h1 sum_i X(i)   (the XX term of the cluster-Ising family is set to zero)

For h1 < 1 the ground state is in the symmetry-protected-topological (SPT) phase (label +1), for h1 > 1 in the
paramagnet (PM, label -1). Ground states are exact (dense diagonalisation at N = 8, Lanczos at N = 16).

* **Labelled pool:** h1 = 1 -/+ delta with delta = 0.3, 0.4, ..., 1.0 (16 states). Each seed draws a stratified
  training set of 8 states (4 per phase). The critical window |h1 - 1| < 0.3 is never labelled.
* **Test grid:** h1 = 1 -/+ delta with delta = 0.025, 0.075, ..., 0.775 (32 states, disjoint from the pool), labelled by
  the thermodynamic-limit rule h1 < 1. Bins: near (delta < 0.15), mid (0.15 to 0.4), far (>= 0.4); "critical region"
  = near + mid. At N = 8 the finite-size transition is rounded, so the near bin is hard for every method.

## Methods compared (equal state-copy budget S per test state)

1. **Quantum Convolutional Neural Network (QCNN), the standard quantum baseline after Cong et al.** log2(N) layers, each: one shared general two-qubit
   convolution unitary (15 parameters) on even then odd bonds, then a pooling gate (4 parameters) that is a
   pair of Rz-Rx rotations on the surviving qubit, controlled by its discarded neighbour. The readout qubit is measured
   in Z; every copy of the state gives one +-1 outcome. 57 parameters at N = 8, 76 at N = 16. Trained with Adam
   on the mean-square error between the exact readout expectation and the label (2 random restarts, best kept).
   N = 16 is initialised from the N = 8 solution of the same seed and fine-tuned.
2. **QCNN with frozen pooling (ablation).** Pooling parameters fixed at zero (identity); only convolution is trained.
3. **String-order threshold (physics-informed classical-readout baseline).** One measurement setting, the product of
   neighbouring ZXZ stabilisers, Z(0) X(1) X(3) ... X(N-3) Z(N-2), with one threshold set at the midpoint of the class means on the
   training states. Same +-1 outcome per copy as the QCNN.
4. **Learned feature classifier (strong trained classical baseline).** Logistic regression on four features: mean X, mean XX
   (all-X setting), mean ZXZ on even sites (X on even, Z on odd qubits), and the string order. Three
   settings share the copies equally (S/3 each). Trained on exact feature values of the same training states; N = 8 only.
5. **Matrix Product State (MPS) compression (classical-simulation probe, not a classifier).** The test state is cut to bond dimension chi
   (open-chain sequential singular value truncation, chi = 2 .. 32, N = 16) and fed to the trained QCNN;
   we record the change of the readout and of the classification.

All thresholds are fitted on training states only. With S copies the sample mean of the +-1 outcomes is compared
with the threshold; the success probability is computed exactly from the binomial distribution (no Monte Carlo
noise) for the QCNN and the string estimator, and by 40 Monte Carlo repetitions for the feature classifier.

## Metrics

Accuracy per distance bin (mean +- standard deviation over seeds; seeds change the training draw, the circuit
initialisation and the sampled shots); S*, the smallest copy number in {10, 30, 100, 300, 1000, 3000} with seed-mean
critical-region accuracy >= 0.9; mean absolute readout error versus chi; compiled controlled-NOT (CX) count (Qiskit transpilation to
the gate set cx, rz, sx, x, on a line and with all-to-all connectivity).

## Ablation, noise sweep, scaling, seeds

* **Ablation:** pooling trained versus frozen (method 2). A second axis is label scarcity: 2, 4, 8, 16 training states.
* **Noise sweep (N = 8, 100 copies):** every logical two-qubit gate is followed by a depolarising channel that replaces
  the pair with the maximally mixed state with probability p2 in {0, 0.002, 0.005, 0.01, 0.02, 0.05} (exact density-matrix
  simulation); each measured qubit flips with probability r in {0, 0.005, 0.01, 0.02, 0.05}. The QCNN reads one qubit (signal
  factor 1 - 2r), the string reads 5 qubits (factor (1 - 2r)^5). Thresholds are re-fitted under the same noise on the training
  states. The QCNN is trained noiselessly (noise-aware training is future work).
* **Scaling:** N = 8 versus N = 16 for the QCNN and string estimator; chi sweep at N = 16.
* **Seeds:** 5 at N = 8, 3 at N = 16 (circuit simulation of 16 qubits is the cost driver).

## Limits of the design

The logical gates are general two-qubit unitaries, so p2 is a per-logical-gate error; the compiled controlled-NOT (CX) count (reported in
`results/e4_compiled_cost.csv`) converts it to a per-CX error. Single-qubit gate error, crosstalk and state-preparation
error of the ground states are not modelled. The MPS cut is an open-chain cut of a periodic state, which costs about the square of
the periodic bond dimension.
