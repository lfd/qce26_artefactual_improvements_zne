"""
Circuit library for QEM reproduction experiments.
=================================================

Provides parameterised quantum circuits used in QEM benchmarking papers,
together with ideal- and noisy-expectation-value helpers.

The flagship circuit is ``build_khan_trotter``, a faithful reconstruction
of the 4-qubit Quantum Trotter Circuit (QTC) described in:

    Khan et al., "Error Mitigation in the NISQ Era: Applying Measurement
    Error Mitigation Techniques to Enhance Quantum Circuit Performance",
    Mathematics 2024, 12(14), 2235.

The paper specifies:
    • 4 qubits
    • Each Trotter step has **13 gates** and **depth 7**
    • Gate composition per step (Algorithm 1):
        - Rx(θ_x) on all 4 qubits                            [4 gates, depth 1]
        - ZZ interaction via CNOT-Rz-CNOT on even pairs (0,1),(2,3)
                                                              [6 gates, depth 3]
        - ZZ interaction via CNOT-Rz-CNOT on odd pair (1,2)
                                                              [3 gates, depth 3]
    • Observable: ⟨Z⊗Z⊗Z⊗Z⟩ (noted as ⟨zzzz⟩ in the paper)
    • Ideal simulator value: 0.8284

The paper does **not** specify:
    - The Hamiltonian parameters (coupling J, field h, evolution time Δt)
    - The ZNE scale factors, folding strategy, or extrapolation method
    - The transpiler optimisation level
    - The noise model details used in simulation
These are the "undocumented parameters" our experiment varies.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ── Trotter circuit construction ─────────────────────────────────────────

def build_khan_trotter(
    n_qubits: int = 4,
    n_steps: int = 1,
    rx_angle: float = 0.3,
    rz_angle: float = 0.6,
) -> QuantumCircuit:
    """Build a Quantum Trotter Circuit matching Khan et al. (2024).

    Simulates first-order Trotterisation of a transverse-field Ising chain
    with open boundary conditions:

        H = −J Σ_{i} Z_i Z_{i+1}  −  h Σ_{i} X_i

    where  rx_angle = 2 h Δt  and  rz_angle = 2 J Δt.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (default 4, as in the paper).
    n_steps : int
        Number of Trotter steps (TC1-TC5 in the paper).
    rx_angle : float
        Single-qubit Rx rotation angle (transverse field term).
    rz_angle : float
        Two-qubit ZZ coupling angle (Ising coupling term).

    Returns
    -------
    QuantumCircuit
        The constructed Trotter circuit (no measurements).

    Notes
    -----
    For 4 qubits the gate count is 13 × n_steps and circuit depth is
    7 × n_steps, matching the paper's Table / Algorithm 1.
    """
    qc = QuantumCircuit(n_qubits)

    for _ in range(n_steps):
        # ── transverse-field rotations ──
        for q in range(n_qubits):
            qc.rx(rx_angle, q)

        # ── ZZ coupling: even qubit pairs (0,1), (2,3), … ──
        for i in range(0, n_qubits - 1, 2):
            qc.cx(i, i + 1)
            qc.rz(rz_angle, i + 1)
            qc.cx(i, i + 1)

        # ── ZZ coupling: odd qubit pairs (1,2), (3,4), … ──
        for i in range(1, n_qubits - 1, 2):
            qc.cx(i, i + 1)
            qc.rz(rz_angle, i + 1)
            qc.cx(i, i + 1)

    return qc


# ── Grover circuit (Desdentado replication) ──────────────────────────────

def build_grover(
    n_qubits: int = 5,
    marked_states: list[str] | None = None,
    n_iterations: int | None = None,
) -> QuantumCircuit:
    """Build a multi-target Grover search circuit.

    Faithful reconstruction of the circuit used by Desdentado et al.
    (2025) — a 5-qubit Grover with M=2 marked states and r_opt=3
    iterations.  The transpiled circuit has ~900 two-qubit gates,
    pushing circuit fidelity to ~0.09 % on ibm_brisbane.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (default 5, as in Desdentado).
    marked_states : list[str] or None
        Bit-strings to search for.  Default ``["00000", "10110"]``
        (M=2, matching Desdentado's 2-solution configuration).
    n_iterations : int or None
        Grover iterations.  If None, uses the optimal count:
        ``r_opt = floor(π / (4 · arcsin(√(M/N))))``.

    Returns
    -------
    QuantumCircuit
        The Grover circuit (no measurements appended).
    """
    if marked_states is None:
        marked_states = ["00000", "10110"]

    M = len(marked_states)
    N = 2 ** n_qubits

    if n_iterations is None:
        n_iterations = int(np.floor(np.pi / (4 * np.arcsin(np.sqrt(M / N)))))

    qc = QuantumCircuit(n_qubits)

    # ── initial superposition ──
    qc.h(range(n_qubits))

    for _ in range(n_iterations):
        # ── Oracle: flip phase of marked states ──
        _grover_oracle(qc, n_qubits, marked_states)
        # ── Diffusion operator: 2|s⟩⟨s| − I ──
        _grover_diffusion(qc, n_qubits)

    return qc


def _grover_oracle(
    qc: QuantumCircuit,
    n_qubits: int,
    marked_states: list[str],
) -> None:
    """Multi-target oracle: flip phase of each marked state."""
    for state in marked_states:
        # Apply X gates to qubits that are '0' in the target state
        for i, bit in enumerate(state):
            if bit == "0":
                qc.x(i)
        # Multi-controlled Z (= H on last, MCX, H on last)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        # Undo X gates
        for i, bit in enumerate(state):
            if bit == "0":
                qc.x(i)


def _grover_diffusion(qc: QuantumCircuit, n_qubits: int) -> None:
    """Grover diffusion operator: 2|s⟩⟨s| − I."""
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))


# ── expectation-value helpers ────────────────────────────────────────────

def _zn_signs(n_qubits: int) -> np.ndarray:
    """(-1)^popcount(x) for x = 0 … 2^n − 1.

    Used to compute ⟨Z⊗…⊗Z⟩ = Σ_x  sign(x) · P(x).
    """
    n_states = 1 << n_qubits
    return np.array(
        [(-1) ** bin(x).count("1") for x in range(n_states)], dtype=float,
    )


def compute_ideal_expectation(circuit: QuantumCircuit) -> float:
    """⟨Z⊗…⊗Z⟩ via noiseless statevector simulation.

    The circuit is run starting from |0…0⟩.
    """
    qc = circuit.copy()
    qc.save_statevector()
    sim = AerSimulator(method="statevector")
    tc = transpile(qc, sim, optimization_level=0)
    result = sim.run(tc).result()
    sv = np.asarray(result.data()["statevector"])
    probs = np.abs(sv) ** 2
    signs = _zn_signs(circuit.num_qubits)
    return float(signs @ probs)


def compute_noisy_expectation(
    circuit: QuantumCircuit,
    noise_model,
) -> float:
    """⟨Z⊗…⊗Z⟩ via density-matrix simulation under *noise_model*.

    No shot noise — returns the exact noisy expectation.
    The circuit should already be in the simulator's basis gates.
    """
    qc = circuit.copy()
    qc.save_density_matrix()
    sim = AerSimulator(method="density_matrix", noise_model=noise_model)
    # optimization_level=0: do NOT re-optimise (would undo folding)
    tc = transpile(qc, sim, optimization_level=0)
    result = sim.run(tc).result()
    dm = np.asarray(result.data()["density_matrix"])
    diag = np.real(np.diag(dm))
    signs = _zn_signs(circuit.num_qubits)
    return float(signs @ diag)


def sample_shot_noise(
    exact_expectation: float,
    n_shots: int,
    rng: np.random.Generator,
) -> float:
    """Simulate shot noise for a ⟨Z⊗…⊗Z⟩ measurement.

    The observable is diagonal in the computational basis, so a
    measurement yields +1 (even parity) or −1 (odd parity).
    The probability of even parity is  P_even = (1 + ⟨ZZ…Z⟩) / 2.
    We draw  n_even ~ Binomial(n_shots, P_even)  and return
    ⟨ZZ…Z⟩_shot = 2 n_even / n_shots − 1.
    """
    p_even = np.clip((1.0 + exact_expectation) / 2.0, 0.0, 1.0)
    n_even = rng.binomial(n_shots, p_even)
    return 2.0 * n_even / n_shots - 1.0


# ── QASM shot-based simulation ──────────────────────────────────────────

def compute_qasm_expectation(
    circuit: QuantumCircuit,
    n_shots: int,
    noise_model=None,
    seed: int = 42,
) -> float:
    """Compute ⟨Z⊗…⊗Z⟩ via shot-based QASM simulation.

    Unlike ``compute_noisy_expectation`` (which returns the *exact* noisy
    expectation via density matrix), this function performs actual
    measurement sampling — matching the behaviour of IBM's QASM Simulator.

    The circuit should already be transpiled and folded.  Only
    measurements are appended; no further gate optimisation is performed.

    Parameters
    ----------
    circuit : QuantumCircuit
        Transpiled + folded circuit (**no** measurements).
    n_shots : int
        Number of measurement shots.
    noise_model : NoiseModel or None
        If None, performs ideal (noiseless) simulation.
    seed : int
        Random seed for shot-noise reproducibility.

    Returns
    -------
    float
        Shot-estimated ⟨Z⊗…⊗Z⟩.
    """
    qc = circuit.copy()
    qc.measure_all()

    kw = {"noise_model": noise_model} if noise_model else {}
    sim = AerSimulator(**kw)

    # optimization_level = 0: preserve the folded gate structure
    tc = transpile(qc, sim, optimization_level=0)
    result = sim.run(tc, shots=n_shots, seed_simulator=seed).result()
    counts = result.get_counts()

    return _expectation_from_counts(counts)


def _expectation_from_counts(counts: dict) -> float:
    """Compute ⟨Z⊗…⊗Z⟩ from measurement bit-string counts.

    Z⊗…⊗Z has eigenvalue (−1)^{popcount(x)} for basis state |x⟩.
    """
    total = 0
    exp_val = 0.0
    for bitstring, count in counts.items():
        bs = bitstring.replace(" ", "")
        parity = (-1) ** bs.count("1")
        exp_val += parity * count
        total += count
    return exp_val / total


# ── angle calibration ────────────────────────────────────────────────────

# ── fast angle calibration (pure numpy — no Qiskit overhead) ─────────

def _fast_zzzz(rx: float, rz: float, n_qubits: int = 4,
               n_steps: int = 1) -> float:
    """Compute ⟨ZZZZ⟩ for a Trotter circuit via direct unitary multiplication.

    Much faster than going through Qiskit/Aer for the calibration scan.

    Parameters
    ----------
    n_steps : int
        Number of Trotter steps (1 for TC1, 3 for TC3, etc.).
    """
    from functools import reduce

    dim = 1 << n_qubits

    # Single-qubit gate matrices
    def rx_mat(theta):
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]])

    def rz_mat(theta):
        return np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)])

    # CNOT matrix (control, target) in the 2-qubit subspace
    cnot_2q = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=complex)

    def eye(n):
        return np.eye(1 << n, dtype=complex)

    def kron_chain(*mats):
        return reduce(np.kron, mats)

    def embed_1q(gate, qubit, nq):
        """Embed a 1-qubit gate on *qubit* into nq-qubit space."""
        parts = [np.eye(2, dtype=complex)] * nq
        parts[qubit] = gate
        return kron_chain(*parts)

    def embed_cnot(ctrl, targ, nq):
        """Embed CNOT(ctrl, targ) into nq-qubit space.

        Uses qubit ordering: qubit 0 = leftmost tensor factor.
        """
        # Build projectors in nq-qubit space
        P0 = np.array([[1, 0], [0, 0]], dtype=complex)
        P1 = np.array([[0, 0], [0, 1]], dtype=complex)
        X  = np.array([[0, 1], [1, 0]], dtype=complex)
        I  = np.eye(2, dtype=complex)

        # |0><0|_ctrl ⊗ I_targ  +  |1><1|_ctrl ⊗ X_targ
        parts_0 = [I] * nq
        parts_0[ctrl] = P0
        parts_1 = [I] * nq
        parts_1[ctrl] = P1
        parts_1[targ] = X
        return kron_chain(*parts_0) + kron_chain(*parts_1)

    nq = n_qubits

    # Start with |0...0⟩
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0

    for _step in range(n_steps):
        # 1. Rx on all qubits
        for q in range(nq):
            U = embed_1q(rx_mat(rx), q, nq)
            state = U @ state

        # 2. Even pairs: CNOT-Rz-CNOT
        for i in range(0, nq - 1, 2):
            state = embed_cnot(i, i + 1, nq) @ state
            state = embed_1q(rz_mat(rz), i + 1, nq) @ state
            state = embed_cnot(i, i + 1, nq) @ state

        # 3. Odd pairs
        for i in range(1, nq - 1, 2):
            state = embed_cnot(i, i + 1, nq) @ state
            state = embed_1q(rz_mat(rz), i + 1, nq) @ state
            state = embed_cnot(i, i + 1, nq) @ state

    probs = np.abs(state) ** 2
    signs = _zn_signs(nq)
    return float(signs @ probs)


def calibrate_angles(
    target: float = 0.8284,
    n_qubits: int = 4,
    n_grid: int = 80,
    trotter_steps: list[int] | None = None,
) -> tuple[float, float]:
    """Find (rx_angle, rz_angle) that gives mean ⟨ZZZZ⟩ ≈ *target* across TCs.

    Khan et al. (2024) state that their ideal simulator yields an expected
    value of 0.8284.  The paper does **not** specify whether this refers to
    TC1 alone or to the average over TC1-TC5.  Cross-checking with Table III
    (R_IS column) shows that the per-TC ideal values are approximately
    0.982 (TC1), 0.931 (TC2), 0.850 (TC3), 0.747 (TC4), 0.631 (TC5)
    ⇒ average ≈ 0.828, confirming that 0.8284 is the **mean** across TCs.

    This function therefore searches for (rx, rz) such that the mean of
    ⟨ZZZZ⟩ computed over *trotter_steps* equals *target*.

    Uses fast numpy-based simulation (no Qiskit overhead).

    Parameters
    ----------
    target : float
        Target mean ⟨ZZZZ⟩ across the Trotter-step depths.
    trotter_steps : list[int] or None
        Trotter depths to average over.  Default [1, 2, 3, 4, 5] (TC1-TC5).
    """
    if trotter_steps is None:
        trotter_steps = [1, 2, 3, 4, 5]

    def _mean_zzzz(rx: float, rz: float) -> float:
        return float(np.mean([
            _fast_zzzz(rx, rz, n_qubits, n_steps=s) for s in trotter_steps
        ]))

    best_rx, best_rz, best_err = 0.1, 0.01, 999.0

    # Coarse grid — small angles (Khan's circuit uses subtle rotations)
    for rx in np.linspace(0.01, 0.5, n_grid):
        for rz in np.linspace(0.01, 0.5, n_grid):
            # Quick pre-filter: check TC1 only first (fast)
            tc1 = _fast_zzzz(float(rx), float(rz), n_qubits, n_steps=1)
            if tc1 < 0.8:  # skip clearly wrong regions
                continue
            val = _mean_zzzz(float(rx), float(rz))
            err = abs(val - target)
            if err < best_err:
                best_rx, best_rz, best_err = float(rx), float(rz), err
                if err < 1e-5:
                    return best_rx, best_rz

    # Fine grid around best
    for rx in np.linspace(best_rx - 0.03, best_rx + 0.03, 60):
        for rz in np.linspace(max(0.001, best_rz - 0.03),
                              best_rz + 0.03, 60):
            val = _mean_zzzz(float(rx), float(rz))
            err = abs(val - target)
            if err < best_err:
                best_rx, best_rz, best_err = float(rx), float(rz), err

    return best_rx, best_rz
