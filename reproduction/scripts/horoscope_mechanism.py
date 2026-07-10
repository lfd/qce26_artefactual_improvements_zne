#!/usr/bin/env python3
"""
The Horoscope Effect — Garbage Folding Demonstration
=====================================================

This script reproduces the two Horoscope experiments and the Σ|cᵢ|
spectrum analysis reported in the paper:

  (A) Grover Horoscope: 6-qubit Grover circuit with garbage folding
      across 20 noise levels.  Demonstrates that garbage Richardson
      produces "improvements" that are MORE statistically significant
      than genuine ZNE (d=20.8 vs d=15.8).

  (B) QFT Mirror Horoscope: 6-qubit QFT mirror circuit
      (Russo et al. 2023), confirming the effect generalises
      beyond Grover search.

  (C) Σ|cᵢ| Spectrum: Same Grover circuit with three scale factor
      configurations (Kim, Hour, Kandala), showing garbage folding
      dominance grows monotonically with Σ|cᵢ|.

Backends:
  --backend simulator   Qiskit Aer with depolarizing noise (default)
  --backend fake        IBM FakeBrisbane (Eagle r3 calibration data)
  --backend ibm         Real IBM Quantum hardware via Qiskit Runtime

Usage:
  python horoscope_mechanism.py --backend simulator
  python horoscope_mechanism.py --backend ibm --token <IBM_TOKEN>

References:
  [1] Govia et al., "An operational definition of quantum error
      mitigation" (2025). arXiv:2407.02471
  [2] Giurgica-Tiron et al., "Digital zero noise extrapolation for
      quantum error mitigation" (2020). doi:10.1109/QCE49297.2020.00045
  [3] Russo et al., "Testing platform-independent quantum error
      mitigation on noisy quantum computers" (2023). arXiv:2210.07194
"""

import argparse
import csv
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats as sp_stats

# Qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

warnings.filterwarnings("ignore", category=DeprecationWarning)

# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)

# Default output directory; overridden by --outdir in main().
RESULTS_DIR = os.path.join(".", "build", "results")


# =============================================================================
# EXPERIMENTAL PARAMETERS
# =============================================================================
@dataclass
class ExperimentConfig:
    """All configurable parameters in one place for reproducibility."""

    # Grover circuit
    n_qubits: int = 6
    r_opt: int = 6  # Grover iterations

    # Noise sweep (20 log-spaced levels)
    p_2q_values: np.ndarray = None  # set in __post_init__
    p_1q_ratio: float = 0.1  # p_1q = ratio * p_2q

    # Scale factors for main experiment (Kim et al.)
    scale_factors: list = None  # set in __post_init__

    # Shot-based validation
    n_shots: int = 4096
    n_reps: int = 200
    n_garbage_seeds: int = 5

    # Σ|cᵢ| spectrum configurations
    spectrum_configs: dict = None  # set in __post_init__
    spectrum_p_2q: float = 1e-3  # fixed noise for spectrum

    # Random seed for reproducibility
    master_seed: int = 42

    def __post_init__(self):
        self.p_2q_values = np.geomspace(5e-5, 5e-2, 20)
        self.scale_factors = [1, 3, 5]
        self.spectrum_configs = {
            "Kim": [1, 3, 5],
            "Hour": [1.0, 1.5, 2.0, 2.5],
            "Kandala": [1.0, 1.1, 1.25, 1.5],
        }

    @property
    def n_states(self) -> int:
        return 2**self.n_qubits

    @property
    def noise_floor(self) -> float:
        return 1.0 / self.n_states

    @property
    def p_theoretical(self) -> float:
        theta = math.asin(1 / math.sqrt(self.n_states))
        return math.sin((2 * self.r_opt + 1) * theta) ** 2


CFG = ExperimentConfig()


# =============================================================================
# LAGRANGE / RICHARDSON COEFFICIENTS
# =============================================================================
def lagrange_coefficients(scales: list[float]) -> np.ndarray:
    """Compute Richardson (Lagrange interpolation at λ=0) coefficients.

    For scale factors {λ₁, ..., λ_K}, the coefficient for point k is:
        c_k = ∏_{j≠k} λ_j / (λ_j - λ_k)

    Source: Standard numerical analysis; see Cai et al. (2023), Eq. 11
    for the QEM-specific sampling overhead C_em ~ (Σ|cᵢ|)².
    """
    x = np.array(scales, float)
    n = len(x)
    coeffs = np.zeros(n)
    for i in range(n):
        c = 1.0
        for j in range(n):
            if i != j:
                c *= (0.0 - x[j]) / (x[i] - x[j])
        coeffs[i] = c
    return coeffs


def sigma_ci(scales: list[float]) -> float:
    """Σ|cᵢ|: sum of absolute Richardson coefficients."""
    return float(np.sum(np.abs(lagrange_coefficients(scales))))


# =============================================================================
# CIRCUIT CONSTRUCTION
# =============================================================================
def build_grover(n_qubits: int = 6, r_opt: int = 6) -> QuantumCircuit:
    """6-qubit Grover circuit searching for |111111⟩ with r_opt iterations.

    parameters chosen to match Kim et al. (2025).
    """
    target = "1" * n_qubits
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    for _ in range(r_opt):
        # Oracle for |111111⟩
        for i in range(n_qubits):
            if target[i] == "0":
                qc.x(i)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        for i in range(n_qubits):
            if target[i] == "0":
                qc.x(i)
        # Diffusion operator
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))
    return qc


def build_qft_mirror(n_qubits: int = 6) -> QuantumCircuit:
    """6-qubit QFT mirror: |000001⟩ → QFT → QFT† → |000001⟩.

    Transpiled to cx/sx/rz/x basis for accurate noise counting.
    Representative of mirror-circuit benchmarks (Russo et al. 2023).
    """
    qc = QuantumCircuit(n_qubits)
    qc.x(0)  # prepare |000001⟩
    qft_circ = QFT(n_qubits, do_swaps=True).decompose()
    qc.compose(qft_circ, inplace=True)
    qc.compose(qft_circ.inverse(), inplace=True)
    sim = AerSimulator(method="density_matrix")
    return transpile(
        qc, sim, optimization_level=1, basis_gates=["cx", "id", "rz", "sx", "x"]
    )


def build_trotter(n_qubits: int = 4, n_steps: int = 1,
                  rx_angle: float = 0.3, rz_angle: float = 0.6) -> QuantumCircuit:
    """4-qubit Trotter circuit (Khan et al. 2024, 1 step = 18 CX after transpile).

    Simulates first-order Trotterisation of a transverse-field Ising chain.
    This is the circuit used in Layer 1 (parameter-space sensitivity).
    """
    qc = QuantumCircuit(n_qubits)
    for _ in range(n_steps):
        for q in range(n_qubits):
            qc.rx(rx_angle, q)
        for i in range(0, n_qubits - 1, 2):
            qc.cx(i, i + 1)
            qc.rz(rz_angle, i + 1)
            qc.cx(i, i + 1)
        for i in range(1, n_qubits - 1, 2):
            qc.cx(i, i + 1)
            qc.rz(rz_angle, i + 1)
            qc.cx(i, i + 1)
    sim = AerSimulator(method="density_matrix")
    return transpile(
        qc, sim, optimization_level=1, basis_gates=["cx", "id", "rz", "sx", "x"]
    )


# =============================================================================
# FOLDING STRATEGIES
# =============================================================================
def fold_genuine(qc: QuantumCircuit, scale_factor) -> QuantumCircuit:
    """Gate-level unitary folding: G → G·(G†·G)^k.

    Standard digital ZNE noise amplification (Giurgica-Tiron et al. 2020).
    Supports arbitrary scale factors ≥ 1:
      - λ=1: no folding (original circuit)
      - λ=3: each gate folded once  → G·G†·G  (3× gates)
      - λ=5: each gate folded twice → G·(G†·G)² (5× gates)
      - Non-integer λ: full folds on all gates + partial fold-from-left

    Total gate count = round(scale_factor × d).
    """
    if scale_factor <= 1.0:
        return qc.copy()
    ops = [(inst.operation, list(inst.qubits)) for inst in qc.data]
    d = len(ops)
    # Total gate count desired
    total_desired = int(round(scale_factor * d))
    # Full folds every gate gets: floor((λ-1)/2)
    num_full_folds = int((scale_factor - 1.0) // 2)
    used = d * (1 + 2 * num_full_folds)
    # Remaining gates get one extra fold (from-left), 2 extra gates each
    n_extra = (total_desired - used) // 2
    n_extra = max(0, min(n_extra, d))
    folded = QuantumCircuit(qc.num_qubits)
    for i, (op, qubits) in enumerate(ops):
        folded.append(op, qubits)
        k = num_full_folds + (1 if i < n_extra else 0)
        for _ in range(k):
            folded.append(op.inverse(), qubits)
            folded.append(op, qubits)
    return folded


def fold_garbage(
    qc: QuantumCircuit, scale_factor, rng: np.random.Generator
) -> QuantumCircuit:
    """Garbage folding: replace G†·G pairs with random gates.

    Matches exact gate count of fold_genuine, so depolarizing noise
    exposure is IDENTICAL.  But noiseless: computes U·V_random ≠ U.
    Supports arbitrary scale factors (multi-fold per gate).
    """
    if scale_factor <= 1.0:
        return qc.copy()
    ops = [(inst.operation, list(inst.qubits)) for inst in qc.data]
    d = len(ops)
    total_desired = int(round(scale_factor * d))
    num_full_folds = int((scale_factor - 1.0) // 2)
    used = d * (1 + 2 * num_full_folds)
    n_extra = (total_desired - used) // 2
    n_extra = max(0, min(n_extra, d))
    n = qc.num_qubits
    one_q_gates = ["h", "x", "y", "z", "s", "sdg"]
    folded = QuantumCircuit(n)
    for i, (op, qubits) in enumerate(ops):
        folded.append(op, qubits)
        k = num_full_folds + (1 if i < n_extra else 0)
        nq = op.num_qubits
        for _ in range(k):
            for _ in range(2):  # two random gates per fold (matching G†, G)
                if nq >= 2:
                    qs = rng.choice(n, 2, replace=False)
                    folded.cx(int(qs[0]), int(qs[1]))
                else:
                    q = int(rng.integers(n))
                    gate = rng.choice(one_q_gates)
                    getattr(folded, gate)(q)
    return folded


# =============================================================================
# BACKEND ABSTRACTION
# =============================================================================
class SimulatorBackend:
    """Qiskit Aer with custom depolarizing noise model."""

    def __init__(self):
        self.name = "aer_simulator"

    def build_noise_model(self, p_1q: float, p_2q: float) -> NoiseModel:
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(
            depolarizing_error(p_1q, 1),
            ["h", "x", "y", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "sx", "id"],
        )
        nm.add_all_qubit_quantum_error(
            depolarizing_error(p_2q, 2), ["cx", "ecr", "cz", "mcx"]
        )
        return nm

    def _run_density_matrix(
        self, circuit: QuantumCircuit, p_1q: float, p_2q: float
    ) -> np.ndarray:
        """Full density-matrix diagonal via Aer simulation."""
        nm = self.build_noise_model(p_1q, p_2q)
        qc = circuit.copy()
        qc.save_density_matrix()
        sim = AerSimulator(method="density_matrix", noise_model=nm)
        tc = transpile(qc, sim, optimization_level=0)
        result = sim.run(tc).result()
        dm = np.asarray(result.data()["density_matrix"])
        return np.real(np.diag(dm))

    def exact_prob(
        self, circuit: QuantumCircuit, p_1q: float, p_2q: float, target_idx: int
    ) -> float:
        """P(target) via density-matrix simulation (no shot noise)."""
        diag = self._run_density_matrix(circuit, p_1q, p_2q)
        return float(diag[target_idx])

    def exact_distribution(
        self, circuit: QuantumCircuit, p_1q: float, p_2q: float
    ) -> np.ndarray:
        """Full probability distribution (all 2^n states) via density matrix."""
        return self._run_density_matrix(circuit, p_1q, p_2q)

    def shot_sample(
        self, exact_prob: float, n_shots: int, rng: np.random.Generator
    ) -> float:
        """Simulate shot noise via binomial sampling from exact probability."""
        return rng.binomial(n_shots, np.clip(exact_prob, 0, 1)) / n_shots


class FakeBackend:
    """IBM FakeBrisbane (Eagle r3) for realistic noise profiles."""

    def __init__(self):
        try:
            from qiskit_ibm_runtime.fake_provider import FakeBrisbane

            self.fake = FakeBrisbane()
            self.name = "fake_brisbane"
        except ImportError:
            print("WARNING: qiskit-ibm-runtime not installed. Falling back to simulator.")
            self.fake = None
            self.name = "aer_simulator_fallback"

    def build_noise_model(self, p_1q: float, p_2q: float) -> NoiseModel:
        if self.fake is not None:
            from qiskit_aer.noise import NoiseModel

            return NoiseModel.from_backend(self.fake)
        # Fallback
        return SimulatorBackend().build_noise_model(p_1q, p_2q)

    def _run_density_matrix(
        self, circuit: QuantumCircuit, p_1q: float, p_2q: float
    ) -> np.ndarray:
        """Full density-matrix diagonal via FakeBackend noise model."""
        nm = self.build_noise_model(p_1q, p_2q)
        qc = circuit.copy()
        qc.save_density_matrix()
        sim_dm = AerSimulator(method="density_matrix", noise_model=nm)
        tc_dm = transpile(qc, sim_dm, optimization_level=0)
        result = sim_dm.run(tc_dm).result()
        dm = np.asarray(result.data()["density_matrix"])
        return np.real(np.diag(dm))

    def exact_prob(
        self, circuit: QuantumCircuit, p_1q: float, p_2q: float, target_idx: int
    ) -> float:
        diag = self._run_density_matrix(circuit, p_1q, p_2q)
        return float(diag[target_idx])

    def exact_distribution(
        self, circuit: QuantumCircuit, p_1q: float, p_2q: float
    ) -> np.ndarray:
        """Full probability distribution (all 2^n states) via density matrix."""
        return self._run_density_matrix(circuit, p_1q, p_2q)

    def shot_sample(
        self, exact_prob: float, n_shots: int, rng: np.random.Generator
    ) -> float:
        return rng.binomial(n_shots, np.clip(exact_prob, 0, 1)) / n_shots


class IBMBackend:
    """Real IBM Quantum hardware via Qiskit Runtime."""

    def __init__(self, token: Optional[str] = None, instance: str = "ibm-q/open/main"):
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService

            if token:
                self.service = QiskitRuntimeService(channel="ibm_quantum", token=token)
            else:
                self.service = QiskitRuntimeService()
            self.backend = self.service.least_busy(
                simulator=False, min_num_qubits=CFG.n_qubits
            )
            self.name = self.backend.name
            print(f"  IBM Backend: {self.name}")
        except Exception as e:
            print(f"ERROR: Could not connect to IBM Quantum: {e}")
            sys.exit(1)

    def run_circuit(
        self, circuit: QuantumCircuit, n_shots: int, target_idx: int
    ) -> float:
        """Run on real hardware, return P(target)."""
        from qiskit_ibm_runtime import SamplerV2

        qc = circuit.copy()
        qc.measure_all()
        tc = transpile(qc, self.backend, optimization_level=2)

        sampler = SamplerV2(self.backend)
        job = sampler.run([tc], shots=n_shots)
        result = job.result()
        counts = result[0].data.meas.get_counts()

        target_bitstring = format(target_idx, f"0{circuit.num_qubits}b")
        return counts.get(target_bitstring, 0) / n_shots


def get_backend(backend_name: str, token: Optional[str] = None):
    """Factory function for backend selection."""
    if backend_name == "simulator":
        return SimulatorBackend()
    elif backend_name == "fake":
        return FakeBackend()
    elif backend_name == "ibm":
        return IBMBackend(token=token)
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


# =============================================================================
# EXTRAPOLATION
# =============================================================================
def extrap_richardson(scales: list, values: list) -> float:
    """Richardson extrapolation (Lagrange interpolation at λ=0).

    Returns UNCLIPPED value to preserve variance information.
    """
    coeffs = lagrange_coefficients(scales)
    return float(np.dot(coeffs, values))


def extrap_linear(scales: list, values: list) -> float:
    """Linear fit, evaluate at λ=0."""
    p = np.polyfit(scales, values, 1)
    return float(np.polyval(p, 0.0))


def negativity_check(
    distributions_at_lambdas: list[np.ndarray],
    scales: list,
) -> dict:
    """Richardson-extrapolate the FULL probability distribution and diagnose
    negativity as an indicator of unphysical / unreliable extrapolation.

    Parameters
    ----------
    distributions_at_lambdas : list[np.ndarray]
        One array per scale factor, each of length 2^n (full diagonal of ρ).
    scales : list
        The noise scale factors [1, λ₁, λ₂, ...].

    Returns
    -------
    dict with keys:
        P_extrap          : np.ndarray — extrapolated distribution (length 2^n)
        n_negative         : int       — number of states with P < 0
        negativity_frac    : float     — fraction of states with P < 0
        negativity_L1      : float     — sum of |P| for negative entries (L1 negativity)
        P_sum              : float     — sum of extrapolated distribution (should ≈ 1)
        P_min              : float     — most negative value
    """
    coeffs = lagrange_coefficients(scales)
    # Stack distributions: shape (n_lambdas, n_states)
    D = np.array(distributions_at_lambdas)
    # Richardson: P_extrap[s] = Σ cᵢ · P_λᵢ[s]  for each state s
    P_extrap = coeffs @ D  # shape (n_states,)

    neg_mask = P_extrap < 0
    n_neg = int(np.sum(neg_mask))
    n_states = len(P_extrap)

    return {
        "P_extrap": P_extrap,
        "n_negative": n_neg,
        "negativity_frac": n_neg / n_states,
        "negativity_L1": float(np.sum(np.abs(P_extrap[neg_mask]))),
        "P_sum": float(np.sum(P_extrap)),
        "P_min": float(np.min(P_extrap)),
    }


# =============================================================================
# PART A: GROVER HOROSCOPE SWEEP
# =============================================================================
def run_grover_horoscope(backend) -> list[dict]:
    """
        Sweep noise levels with genuine and garbage ZNE on Grover circuit.
    """
    print("\n" + "=" * 72)
    print("  PART A: Grover Horoscope — Noise Sweep (20 levels)")
    print("=" * 72)

    grover = build_grover(CFG.n_qubits, CFG.r_opt)
    target_idx = int("1" * CFG.n_qubits, 2)  # |111111⟩ = 63
    sf = CFG.scale_factors
    coeffs = lagrange_coefficients(sf)
    sci = sigma_ci(sf)

    print(f"  Circuit: {grover.size()} gates, depth {grover.depth()}")
    print(f"  P_theoretical = {CFG.p_theoretical:.5f}")
    print(f"  Scale factors: {sf}, Σ|cᵢ| = {sci:.1f}")
    print(f"  c₁ = {coeffs[0]:.3f}")

    # Verify gate count matching
    for s in sf:
        fc = fold_genuine(grover, s)
        fg = fold_garbage(grover, s, np.random.default_rng(42))
        match = "✓" if fc.size() == fg.size() else "✗"
        print(f"  λ={s}: genuine={fc.size()} gates, garbage={fg.size()} {match}")

    # Pre-build garbage circuits
    garbage_circuits = {}
    for seed in range(CFG.n_garbage_seeds):
        rng = np.random.default_rng(seed * 1000 + CFG.master_seed)
        garbage_circuits[seed] = {s: fold_garbage(grover, s, rng) for s in sf}

    results = []
    t0 = time.time()

    for ip, p_2q in enumerate(CFG.p_2q_values):
        p_1q = p_2q * CFG.p_1q_ratio

        E_noisy = backend.exact_prob(grover, p_1q, p_2q, target_idx)

        # Genuine ZNE
        E_genuine = [backend.exact_prob(fold_genuine(grover, s), p_1q, p_2q, target_idx)
                     for s in sf]
        E_gen_rich = extrap_richardson(sf, E_genuine)
        E_gen_lin = extrap_linear(sf, E_genuine)

        # Genuine negativity check (full distribution)
        gen_dists = [backend.exact_distribution(fold_genuine(grover, s), p_1q, p_2q)
                     for s in sf]
        neg_gen = negativity_check(gen_dists, sf)

        # Garbage ZNE (average over seeds)
        E_garb_rich_seeds = []
        E_garb_lin_seeds = []
        neg_garb_agg = {"n_negative": [], "negativity_frac": [], "negativity_L1": [], "P_min": []}
        for seed in range(CFG.n_garbage_seeds):
            E_garbage = [backend.exact_prob(garbage_circuits[seed][s], p_1q, p_2q, target_idx)
                         for s in sf]
            E_garb_rich_seeds.append(extrap_richardson(sf, E_garbage))
            E_garb_lin_seeds.append(extrap_linear(sf, E_garbage))
            # Garbage negativity check
            garb_dists = [backend.exact_distribution(garbage_circuits[seed][s], p_1q, p_2q)
                          for s in sf]
            ng = negativity_check(garb_dists, sf)
            for k in neg_garb_agg:
                neg_garb_agg[k].append(ng[k])
        E_garb_rich = float(np.mean(E_garb_rich_seeds))
        E_garb_lin = float(np.mean(E_garb_lin_seeds))
        neg_garb = {k: float(np.mean(v)) for k, v in neg_garb_agg.items()}

        # Analytical prediction (Eq. 5 in paper)
        E_predicted = coeffs[0] * E_noisy + (1 - coeffs[0]) * CFG.noise_floor

        results.append({
            "circuit": "grover",
            "p_2q": p_2q,
            "E_noisy": E_noisy,
            "E_genuine_rich": E_gen_rich,
            "E_genuine_lin": E_gen_lin,
            "E_garbage_rich": E_garb_rich,
            "E_garbage_lin": E_garb_lin,
            "E_analytical": E_predicted,
            "E_genuine_at_lambdas": E_genuine,
            "neg_genuine_frac": neg_gen["negativity_frac"],
            "neg_genuine_L1": neg_gen["negativity_L1"],
            "neg_genuine_n": neg_gen["n_negative"],
            "neg_genuine_Pmin": neg_gen["P_min"],
            "neg_garbage_frac": neg_garb["negativity_frac"],
            "neg_garbage_L1": neg_garb["negativity_L1"],
            "neg_garbage_n": neg_garb["n_negative"],
            "neg_garbage_Pmin": neg_garb["P_min"],
        })

        elapsed = time.time() - t0
        eta = elapsed / (ip + 1) * (len(CFG.p_2q_values) - ip - 1)
        imp = "✓" if E_garb_rich > E_noisy else "✗"
        n_states = 2 ** CFG.n_qubits
        print(f"  [{ip+1:2d}/20] p_2q={p_2q:.5f} | noisy={E_noisy:.4f} | "
              f"gen_R={E_gen_rich:.4f} | garb_R={E_garb_rich:.4f} {imp} | "
              f"neg: gen={neg_gen['n_negative']}/{n_states} "
              f"garb={neg_garb['n_negative']:.0f}/{n_states} | ETA {eta:.0f}s")
        sys.stdout.flush()

    print(f"  Sweep done in {time.time() - t0:.1f}s")
    return results


# =============================================================================
# PART B: QFT MIRROR HOROSCOPE
# =============================================================================
def run_qft_mirror_horoscope(backend) -> list[dict]:
    """Same sweep on QFT mirror circuit (Russo et al. 2023)."""
    print("\n" + "=" * 72)
    print("  PART B: QFT Mirror Horoscope — Noise Sweep (20 levels)")
    print("=" * 72)

    circ = build_qft_mirror(CFG.n_qubits)
    target_idx = 1  # |000001⟩
    sf = CFG.scale_factors
    coeffs = lagrange_coefficients(sf)

    n_cx = sum(1 for inst in circ.data if inst.operation.name == "cx")
    print(f"  Circuit: {circ.size()} gates ({n_cx} CX), depth {circ.depth()}")
    print(f"  Ideal P(|000001⟩) = 1.0")
    print(f"  Scale factors: {sf}, Σ|cᵢ| = {sigma_ci(sf):.1f}")

    # Pre-build garbage circuits
    garbage_circuits = {}
    for seed in range(CFG.n_garbage_seeds):
        rng = np.random.default_rng(seed * 1000 + CFG.master_seed + 100)
        garbage_circuits[seed] = {s: fold_garbage(circ, s, rng) for s in sf}

    results = []
    t0 = time.time()

    for ip, p_2q in enumerate(CFG.p_2q_values):
        p_1q = p_2q * CFG.p_1q_ratio

        E_noisy = backend.exact_prob(circ, p_1q, p_2q, target_idx)

        E_genuine = [backend.exact_prob(fold_genuine(circ, s), p_1q, p_2q, target_idx)
                     for s in sf]
        E_gen_rich = extrap_richardson(sf, E_genuine)

        # Genuine negativity check
        gen_dists = [backend.exact_distribution(fold_genuine(circ, s), p_1q, p_2q)
                     for s in sf]
        neg_gen = negativity_check(gen_dists, sf)

        E_garb_rich_seeds = []
        neg_garb_agg = {"n_negative": [], "negativity_frac": [], "negativity_L1": [], "P_min": []}
        for seed in range(CFG.n_garbage_seeds):
            E_garbage = [backend.exact_prob(garbage_circuits[seed][s], p_1q, p_2q, target_idx)
                         for s in sf]
            E_garb_rich_seeds.append(extrap_richardson(sf, E_garbage))
            # Garbage negativity check
            garb_dists = [backend.exact_distribution(garbage_circuits[seed][s], p_1q, p_2q)
                          for s in sf]
            ng = negativity_check(garb_dists, sf)
            for k in neg_garb_agg:
                neg_garb_agg[k].append(ng[k])
        E_garb_rich = float(np.mean(E_garb_rich_seeds))
        neg_garb = {k: float(np.mean(v)) for k, v in neg_garb_agg.items()}

        E_predicted = coeffs[0] * E_noisy + (1 - coeffs[0]) * CFG.noise_floor

        results.append({
            "circuit": "qft_mirror",
            "p_2q": p_2q,
            "E_noisy": E_noisy,
            "E_genuine_rich": E_gen_rich,
            "E_garbage_rich": E_garb_rich,
            "E_analytical": E_predicted,
            "neg_genuine_frac": neg_gen["negativity_frac"],
            "neg_genuine_L1": neg_gen["negativity_L1"],
            "neg_genuine_n": neg_gen["n_negative"],
            "neg_genuine_Pmin": neg_gen["P_min"],
            "neg_garbage_frac": neg_garb["negativity_frac"],
            "neg_garbage_L1": neg_garb["negativity_L1"],
            "neg_garbage_n": neg_garb["n_negative"],
            "neg_garbage_Pmin": neg_garb["P_min"],
        })

        elapsed = time.time() - t0
        eta = elapsed / (ip + 1) * (len(CFG.p_2q_values) - ip - 1)
        imp = "✓" if E_garb_rich > E_noisy else "✗"
        n_states = 2 ** CFG.n_qubits
        print(f"  [{ip+1:2d}/20] p_2q={p_2q:.5f} | noisy={E_noisy:.4f} | "
              f"gen_R={E_gen_rich:.4f} | garb_R={E_garb_rich:.4f} {imp} | "
              f"neg: gen={neg_gen['n_negative']}/{n_states} "
              f"garb={neg_garb['n_negative']:.0f}/{n_states} | ETA {eta:.0f}s")
        sys.stdout.flush()

    n_imp = sum(1 for r in results if r["E_garbage_rich"] > r["E_noisy"])
    print(f"  Done. Garbage improves in {n_imp}/{len(results)} configs.")
    return results


# =============================================================================
# PART C: SHOT-BASED VALIDATION (for Table 7 in paper)
# =============================================================================
def run_shot_validation(backend, grover_results: list[dict]) -> dict:
    """N=200 shot-based reps at p_2q=0.001, comparing genuine vs garbage.

    This produces the data for Table 7:
      genuine Richardson: d=15.8, p<10^{-240}
      garbage Richardson: d=20.8, p<10^{-264}
    """
    print("\n" + "=" * 72)
    print(f"  PART C: Shot-Based Validation (N={CFG.n_reps}, {CFG.n_shots} shots)")
    print("=" * 72)

    # Find exact values at p_2q closest to 0.001
    idx = int(np.argmin(np.abs(CFG.p_2q_values - 0.001)))
    ref = grover_results[idx]
    p_2q = ref["p_2q"]
    E_noisy = ref["E_noisy"]
    E_genuine = ref["E_genuine_at_lambdas"]

    # For garbage: λ=1 uses real circuit, λ>1 collapses to noise floor
    # (validated by exact sweep above)

    print(f"  p_2q = {p_2q:.5f}")
    print(f"  E_noisy = {E_noisy:.5f}")
    print(f"  E_genuine(λ) = {[f'{v:.5f}' for v in E_genuine]}")

    rng = np.random.default_rng(12345)
    sf = CFG.scale_factors

    raw_trials = np.zeros(CFG.n_reps)
    strategies = {
        "genuine_richardson": np.zeros(CFG.n_reps),
        "genuine_linear": np.zeros(CFG.n_reps),
        "garbage_richardson": np.zeros(CFG.n_reps),
        "garbage_linear": np.zeros(CFG.n_reps),
    }

    for rep in range(CFG.n_reps):
        # Raw
        raw_trials[rep] = backend.shot_sample(E_noisy, CFG.n_shots, rng)

        # Genuine ZNE with shot noise
        gen_shots = [backend.shot_sample(e, CFG.n_shots, rng) for e in E_genuine]
        strategies["genuine_richardson"][rep] = extrap_richardson(sf, gen_shots)
        strategies["genuine_linear"][rep] = extrap_linear(sf, gen_shots)

        # Garbage ZNE: λ=1 is real, λ>1 is noise floor
        garb_shots = [backend.shot_sample(E_noisy, CFG.n_shots, rng)]
        for _ in sf[1:]:
            garb_shots.append(backend.shot_sample(CFG.noise_floor, CFG.n_shots, rng))
        strategies["garbage_richardson"][rep] = extrap_richardson(sf, garb_shots)
        strategies["garbage_linear"][rep] = extrap_linear(sf, garb_shots)

    # Compute statistics
    shot_results = {}
    print(f"\n  {'Strategy':<25s} {'Mean':>8s} {'t':>8s} {'p':>12s} "
          f"{'d':>7s} {'%impr':>6s}")
    print("  " + "-" * 70)

    for name, mit_vals in strategies.items():
        improvement = mit_vals - raw_trials
        t_stat, p_val = sp_stats.ttest_rel(mit_vals, raw_trials)
        d_cohen = np.mean(improvement) / (np.std(improvement, ddof=1) + 1e-15)
        frac_better = np.mean(mit_vals > raw_trials) * 100

        shot_results[name] = {
            "mean_raw": float(np.mean(raw_trials)),
            "mean_mit": float(np.mean(mit_vals)),
            "std_mit": float(np.std(mit_vals, ddof=1)),
            "t_stat": float(t_stat),
            "p_val": float(p_val),
            "cohen_d": float(d_cohen),
            "frac_better": float(frac_better),
        }

        p_str = f"{p_val:.2e}" if p_val > 1e-300 else "< 1e-300"
        print(f"  {name:<25s} {np.mean(mit_vals):8.4f} {t_stat:8.1f} "
              f"{p_str:>12s} {d_cohen:7.1f} {frac_better:5.0f}%")

    return shot_results


# =============================================================================
# PART D: Σ|cᵢ| SPECTRUM (for Figure 8 and Table 8)
# =============================================================================
def run_sigma_spectrum(backend) -> dict:
    """Compare three scale factor configs on same Grover circuit."""
    print("\n" + "=" * 72)
    print("  PART D: Σ|cᵢ| Spectrum Experiment")
    print(f"  Fixed noise: p_2q = {CFG.spectrum_p_2q}")
    print("=" * 72)

    grover = build_grover(CFG.n_qubits, CFG.r_opt)
    target_idx = int("1" * CFG.n_qubits, 2)
    p_1q = CFG.spectrum_p_2q * CFG.p_1q_ratio
    p_2q = CFG.spectrum_p_2q

    E_noisy = backend.exact_prob(grover, p_1q, p_2q, target_idx)
    # Noiseless ideal for ρ computation
    E_noisy_no_noise = backend.exact_prob(grover, 0, 0, target_idx)
    print(f"  E_noisy = {E_noisy:.5f}")
    print(f"  E_ideal (noiseless) = {E_noisy_no_noise:.5f}")

    spectrum_results = {}
    rng_shot = np.random.default_rng(12345)

    for name, scales in CFG.spectrum_configs.items():
        coeffs = lagrange_coefficients(scales)
        sci = float(np.sum(np.abs(coeffs)))
        c1 = coeffs[0]

        print(f"\n  Config: {name} (scales={scales}, Σ|cᵢ|={sci:.1f}, c₁={c1:.3f})")

        # Exact genuine E(λ)
        gen_exact = [backend.exact_prob(fold_genuine(grover, s), p_1q, p_2q, target_idx)
                     for s in scales]

        # Exact garbage E(λ), averaged over seeds
        garb_per_seed = []
        for gs in range(CFG.n_garbage_seeds):
            rng = np.random.default_rng(CFG.master_seed + gs)
            garb = [backend.exact_prob(fold_garbage(grover, s, rng), p_1q, p_2q, target_idx)
                    for s in scales]
            garb_per_seed.append(garb)
        garb_exact = np.mean(garb_per_seed, axis=0).tolist()

        print(f"    E_genuine(λ) = {[f'{v:.5f}' for v in gen_exact]}")
        print(f"    E_garbage(λ) = {[f'{v:.5f}' for v in garb_exact]}")

        # Shot-based simulation
        raw_vals = np.array([rng_shot.binomial(CFG.n_shots, E_noisy) / CFG.n_shots
                             for _ in range(CFG.n_reps)])

        gen_rich = np.zeros(CFG.n_reps)
        garb_rich = np.zeros(CFG.n_reps)
        gen_rich_unclipped = np.zeros(CFG.n_reps)
        garb_rich_unclipped = np.zeros(CFG.n_reps)

        for rep in range(CFG.n_reps):
            g_shots = [rng_shot.binomial(CFG.n_shots, e) / CFG.n_shots for e in gen_exact]
            r_val = float(np.dot(coeffs, g_shots))
            gen_rich_unclipped[rep] = r_val
            gen_rich[rep] = np.clip(r_val, 0.0, 1.0)

            g_shots = [rng_shot.binomial(CFG.n_shots, e) / CFG.n_shots for e in garb_exact]
            r_val = float(np.dot(coeffs, g_shots))
            garb_rich_unclipped[rep] = r_val
            garb_rich[rep] = np.clip(r_val, 0.0, 1.0)

        # Statistics
        def paired_stats(mit, raw):
            imp = mit - raw
            t, p = sp_stats.ttest_rel(mit, raw)
            d = np.mean(imp) / (np.std(imp, ddof=1) + 1e-15)
            return {"t": float(t), "p": float(p), "d": float(d),
                    "mean_imp": float(np.mean(imp)),
                    "frac_better": float(np.mean(mit > raw))}

        stats_gen = paired_stats(gen_rich, raw_vals)
        stats_garb = paired_stats(garb_rich, raw_vals)

        var_raw = float(np.var(raw_vals))
        var_gen_unc = float(np.var(gen_rich_unclipped))
        var_ratio = var_gen_unc / var_raw if var_raw > 0 else float("inf")

        # Compute ρ from exact values
        E_mit_gen_exact = float(np.dot(coeffs, gen_exact))
        E_mit_garb_exact = float(np.dot(coeffs, garb_exact))
        E_raw_exact = gen_exact[0]  # E(λ=1)
        denom = E_noisy_no_noise - E_raw_exact  # ideal - raw
        # For Grover: E_ideal ≈ p_theoretical (set in CFG)
        E_ideal = CFG.p_theoretical  # same Grover circuit for all spectrum configs
        denom = E_ideal - E_raw_exact
        rho_gen = (E_mit_gen_exact - E_raw_exact) / denom if abs(denom) > 1e-12 else float("nan")
        rho_garb = (E_mit_garb_exact - E_raw_exact) / denom if abs(denom) > 1e-12 else float("nan")

        spectrum_results[name] = {
            "scales": scales,
            "sigma_ci": sci,
            "c1": float(c1),
            "gen_exact": gen_exact,
            "garb_exact": garb_exact,
            "E_mit_gen": E_mit_gen_exact,
            "E_mit_garb": E_mit_garb_exact,
            "rho_gen": rho_gen,
            "rho_garb": rho_garb,
            "stats_gen": stats_gen,
            "stats_garb": stats_garb,
            "var_ratio": var_ratio,
            "sum_ci_sq": float(np.sum(coeffs**2)),
        }

        p_gen = f"{stats_gen['p']:.2e}" if stats_gen["p"] > 1e-300 else "< 1e-300"
        p_garb = f"{stats_garb['p']:.2e}" if stats_garb["p"] > 1e-300 else "< 1e-300"
        print(f"    Genuine:  Δ={stats_gen['mean_imp']*100:+.1f}pp, d={stats_gen['d']:+.1f}, "
              f"p={p_gen}, %better={stats_gen['frac_better']*100:.0f}%")
        print(f"    Garbage:  Δ={stats_garb['mean_imp']*100:+.1f}pp, d={stats_garb['d']:+.1f}, "
              f"p={p_garb}, %better={stats_garb['frac_better']*100:.0f}%")
        print(f"    Var amp.: {var_ratio:.1f}× (theory Σcᵢ² = {np.sum(coeffs**2):.1f})")

    return spectrum_results


# =============================================================================
# PART E: MULTI-CIRCUIT TABLE (for Table and Figure in paper)
# =============================================================================
def compute_trotter_ideal(n_qubits: int = 4, n_steps: int = 3,
                          rx_angle: float = 0.3, rz_angle: float = 0.6) -> float:
    """Compute ideal ⟨Z⊗Z⊗Z⊗Z⟩ for the Trotter circuit via statevector."""
    qc = build_trotter(n_qubits, n_steps, rx_angle, rz_angle)
    qc.save_statevector()
    sim = AerSimulator(method="statevector")
    tc = transpile(qc, sim, optimization_level=0)
    result = sim.run(tc).result()
    sv = np.asarray(result.data()["statevector"])
    probs = np.abs(sv) ** 2
    # ⟨ZZZZ⟩ = Σ_s (-1)^{popcount(s)} · P(s)
    n = n_qubits
    zzzz = 0.0
    for s in range(2**n):
        sign = (-1) ** bin(s).count("1")
        zzzz += sign * probs[s]
    return float(zzzz)


def run_multi_circuit_table(backend) -> list[dict]:
    """
    Run genuine + garbage ZNE on three circuits at p_2q = 10^{-3},
    computing ρ for the paper's multi-circuit comparison table and figure.

    Circuits:
      1. Grover 6q (r=6): deep, signal destroyed at λ>1
      2. QFT Mirror 6q: medium depth, signal retained
      3. Trotter 4q (1 step): shallow, signal mostly retained
    """
    print("\n" + "=" * 72)
    print("  PART E: Multi-Circuit Table (ρ values for paper)")
    print("=" * 72)

    p_2q = CFG.spectrum_p_2q
    p_1q = p_2q * CFG.p_1q_ratio
    sf = CFG.scale_factors  # [1, 3, 5]
    coeffs = lagrange_coefficients(sf)

    # Define circuits
    circuits = []

    # 1) Grover 6q — transpile so CX count is accurate
    grover_raw = build_grover(CFG.n_qubits, CFG.r_opt)
    sim_transpile = AerSimulator(method="density_matrix")
    grover = transpile(
        grover_raw, sim_transpile, optimization_level=1,
        basis_gates=["cx", "id", "rz", "sx", "x"]
    )
    grover_target = int("1" * CFG.n_qubits, 2)
    n_cx_grover = sum(1 for inst in grover.data if inst.operation.name == "cx")
    circuits.append({
        "name": "Grover 6q",
        "circuit": grover,
        "n_qubits": 6,
        "target_idx": grover_target,
        "E_ideal": CFG.p_theoretical,
        "n_cx": n_cx_grover,
        "observable": "P(target)",
    })

    # 2) QFT Mirror 6q
    qft = build_qft_mirror(6)
    n_cx_qft = sum(1 for inst in qft.data if inst.operation.name == "cx")
    circuits.append({
        "name": "QFT Mirror 6q",
        "circuit": qft,
        "n_qubits": 6,
        "target_idx": 1,  # |000001⟩
        "E_ideal": 1.0,
        "n_cx": n_cx_qft,
        "observable": "P(target)",
    })

    # 3) Trotter 4q (TC1: 1 step = 6 CX, matching Khan et al.)
    trotter = build_trotter(4, 1)
    trotter_ideal = compute_trotter_ideal(4, 1)
    n_cx_trotter = sum(1 for inst in trotter.data if inst.operation.name == "cx")

    # Trotter uses ⟨ZZZZ⟩ not P(target), so we need special handling
    circuits.append({
        "name": "Trotter 4q",
        "circuit": trotter,
        "n_qubits": 4,
        "target_idx": None,  # special: uses ⟨ZZZZ⟩
        "E_ideal": trotter_ideal,
        "n_cx": n_cx_trotter,
        "observable": "ZZZZ",
    })

    results = []

    for ci in circuits:
        name = ci["name"]
        circ = ci["circuit"]
        nq = ci["n_qubits"]
        noise_floor = 1.0 / (2 ** nq)

        print(f"\n  {name}: {ci['n_cx']} CX, {nq}q, 1/2^n = {noise_floor:.4f}")

        if ci["observable"] == "P(target)":
            target_idx = ci["target_idx"]

            # Genuine E(λ)
            E_gen = [backend.exact_prob(fold_genuine(circ, s), p_1q, p_2q, target_idx)
                     for s in sf]

            # Garbage E(λ), averaged over seeds
            garb_per_seed = []
            for gs in range(CFG.n_garbage_seeds):
                rng = np.random.default_rng(CFG.master_seed + gs)
                garb = [backend.exact_prob(fold_garbage(circ, s, rng), p_1q, p_2q, target_idx)
                        for s in sf]
                garb_per_seed.append(garb)
            E_garb = np.mean(garb_per_seed, axis=0).tolist()

        else:
            # Trotter: compute ⟨ZZZZ⟩ from full distribution
            def compute_zzzz(circuit, p1, p2):
                diag = backend.exact_distribution(circuit, p1, p2)
                zzzz = 0.0
                for s in range(len(diag)):
                    sign = (-1) ** bin(s).count("1")
                    zzzz += sign * diag[s]
                return float(zzzz)

            E_gen = [compute_zzzz(fold_genuine(circ, s), p_1q, p_2q) for s in sf]

            garb_per_seed = []
            for gs in range(CFG.n_garbage_seeds):
                rng = np.random.default_rng(CFG.master_seed + gs)
                garb = [compute_zzzz(fold_garbage(circ, s, rng), p_1q, p_2q)
                        for s in sf]
                garb_per_seed.append(garb)
            E_garb = np.mean(garb_per_seed, axis=0).tolist()

        # Richardson extrapolation
        E_mit_gen = float(np.dot(coeffs, E_gen))
        E_mit_garb = float(np.dot(coeffs, E_garb))

        E_raw = E_gen[0]  # E(λ=1) is the raw (unmitigated) value
        E_ideal = ci["E_ideal"]

        rho_gen = (E_mit_gen - E_raw) / (E_ideal - E_raw) if abs(E_ideal - E_raw) > 1e-12 else float("nan")
        rho_garb = (E_mit_garb - E_raw) / (E_ideal - E_raw) if abs(E_ideal - E_raw) > 1e-12 else float("nan")

        row = {
            "circuit": name,
            "n_qubits": nq,
            "n_cx": ci["n_cx"],
            "observable": ci["observable"],
            "E_ideal": E_ideal,
            "noise_floor": noise_floor,
            "E1": E_gen[0],
            "E3": E_gen[1],
            "E5": E_gen[2],
            "E1_garb": E_garb[0],
            "E3_garb": E_garb[1],
            "E5_garb": E_garb[2],
            "E_mit_gen": E_mit_gen,
            "E_mit_garb": E_mit_garb,
            "rho_gen": rho_gen,
            "rho_garb": rho_garb,
        }
        results.append(row)

        print(f"    E_genuine(λ) = [{E_gen[0]:.5f}, {E_gen[1]:.5f}, {E_gen[2]:.5f}]")
        print(f"    E_garbage(λ) = [{E_garb[0]:.5f}, {E_garb[1]:.5f}, {E_garb[2]:.5f}]")
        print(f"    E_mit_gen = {E_mit_gen:.5f},  E_mit_garb = {E_mit_garb:.5f}")
        print(f"    ρ_gen = {rho_gen:.4f},  ρ_garb = {rho_garb:.4f}")

    return results


# =============================================================================
# CSV OUTPUT
# =============================================================================
def save_results(
    grover_results: list[dict],
    mirror_results: list[dict],
    shot_results: dict,
    spectrum_results: dict,
    circuit_results: list[dict],
    backend_name: str,
):
    """Save all results to CSV files."""

    # --- Sweep results (Grover + QFT mirror) ---
    path = os.path.join(RESULTS_DIR, "horoscope_sweep.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["circuit", "backend", "p_2q", "E_noisy",
                     "E_genuine_rich", "E_garbage_rich", "E_analytical",
                     "neg_genuine_frac", "neg_genuine_L1", "neg_genuine_n",
                     "neg_genuine_Pmin",
                     "neg_garbage_frac", "neg_garbage_L1", "neg_garbage_n",
                     "neg_garbage_Pmin"])
        for r in grover_results:
            w.writerow(["grover", backend_name, f"{r['p_2q']:.6f}",
                         f"{r['E_noisy']:.6f}", f"{r['E_genuine_rich']:.6f}",
                         f"{r['E_garbage_rich']:.6f}", f"{r['E_analytical']:.6f}",
                         f"{r.get('neg_genuine_frac', 0):.4f}",
                         f"{r.get('neg_genuine_L1', 0):.6f}",
                         f"{r.get('neg_genuine_n', 0)}",
                         f"{r.get('neg_genuine_Pmin', 0):.6f}",
                         f"{r.get('neg_garbage_frac', 0):.4f}",
                         f"{r.get('neg_garbage_L1', 0):.6f}",
                         f"{r.get('neg_garbage_n', 0):.0f}",
                         f"{r.get('neg_garbage_Pmin', 0):.6f}"])
        for r in mirror_results:
            w.writerow(["qft_mirror", backend_name, f"{r['p_2q']:.6f}",
                         f"{r['E_noisy']:.6f}", f"{r['E_genuine_rich']:.6f}",
                         f"{r['E_garbage_rich']:.6f}", f"{r['E_analytical']:.6f}",
                         f"{r.get('neg_genuine_frac', 0):.4f}",
                         f"{r.get('neg_genuine_L1', 0):.6f}",
                         f"{r.get('neg_genuine_n', 0)}",
                         f"{r.get('neg_genuine_Pmin', 0):.6f}",
                         f"{r.get('neg_garbage_frac', 0):.4f}",
                         f"{r.get('neg_garbage_L1', 0):.6f}",
                         f"{r.get('neg_garbage_n', 0):.0f}",
                         f"{r.get('neg_garbage_Pmin', 0):.6f}"])
    print(f"  Saved: {path}")

    # --- Shot-based validation ---
    path = os.path.join(RESULTS_DIR, "horoscope_shots.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "backend", "mean_raw", "mean_mit", "std_mit",
                     "t_stat", "p_value", "cohen_d", "frac_better_pct"])
        for name, s in shot_results.items():
            w.writerow([name, backend_name,
                         f"{s['mean_raw']:.6f}", f"{s['mean_mit']:.6f}",
                         f"{s['std_mit']:.6f}", f"{s['t_stat']:.2f}",
                         f"{s['p_val']:.2e}", f"{s['cohen_d']:.2f}",
                         f"{s['frac_better']:.1f}"])
    print(f"  Saved: {path}")

    # --- Σ|cᵢ| spectrum ---
    path = os.path.join(RESULTS_DIR, "horoscope_spectrum.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "backend", "sigma_ci", "c1",
                     "E_mit_gen", "E_mit_garb", "rho_gen", "rho_garb",
                     "genuine_improvement_pp", "genuine_cohen_d", "genuine_p",
                     "genuine_frac_better",
                     "garbage_improvement_pp", "garbage_cohen_d", "garbage_p",
                     "garbage_frac_better",
                     "variance_ratio", "sum_ci_sq"])
        for name, r in spectrum_results.items():
            sg = r["stats_gen"]
            sgarb = r["stats_garb"]
            w.writerow([name, backend_name,
                         f"{r['sigma_ci']:.1f}", f"{r['c1']:.3f}",
                         f"{r['E_mit_gen']:.6f}", f"{r['E_mit_garb']:.6f}",
                         f"{r['rho_gen']:.4f}", f"{r['rho_garb']:.4f}",
                         f"{sg['mean_imp']*100:.3f}", f"{sg['d']:.3f}",
                         f"{sg['p']:.2e}", f"{sg['frac_better']:.3f}",
                         f"{sgarb['mean_imp']*100:.3f}", f"{sgarb['d']:.3f}",
                         f"{sgarb['p']:.2e}", f"{sgarb['frac_better']:.3f}",
                         f"{r['var_ratio']:.2f}", f"{r['sum_ci_sq']:.2f}"])
    print(f"  Saved: {path}")

    # --- Multi-circuit table ---
    if circuit_results:
        path = os.path.join(RESULTS_DIR, "horoscope_circuits.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["circuit", "backend", "n_qubits", "n_cx", "observable",
                         "E_ideal", "noise_floor",
                         "E1", "E3", "E5",
                         "E1_garb", "E3_garb", "E5_garb",
                         "E_mit_gen", "E_mit_garb",
                         "rho_gen", "rho_garb"])
            for r in circuit_results:
                w.writerow([r["circuit"], backend_name,
                             r["n_qubits"], r["n_cx"], r["observable"],
                             f"{r['E_ideal']:.6f}", f"{r['noise_floor']:.6f}",
                             f"{r['E1']:.6f}", f"{r['E3']:.6f}", f"{r['E5']:.6f}",
                             f"{r['E1_garb']:.6f}", f"{r['E3_garb']:.6f}", f"{r['E5_garb']:.6f}",
                             f"{r['E_mit_gen']:.6f}", f"{r['E_mit_garb']:.6f}",
                             f"{r['rho_gen']:.4f}", f"{r['rho_garb']:.4f}"
                             ])
        print(f"  Saved: {path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Horoscope Effect — Garbage Folding Demonstration"
    )
    parser.add_argument(
        "--backend", choices=["simulator", "fake", "ibm"], default="simulator",
        help="Backend: simulator (Aer), fake (FakeBrisbane), ibm (real hardware)"
    )
    parser.add_argument("--outdir", default=None,
                        help="Output directory for CSV results (default: build/results/)")
    parser.add_argument("--token", default=os.environ.get("IBM_QUANTUM_TOKEN"),
                        help="IBM Quantum token (for --backend ibm)")
    parser.add_argument("--skip-mirror", action="store_true",
                        help="Skip QFT mirror experiment (faster)")
    parser.add_argument("--skip-spectrum", action="store_true",
                        help="Skip Σ|cᵢ| spectrum experiment (faster)")
    parser.add_argument("--skip-circuits", action="store_true",
                        help="Skip multi-circuit table experiment (faster)")
    args = parser.parse_args()

    global RESULTS_DIR
    if args.outdir is not None:
        RESULTS_DIR = args.outdir
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 72)
    print("  The Horoscope Effect — Garbage Folding Demonstration")
    print(f"  Backend: {args.backend}")
    print(f"  N_reps={CFG.n_reps}, N_shots={CFG.n_shots}")
    print("=" * 72)

    backend = get_backend(args.backend, args.token)
    t_total = time.time()

    # Part A: Grover sweep
    grover_results = run_grover_horoscope(backend)

    # Part B: QFT Mirror sweep
    if not args.skip_mirror:
        mirror_results = run_qft_mirror_horoscope(backend)
    else:
        mirror_results = []
        print("\n  [Skipped QFT mirror experiment]")

    # Part C: Shot-based validation
    shot_results = run_shot_validation(backend, grover_results)

    # Part D: Σ|cᵢ| spectrum
    if not args.skip_spectrum:
        spectrum_results = run_sigma_spectrum(backend)
    else:
        spectrum_results = {}
        print("\n  [Skipped Σ|cᵢ| spectrum experiment]")

    # Part E: Multi-circuit table (Grover/QFT/Trotter at p_2q=1e-3)
    if not args.skip_circuits:
        circuit_results = run_multi_circuit_table(backend)
    else:
        circuit_results = []
        print("\n  [Skipped multi-circuit table experiment]")

    # Save all CSV results
    print("\n" + "=" * 72)
    print("  SAVING RESULTS")
    print("=" * 72)
    save_results(grover_results, mirror_results, shot_results,
                 spectrum_results, circuit_results,
                 backend.name if hasattr(backend, "name") else args.backend)

    # Summary
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)

    n_grover_imp = sum(1 for r in grover_results if r["E_garbage_rich"] > r["E_noisy"])
    print(f"  Grover: garbage improves in {n_grover_imp}/{len(grover_results)} configs")

    if mirror_results:
        n_mirror_imp = sum(1 for r in mirror_results if r["E_garbage_rich"] > r["E_noisy"])
        print(f"  QFT Mirror: garbage improves in {n_mirror_imp}/{len(mirror_results)} configs")

    # Negativity diagnostic summary
    all_sweep = grover_results + mirror_results
    if all_sweep and "neg_genuine_frac" in all_sweep[0]:
        n_states = 2 ** CFG.n_qubits
        avg_gen_frac = np.mean([r["neg_genuine_frac"] for r in all_sweep])
        avg_garb_frac = np.mean([r["neg_garbage_frac"] for r in all_sweep])
        avg_gen_L1 = np.mean([r["neg_genuine_L1"] for r in all_sweep])
        avg_garb_L1 = np.mean([r["neg_garbage_L1"] for r in all_sweep])
        worst_gen_Pmin = min(r["neg_genuine_Pmin"] for r in all_sweep)
        worst_garb_Pmin = min(r["neg_garbage_Pmin"] for r in all_sweep)
        print(f"\n  Negativity Diagnostic (extrapolated distribution over {n_states} states):")
        print(f"    Genuine:  avg {avg_gen_frac*100:.1f}% neg. states, "
              f"avg L1={avg_gen_L1:.4f}, worst P_min={worst_gen_Pmin:.4f}")
        print(f"    Garbage:  avg {avg_garb_frac*100:.1f}% neg. states, "
              f"avg L1={avg_garb_L1:.4f}, worst P_min={worst_garb_Pmin:.4f}")
        L1_ratio = avg_garb_L1 / max(avg_gen_L1, 1e-15)
        print(f"    → L1 negativity ratio (garbage/genuine): {L1_ratio:.1f}×")
        print(f"    → Garbage produces {L1_ratio:.0f}× larger unphysical mass")

    if shot_results:
        sr_gen = shot_results.get("genuine_richardson", {})
        sr_garb = shot_results.get("garbage_richardson", {})
        print(f"  Shot validation (Richardson):")
        print(f"    Genuine: d={sr_gen.get('cohen_d', 0):.1f}, "
              f"p={sr_gen.get('p_val', 1):.2e}")
        print(f"    Garbage: d={sr_garb.get('cohen_d', 0):.1f}, "
              f"p={sr_garb.get('p_val', 1):.2e}")

    if spectrum_results:
        print(f"  Σ|cᵢ| Spectrum:")
        for name, r in spectrum_results.items():
            sg = r["stats_gen"]
            sgarb = r["stats_garb"]
            print(f"    {name} (Σ|cᵢ|={r['sigma_ci']:.0f}): "
                  f"genuine d={sg['d']:+.1f}, garbage d={sgarb['d']:+.1f}, "
                  f"var amp={r['var_ratio']:.0f}×, "
                  f"ρ_gen={r['rho_gen']:.3f}, ρ_garb={r['rho_garb']:.3f}")

    if circuit_results:
        print(f"  Multi-circuit table:")
        for r in circuit_results:
            print(f"    {r['circuit']}: {r['n_cx']} CX, "
                  f"ρ_gen={r['rho_gen']:.3f}, ρ_garb={r['rho_garb']:.3f}")

    print(f"\n  Total runtime: {time.time() - t_total:.1f}s")
    print(f"  Results in: {RESULTS_DIR}/")
    print("  Run R/plot_horoscope.R to generate plots.")


if __name__ == "__main__":
    main()
