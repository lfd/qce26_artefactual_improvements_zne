"""
Zero-Noise Extrapolation (ZNE) toolkit.
=======================================

Provides gate-level **folding** (from-left, from-right, global) and
multiple **extrapolation** methods (linear, polynomial, exponential,
Richardson).

The key Richardson helpers ``lagrange_coefficients`` and ``sigma_ci``
are the canonical implementation used across all experiment scripts.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit


# ═════════════════════════════════════════════════════════════════════════
# 1.  Gate-level folding
# ═════════════════════════════════════════════════════════════════════════

def fold_circuit(
    circuit: QuantumCircuit,
    scale_factor: float,
    strategy: str = "from_left",
) -> QuantumCircuit:
    """Noise-scale a circuit by gate-level or global folding.

    Parameters
    ----------
    circuit : QuantumCircuit
        Already-transpiled circuit in basis gates.
    scale_factor : float  (≥ 1)
        Desired noise amplification factor.  Must be ≥ 1.
        For gate-level strategies the effective scale is
        ``1 + 2k`` where ``k`` gates receive one extra G†G pair.
    strategy : {'from_left', 'from_right', 'global', 'random'}
        Folding direction.

    Returns
    -------
    QuantumCircuit
    """
    if scale_factor < 1.0:
        raise ValueError(f"scale_factor must be ≥ 1, got {scale_factor}")
    if abs(scale_factor - 1.0) < 1e-12:
        return circuit.copy()

    dispatchers = {
        "from_left": _fold_gate_level_left,
        "from_right": _fold_gate_level_right,
        "global": _fold_global,
        "random": _fold_gate_level_random,
    }
    if strategy not in dispatchers:
        raise ValueError(
            f"Unknown folding strategy '{strategy}'. "
            f"Choose from {list(dispatchers)}."
        )
    return dispatchers[strategy](circuit, scale_factor)


# ── gate-level: from left ────────────────────────────────────────────────

def _fold_gate_level_left(
    circuit: QuantumCircuit,
    scale_factor: float,
) -> QuantumCircuit:
    """Each gate G → G (G†G)^k.  Extra folds go to the *first* gates."""
    ops = _extract_ops(circuit)
    n_gates = len(ops)
    if n_gates == 0:
        return circuit.copy()

    # How many G†G pairs does every gate get?
    n_full = int((scale_factor - 1) // 2)
    # How many gates get one *additional* pair?
    total_desired = int(round(scale_factor * n_gates))
    gates_after_full = n_gates * (1 + 2 * n_full)
    n_extra = max(0, (total_desired - gates_after_full) // 2)
    n_extra = min(n_extra, n_gates)

    folded = QuantumCircuit(circuit.num_qubits)
    for i, (op, qubits) in enumerate(ops):
        folded.append(op, qubits)
        k = n_full + (1 if i < n_extra else 0)
        for _ in range(k):
            folded.append(op.inverse(), qubits)
            folded.append(op, qubits)
    return folded


# ── gate-level: from right ───────────────────────────────────────────────

def _fold_gate_level_right(
    circuit: QuantumCircuit,
    scale_factor: float,
) -> QuantumCircuit:
    """Same as from-left, but extra folds go to the *last* gates."""
    ops = _extract_ops(circuit)
    n_gates = len(ops)
    if n_gates == 0:
        return circuit.copy()

    n_full = int((scale_factor - 1) // 2)
    total_desired = int(round(scale_factor * n_gates))
    gates_after_full = n_gates * (1 + 2 * n_full)
    n_extra = max(0, (total_desired - gates_after_full) // 2)
    n_extra = min(n_extra, n_gates)

    # Extra folds go to the LAST n_extra gates
    threshold = n_gates - n_extra

    folded = QuantumCircuit(circuit.num_qubits)
    for i, (op, qubits) in enumerate(ops):
        folded.append(op, qubits)
        k = n_full + (1 if i >= threshold else 0)
        for _ in range(k):
            folded.append(op.inverse(), qubits)
            folded.append(op, qubits)
    return folded


# ── gate-level: random ───────────────────────────────────────────────────

def _fold_gate_level_random(
    circuit: QuantumCircuit,
    scale_factor: float,
    seed: int = 42,
) -> QuantumCircuit:
    """Same as from-left, but extra folds go to *randomly* selected gates.

    Uses a fixed seed for reproducibility within a single call, but the
    seed can be varied externally for different noise realisations.
    """
    ops = _extract_ops(circuit)
    n_gates = len(ops)
    if n_gates == 0:
        return circuit.copy()

    n_full = int((scale_factor - 1) // 2)
    total_desired = int(round(scale_factor * n_gates))
    gates_after_full = n_gates * (1 + 2 * n_full)
    n_extra = max(0, (total_desired - gates_after_full) // 2)
    n_extra = min(n_extra, n_gates)

    # Randomly select which gates get the extra fold
    rng = np.random.RandomState(seed)
    extra_indices = set(rng.choice(n_gates, size=n_extra, replace=False))

    folded = QuantumCircuit(circuit.num_qubits)
    for i, (op, qubits) in enumerate(ops):
        folded.append(op, qubits)
        k = n_full + (1 if i in extra_indices else 0)
        for _ in range(k):
            folded.append(op.inverse(), qubits)
            folded.append(op, qubits)
    return folded


# ── global folding ───────────────────────────────────────────────────────

def _fold_global(
    circuit: QuantumCircuit,
    scale_factor: float,
) -> QuantumCircuit:
    r"""U → U (U†U)^k, optionally with a partial tail.

    For scale_factor = 2k+1  (odd integer):  exact.
    For non-integer scales: full global folds + gate-level partial fold
    from left on the remaining U†U pair.
    """
    ops = _extract_ops(circuit)
    n_gates = len(ops)
    if n_gates == 0:
        return circuit.copy()

    n_full = int((scale_factor - 1) // 2)
    remaining = scale_factor - (1 + 2 * n_full)

    inv_ops = [(op.inverse(), qubits) for op, qubits in reversed(ops)]

    folded = QuantumCircuit(circuit.num_qubits)

    # Original U
    for op, qubits in ops:
        folded.append(op, qubits)

    # Full (U†U) pairs
    for _ in range(n_full):
        for op, qubits in inv_ops:
            folded.append(op, qubits)
        for op, qubits in ops:
            folded.append(op, qubits)

    # Partial pair if needed (Giurgica-Tiron Eq. 11: fold last s layers)
    if remaining > 0.01:
        total_pair = 2 * n_gates          # one U† + one U
        n_partial = int(round(remaining * total_pair / 2))
        n_partial = min(n_partial, n_gates)
        # partial U† (last n_partial gates, reversed order)
        for idx, (op, qubits) in enumerate(inv_ops):
            if idx >= n_partial:
                break
            folded.append(op, qubits)
        # partial U (last n_partial gates, forward order)
        # ops[-n_partial:] gives the last n_partial gates
        tail_ops = ops[-n_partial:] if n_partial > 0 else []
        for op, qubits in tail_ops:
            folded.append(op, qubits)

    return folded


# ── helpers ──────────────────────────────────────────────────────────────

def _extract_ops(circuit: QuantumCircuit):
    """Return list of (operation, qubit_indices) tuples, skipping barriers."""
    ops = []
    for inst in circuit.data:
        if inst.operation.name in ("barrier", "measure",
                                    "save_density_matrix",
                                    "save_statevector"):
            continue
        qubits = [circuit.qubits.index(q) for q in inst.qubits]
        ops.append((inst.operation, qubits))
    return ops


# ═════════════════════════════════════════════════════════════════════════
# 2.  Extrapolation methods
# ═════════════════════════════════════════════════════════════════════════

def extrapolate(
    scales: Sequence[float],
    values: Sequence[float],
    method: str = "linear",
) -> float:
    """Extrapolate noisy expectation values to zero noise.

    Parameters
    ----------
    scales : sequence of float
        Noise scale factors (λ₁, λ₂, …).
    values : sequence of float
        Expectation values at each scale.
    method : {'linear', 'polynomial', 'exponential', 'richardson'}
        Extrapolation model.

    Returns
    -------
    float : estimated zero-noise expectation value.
    """
    s = np.asarray(scales, dtype=float)
    v = np.asarray(values, dtype=float)

    if method == "linear":
        p = np.polyfit(s, v, 1)
        return float(np.polyval(p, 0.0))

    if method == "polynomial":
        deg = min(len(s) - 1, 2)
        p = np.polyfit(s, v, deg)
        return float(np.polyval(p, 0.0))

    if method == "exponential":
        return _extrapolate_exponential(s, v)

    if method == "richardson":
        c = lagrange_coefficients(s.tolist())
        return float(c @ v)

    raise ValueError(f"Unknown extrapolation method '{method}'.")


def _extrapolate_exponential(s: np.ndarray, v: np.ndarray) -> float:
    """Fit  y = a exp(b x) + c  and return y(0) = a + c."""
    from scipy.optimize import curve_fit

    def _model(x, a, b, c):
        return a * np.exp(b * x) + c

    # Sensible initial guess
    a0 = v[0] - v[-1]
    b0 = -0.5
    c0 = v[-1]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                _model, s, v,
                p0=[a0, b0, c0],
                maxfev=10_000,
            )
        return float(_model(0.0, *popt))
    except (RuntimeError, ValueError):
        # Fall back to linear if the fit fails
        p = np.polyfit(s, v, 1)
        return float(np.polyval(p, 0.0))


# ── Richardson coefficients ──────────────────────────────────────────────

def lagrange_coefficients(scales: list[float]) -> np.ndarray:
    """Lagrange interpolation weights evaluated at x = 0.

    c_k  =  ∏_{j≠k}  (0 - λ_j) / (λ_k - λ_j)
         =  ∏_{j≠k}  λ_j / (λ_j - λ_k)

    Returns an array of length ``len(scales)``.
    """
    n = len(scales)
    lam = np.asarray(scales, dtype=float)
    c = np.ones(n, dtype=float)
    for k in range(n):
        for j in range(n):
            if j != k:
                c[k] *= lam[j] / (lam[j] - lam[k])
    return c


def sigma_ci(scales: list[float]) -> float:
    """Noise amplification factor  Σ|c_i|  for Richardson extrapolation."""
    return float(np.sum(np.abs(lagrange_coefficients(scales))))
