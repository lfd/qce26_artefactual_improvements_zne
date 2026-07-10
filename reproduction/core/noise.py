"""
Noise-model factory for QEM reproduction experiments.
====================================================

Provides ``make_noise_model`` which creates a Qiskit-Aer ``NoiseModel``
for three physically distinct noise channels:

* **depolarizing** — symmetric Pauli channel; the de-facto standard in
  most QEM simulation papers.
* **amplitude_damping** — asymmetric channel modelling energy
  relaxation (T₁ process); biases toward |0⟩.
* **thermal_relaxation** — combined T₁ + T₂ process; the most
  physically realistic of the three.

The point of offering multiple models is to show that ZNE behaviour
depends strongly on the *type* of noise, not only its rate — yet most
papers only report "noise rate" without specifying the channel.
"""

from __future__ import annotations

import numpy as np
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
)

# Gate sets that the noise model attaches errors to.
# NB: 'sxdg' is included because gate-level folding produces G† for
# sx, which is sxdg.  Without it, folded sx† gates would be noiseless.
SINGLE_QUBIT_GATES = [
    "rx", "ry", "rz", "sx", "sxdg", "x", "h", "id", "s", "sdg",
]
TWO_QUBIT_GATES = ["cx", "cz", "ecr"]


# ── public API ───────────────────────────────────────────────────────────

def make_noise_model(
    noise_type: str,
    p_1q: float,
    p_2q: float,
    **kwargs,
) -> NoiseModel:
    """Create a ``NoiseModel`` with the requested channel.

    Parameters
    ----------
    noise_type : {'depolarizing', 'amplitude_damping', 'thermal_relaxation'}
        Physical noise channel.
    p_1q : float
        Error parameter for single-qubit gates.
    p_2q : float
        Error parameter for two-qubit gates.
    **kwargs
        Extra keyword arguments forwarded to the specific builder
        (e.g. ``gate_time_1q``, ``gate_time_2q`` for thermal relaxation).

    Returns
    -------
    NoiseModel
    """
    builders = {
        "depolarizing": _build_depolarizing,
        "amplitude_damping": _build_amplitude_damping,
        "thermal_relaxation": _build_thermal_relaxation,
    }
    if noise_type not in builders:
        raise ValueError(
            f"Unknown noise_type '{noise_type}'. "
            f"Choose from {list(builders)}."
        )
    return builders[noise_type](p_1q, p_2q, **kwargs)


# ── builders ─────────────────────────────────────────────────────────────

def _build_depolarizing(p_1q: float, p_2q: float, **_kw) -> NoiseModel:
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(
        depolarizing_error(p_1q, 1), SINGLE_QUBIT_GATES,
    )
    nm.add_all_qubit_quantum_error(
        depolarizing_error(p_2q, 2), TWO_QUBIT_GATES,
    )
    return nm


def _build_amplitude_damping(p_1q: float, p_2q: float, **_kw) -> NoiseModel:
    """Amplitude-damping channel (energy relaxation / T₁ decay).

    γ_1q = p_1q  is the probability of |1⟩→|0⟩ per 1-qubit gate.
    For 2-qubit gates we tensor two independent single-qubit AD
    channels each with γ = p_2q.
    """
    from qiskit.quantum_info import Kraus
    from qiskit_aer.noise import QuantumError

    def _ad_kraus_1q(gamma: float):
        K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
        K1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
        return QuantumError(Kraus([K0, K1]))

    e1q = _ad_kraus_1q(p_1q)
    e2q = _ad_kraus_1q(p_2q).tensor(_ad_kraus_1q(p_2q))

    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(e1q, SINGLE_QUBIT_GATES)
    nm.add_all_qubit_quantum_error(e2q, TWO_QUBIT_GATES)
    return nm


def _build_thermal_relaxation(
    p_1q: float,
    p_2q: float,
    *,
    gate_time_1q: float = 50,   # nanoseconds
    gate_time_2q: float = 300,  # nanoseconds
    **_kw,
) -> NoiseModel:
    """Thermal-relaxation channel (T₁ + T₂ process).

    We derive T₁ from the target two-qubit error rate so that the
    effective population decay per 2-qubit gate matches p_2q:

        p_2q ≈ 1 - exp(-t_2q / T₁)   →   T₁ = -t_2q / ln(1 - p_2q)

    T₂ is set to 0.75 T₁ (physically T₂ ≤ T₁; 0.5-0.9 is typical).
    """
    if p_2q >= 1.0:
        p_2q = 0.999  # safety clamp
    t1 = -gate_time_2q / np.log(1 - p_2q)
    t2 = 0.75 * t1  # T2/T1 ratio

    e1q = thermal_relaxation_error(t1, t2, gate_time_1q)
    e2q_single = thermal_relaxation_error(t1, t2, gate_time_2q)
    e2q = e2q_single.tensor(e2q_single)

    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(e1q, SINGLE_QUBIT_GATES)
    nm.add_all_qubit_quantum_error(e2q, TWO_QUBIT_GATES)
    return nm


# ── Fake IBM backend support ────────────────────────────────────────────

# Mapping of short names to FakeBackend classes (lazy-imported).
_FAKE_BACKEND_NAMES = ("fake_kyoto", "fake_osaka")


def get_fake_backend(backend_name: str):
    """Instantiate an IBM fake backend and extract its noise model.

    These backends contain snapshot calibration data from real IBM Quantum
    hardware (ibm_kyoto, ibm_osaka) and produce realistic, qubit-specific
    noise models including T₁/T₂ relaxation, gate errors and readout
    errors.

    Parameters
    ----------
    backend_name : {'fake_kyoto', 'fake_osaka'}
        Short name of the IBM fake backend.

    Returns
    -------
    backend : FakeBackendV2
        The fake backend instance (use as transpilation target).
    noise_model : NoiseModel
        Calibration-derived noise model (pass to AerSimulator).
    """
    from qiskit_ibm_runtime.fake_provider import FakeKyoto, FakeOsaka

    _cls_map = {
        "fake_kyoto": FakeKyoto,
        "fake_osaka": FakeOsaka,
    }
    if backend_name not in _cls_map:
        raise ValueError(
            f"Unknown backend '{backend_name}'. "
            f"Choose from {list(_cls_map)}."
        )
    backend = _cls_map[backend_name]()
    noise_model = NoiseModel.from_backend(backend)
    return backend, noise_model
