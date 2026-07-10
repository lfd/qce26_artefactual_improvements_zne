"""
Core library for QEM reproduction experiments.

Provides reusable components for:
- Circuit construction (faithful reproduction of published circuits)
- Noise model factory (multiple physically distinct channels)
- ZNE implementation (folding strategies + extrapolation methods)
- Statistical analysis (paired tests, effect sizes, power analysis)
"""

from core.circuits import (
    build_khan_trotter,
    compute_ideal_expectation,
    compute_noisy_expectation,
    compute_qasm_expectation,
    sample_shot_noise,
    calibrate_angles,
)
from core.noise import make_noise_model, get_fake_backend
from core.zne import (
    fold_circuit,
    extrapolate,
    lagrange_coefficients,
    sigma_ci,
)
from core.stats import paired_analysis

__all__ = [
    "build_khan_trotter",
    "compute_ideal_expectation",
    "compute_noisy_expectation",
    "compute_qasm_expectation",
    "sample_shot_noise",
    "calibrate_angles",
    "make_noise_model",
    "get_fake_backend",
    "fold_circuit",
    "extrapolate",
    "lagrange_coefficients",
    "sigma_ci",
    "paired_analysis",
]
