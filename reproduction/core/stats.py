"""
Statistical analysis helpers for QEM reproduction experiments.
==============================================================

All functions operate on paired raw / mitigated expectation-value
arrays.  The key output is a dictionary compatible with CSV export.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats


def paired_analysis(
    raw: np.ndarray,
    mitigated: np.ndarray,
    ideal: float,
    alpha: float = 0.05,
) -> dict:
    """Full paired statistical analysis.

    Parameters
    ----------
    raw : array_like, shape (n,)
        Shot-noise samples of the *raw* (unmitigated) expectation.
    mitigated : array_like, shape (n,)
        Shot-noise samples of the *mitigated* expectation.
    ideal : float
        Noiseless ideal expectation value.
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    dict with keys:
        n, mean_raw, mean_mit, std_raw, std_mit, sigma_ratio,
        mean_delta, mean_improvement, t_stat, p_value_t,
        w_stat, p_value_w, cohen_d, frac_worse, significant
    """
    raw = np.asarray(raw, dtype=float)
    mit = np.asarray(mitigated, dtype=float)
    n = len(raw)

    # Distance to ideal
    raw_err = np.abs(raw - ideal)
    mit_err = np.abs(mit - ideal)

    # Improvement = reduction in error (positive ⇒ mitigation helped)
    improvement = raw_err - mit_err

    # ── paired t-test on error distances ─────────────────────────────
    t_stat, p_value_t = sp_stats.ttest_rel(raw_err, mit_err)

    # ── Wilcoxon signed-rank test ────────────────────────────────────
    try:
        w_stat, p_value_w = sp_stats.wilcoxon(improvement)
    except ValueError:
        # All differences zero — no test possible
        w_stat, p_value_w = np.nan, np.nan

    # ── Cohen's d  (paired, on improvement) ──────────────────────────
    imp_std = improvement.std(ddof=1)
    cohen_d = float(improvement.mean() / imp_std) if imp_std > 0 else 0.0

    # ── fraction of samples where mitigation made things worse ───────
    frac_worse = float(np.mean(mit_err > raw_err))

    # ── variance ratio (σ_mit / σ_raw) ──────────────────────────────
    std_raw = float(raw.std(ddof=1))
    std_mit = float(mit.std(ddof=1))
    sigma_ratio = std_mit / std_raw if std_raw > 0 else float("inf")

    return {
        "n": n,
        "mean_raw": float(raw.mean()),
        "mean_mit": float(mit.mean()),
        "std_raw": std_raw,
        "std_mit": std_mit,
        "sigma_ratio": sigma_ratio,
        "mean_delta": float((mit - raw).mean()),
        "mean_improvement": float(improvement.mean()),
        "t_stat": float(t_stat),
        "p_value_t": float(p_value_t),
        "w_stat": float(w_stat) if not np.isnan(w_stat) else None,
        "p_value_w": float(p_value_w) if not np.isnan(p_value_w) else None,
        "cohen_d": cohen_d,
        "frac_worse": frac_worse,
        "significant": bool(p_value_t < alpha),
    }
