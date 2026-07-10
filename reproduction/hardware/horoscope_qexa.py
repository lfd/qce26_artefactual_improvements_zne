#!/usr/bin/env python3
"""
Garbage-Folding Falsification on Real Hardware — IQM Euro-Q-Exa (EQE1)
=====================================================================

Workshop paper:  "When Noise Amplification Fails: Artefact Improvements
                  in Richardson Zero-Noise Extrapolation" (QuBench 2026)

This script runs the garbage-folding falsification experiment of the paper
directly on the 54-qubit IQM Euro-Q-Exa machine (backend ``EQE1`` via the
LRZ MQSS adapter).  The simulator companion lives in
``scripts/horoscope_mechanism.py``; here we collect the *hardware* evidence
that the artefact is not a simulation-only effect.

Protocol
--------
  * Circuit:   TC1 Khan-Trotter (4 qubits, 1 step, calibrated angles),
               observable ``<ZZZZ>``  (Khan et al. 2024).
  * Folding:   gate-level unitary folding at lambda in {1, 3, 5}.
                 - genuine:  G -> G (G^dagger G)^k       (real noise amplification)
                 - garbage:  identical gate *count* and two-qubit (CZ) count,
                             but every single-qubit fold copy is replaced by a
                             random rotation, so the fold no longer implements
                             G^dagger G = I.  The dominant CZ noise is therefore
                             identical to genuine folding, while the coherent
                             signal at lambda > 1 is destroyed.
  * Sampling:  N_REPS independent repetitions x N_SHOTS shots, per method,
               per scale factor.  Full per-state counts are stored so the
               negative-probability diagnostic can be computed offline.

The script is crash-resumable (re-running skips already-collected
(method, rep) pairs) and retries QPU submissions with exponential backoff.

Outputs (under ``--outdir``, default ``build/results/``)
  * ``horoscope_qexa_raw.csv``     — one row per (method, rep, lambda),
                                     including JSON counts.
  * ``horoscope_qexa_summary.csv`` — aggregated genuine/garbage E(lambda),
                                     Richardson E_mit, rho, and per-state
                                     negativity.  Schema is compatible with
                                     ``data/qexa_hardware.csv``.

Usage
-----
  # Local smoke test (Aer depolarising noise, no hardware):
  python hardware/horoscope_qexa.py --local-test

  # Production run on EQE1 (deploy inside tmux):
  MQSS_TOKEN=... python hardware/horoscope_qexa.py --reps 30 --shots 4096

  # Resume after an interruption (same --outdir):
  python hardware/horoscope_qexa.py --reps 30 --shots 4096

References
----------
  [1] Govia et al., "Bounding the Systematic Error in Quantum Error
      Mitigation due to Model Violation", PRX Quantum 6, 010354 (2025).
  [2] Khan et al., "Error Mitigation in the NISQ Era ...",
      Mathematics 12(14), 2235 (2024).
  [3] Giurgica-Tiron et al., "Digital zero noise extrapolation for quantum
      error mitigation", IEEE QCE (2020).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile

# ── path setup ──
SCRIPT_DIR = Path(__file__).resolve().parent      # reproduction/hardware/
REPO_DIR = SCRIPT_DIR.parent                       # reproduction/
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_DIR))

from core.circuits import build_khan_trotter, compute_ideal_expectation
from core.zne import fold_circuit, lagrange_coefficients, sigma_ci

import eqe_common as eqe
import eqe_topology as topo


# ═══════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════

BACKEND_NAME = "EQE1"
N_QUBITS = 4
TROTTER_STEPS = 1
SCALE_FACTORS = [1.0, 3.0, 5.0]
FOLDING_STRATEGY = "from_left"

# Uniform QC-usage convention (see eqe_common): a vetted, connected, low-error
# physical chain mapped 1:1 onto the linear Trotter circuit with
# optimization_level=0 and no_modify=True.  The earlier study used physical
# qubits 1,2,3,4 which are NOT a connected chain, degrading the results.
DEFAULT_LAYOUT = topo.CHAIN4      # [8, 9, 10, 11]

# Pre-calibrated TC1 angles (match the EQE1 drift study; ideal <ZZZZ> ~ 0.981)
RX_ANGLE = 0.097344
RZ_ANGLE = 0.133849

N_REPS_DEFAULT = 30
N_SHOTS_DEFAULT = 4096
MASTER_SEED = 42

# Retry settings (hardware submission)
MAX_RETRIES = 5
RETRY_BASE_DELAY_S = 30

RESULTS_DIR_DEFAULT = REPO_DIR.parent / "build" / "results"
LOGS_DIR = REPO_DIR / "logs"


# ═══════════════════════════════════════════════════════════════════════
#  Logging
# ═══════════════════════════════════════════════════════════════════════

def setup_logging(log_dir: Path, name: str = "horoscope_qexa") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fh = logging.FileHandler(log_dir / f"{name}.log", mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)
    return logger


log = setup_logging(LOGS_DIR)


# ═══════════════════════════════════════════════════════════════════════
#  Garbage folding
# ═══════════════════════════════════════════════════════════════════════

def fold_garbage_hw(
    circuit: QuantumCircuit,
    scale_factor: float,
    rng: np.random.Generator,
) -> QuantumCircuit:
    """Garbage folding (delegates to the shared :func:`eqe_common.fold_garbage`)."""
    return eqe.fold_garbage(circuit, scale_factor, rng)


# ═══════════════════════════════════════════════════════════════════════
#  Backends
# ═══════════════════════════════════════════════════════════════════════

def _read_mqss_token() -> str:
    token = os.environ.get("MQSS_TOKEN")
    if not token:
        env_file = REPO_DIR / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("MQSS_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip("'\"")
                    break
    if not token:
        raise RuntimeError(
            "MQSS_TOKEN not found. Set the MQSS_TOKEN environment variable "
            "or add MQSS_TOKEN=... to reproduction/.env (never commit the token)."
        )
    return token


def get_qexa_backend(backend_name: str = BACKEND_NAME):
    """Connect to the IQM Euro-Q-Exa backend (delegates to eqe_common)."""
    return eqe.get_qexa_backend(backend_name)


# ═══════════════════════════════════════════════════════════════════════
#  Circuit preparation
# ═══════════════════════════════════════════════════════════════════════

def prepare_circuits(backend=None, scale_factors=SCALE_FACTORS,
                     layout=None) -> dict:
    """Build, transpile and fold (genuine + garbage) the TC1 circuit.

    Uses the uniform EQE convention (optimization_level=0 + fixed connected
    ``initial_layout`` + EQE basis).  ``layout`` defaults to the vetted
    :data:`DEFAULT_LAYOUT` chain ``[8, 9, 10, 11]``.
    """
    if layout is None:
        layout = DEFAULT_LAYOUT
    qc_base = build_khan_trotter(
        n_qubits=N_QUBITS, n_steps=TROTTER_STEPS,
        rx_angle=RX_ANGLE, rz_angle=RZ_ANGLE,
    )
    ideal = compute_ideal_expectation(qc_base)
    log.info(f"Ideal <ZZZZ> = {ideal:.6f}")

    # Linear Trotter chain maps 1:1 onto the physical chain -> zero swaps.
    qc_t = eqe.transpile_fixed(qc_base, backend, initial_layout=layout)

    n2q = eqe.two_qubit_count(qc_t)
    log.info(f"Transpiled TC1 on layout {layout}: {qc_t.size()} gates, "
             f"{n2q} two-qubit, depth {qc_t.depth()}")

    genuine, garbage = {}, {}
    for lam in scale_factors:
        rng = np.random.default_rng(MASTER_SEED + int(round(lam * 1000)))
        if lam > 1:
            g = fold_circuit(qc_t, scale_factor=lam, strategy=FOLDING_STRATEGY)
            gb = fold_garbage_hw(qc_t, scale_factor=lam, rng=rng)
        else:
            g = qc_t.copy()
            gb = qc_t.copy()
        eqe.add_measurements(g, layout, N_QUBITS)
        eqe.add_measurements(gb, layout, N_QUBITS)
        genuine[lam] = g
        garbage[lam] = gb
        cz_g = sum(v for k, v in g.count_ops().items() if k in ("cz", "cx", "ecr"))
        cz_b = sum(v for k, v in gb.count_ops().items() if k in ("cz", "cx", "ecr"))
        match = "OK" if (cz_g == cz_b and g.size() == gb.size()) else "MISMATCH"
        log.info(f"  lambda={lam}: genuine={g.size()}g/{cz_g}cz  "
                 f"garbage={gb.size()}g/{cz_b}cz  [{match}]")

    backend_name = backend.name if hasattr(backend, "name") else "local_simulator"
    return {
        "ideal": float(ideal),
        "genuine": genuine,
        "garbage": garbage,
        "scale_factors": list(scale_factors),
        "backend_name": backend_name,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Measurement helpers
# ═══════════════════════════════════════════════════════════════════════

def expectation_from_counts(counts: dict, n_qubits: int = N_QUBITS) -> float:
    """<Z^{otimes n}> = sum_x (-1)^popcount(x) P(x)."""
    total = sum(counts.values())
    val = 0.0
    for bitstring, c in counts.items():
        parity = (-1) ** bitstring.replace(" ", "").count("1")
        val += parity * c
    return val / total


def counts_to_probs(counts: dict, n_qubits: int = N_QUBITS) -> np.ndarray:
    """Dense probability vector (length 2^n) from sparse counts."""
    total = sum(counts.values())
    probs = np.zeros(2 ** n_qubits)
    for bitstring, c in counts.items():
        idx = int(bitstring.replace(" ", ""), 2)
        probs[idx] += c / total
    return probs


# ═══════════════════════════════════════════════════════════════════════
#  Data collection
# ═══════════════════════════════════════════════════════════════════════

def _completed_pairs(raw_path: Path) -> set:
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        return set()
    done = set()
    with open(raw_path) as f:
        for row in csv.DictReader(f):
            done.add((row["method"], int(row["rep"]), float(row["scale_factor"])))
    return done


def _append_rows(raw_path: Path, rows: list[dict]) -> None:
    exists = raw_path.exists() and raw_path.stat().st_size > 0
    with open(raw_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(rows)


def _run_batch_hw(backend, circuits, n_shots):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info(f"  Submitting {len(circuits)} circuits "
                     f"(attempt {attempt}/{MAX_RETRIES}) ...")
            job = backend.run(circuits, shots=n_shots, no_modify=True)
            log.info(f"  Job ID: {job.job_id()}")
            result = job.result()
            return result
        except Exception as e:  # noqa: BLE001 — network/QPU faults are expected
            log.warning(f"  Attempt {attempt} failed: {e}")
            if attempt == MAX_RETRIES:
                raise
            delay = RETRY_BASE_DELAY_S * (2 ** (attempt - 1))
            log.info(f"  Retrying in {delay}s ...")
            time.sleep(delay)


def collect(circuit_set, backend, n_reps, n_shots, raw_path, local=False):
    """Collect genuine + garbage data, one (method, rep) at a time, resumable."""
    sf = circuit_set["scale_factors"]
    ideal = circuit_set["ideal"]
    bname = circuit_set["backend_name"]
    done = _completed_pairs(raw_path)
    if done:
        log.info(f"Resuming: {len(done)} (method, rep, lambda) rows already present.")

    if local:
        from qiskit_aer import AerSimulator
        from core.noise import make_noise_model
        noise_model = make_noise_model("depolarizing", p_1q=1e-3 / 10,
                                       p_2q=1e-3, n_qubits=N_QUBITS)
        sim = AerSimulator(noise_model=noise_model)

    for method in ("genuine", "garbage"):
        circuits = circuit_set[method]
        for rep in range(n_reps):
            todo = [lam for lam in sf if (method, rep, lam) not in done]
            if not todo:
                continue
            batch = [circuits[lam] for lam in todo]
            ts = datetime.now(timezone.utc).isoformat()
            if local:
                seed = MASTER_SEED + hash((method, rep)) % 10_000
                result = sim.run(batch, shots=n_shots, seed_simulator=seed).result()
            else:
                result = _run_batch_hw(backend, batch, n_shots)

            rows = []
            for j, lam in enumerate(todo):
                counts = result.get_counts(j) if len(todo) > 1 else result.get_counts()
                exp_val = expectation_from_counts(counts)
                rows.append({
                    "method": method, "backend": bname, "rep": rep,
                    "scale_factor": lam, "exp_val": f"{exp_val:.6f}",
                    "n_shots": n_shots, "ideal": f"{ideal:.6f}",
                    "timestamp": ts, "counts": json.dumps(counts),
                })
            _append_rows(raw_path, rows)
            evs = {r["scale_factor"]: float(r["exp_val"]) for r in rows}
            log.info(f"  [{method:7s} rep {rep + 1:>3d}/{n_reps}] "
                     + " ".join(f"E({l})={evs.get(l, float('nan')):+.4f}" for l in sf))
    log.info("Collection complete.")


# ═══════════════════════════════════════════════════════════════════════
#  Aggregation & diagnostics
# ═══════════════════════════════════════════════════════════════════════

def aggregate(raw_path: Path, summary_path: Path, scale_factors=SCALE_FACTORS):
    """Aggregate raw rows into E(lambda), Richardson rho, and negativity."""
    import collections
    ev = collections.defaultdict(list)      # (method, lambda) -> [exp_val]
    dist = collections.defaultdict(list)    # (method, lambda) -> [prob vectors]
    ideal_vals = []
    with open(raw_path) as f:
        for row in csv.DictReader(f):
            key = (row["method"], float(row["scale_factor"]))
            ev[key].append(float(row["exp_val"]))
            ideal_vals.append(float(row["ideal"]))
            if row.get("counts"):
                dist[key].append(counts_to_probs(json.loads(row["counts"])))
    ideal = float(np.mean(ideal_vals))
    coeffs = lagrange_coefficients(scale_factors)

    def mean_e(method, lam):
        return float(np.mean(ev[(method, lam)])) if ev[(method, lam)] else float("nan")

    out = {"circuit": "TC1 Trotter 4q", "backend": "EQE1", "n_qubits": N_QUBITS,
           "observable": "ZZZZ", "E_ideal": round(ideal, 6),
           "sigma_ci": round(sigma_ci(scale_factors), 3)}

    for method, tag in (("genuine", "gen"), ("garbage", "garb")):
        E = [mean_e(method, lam) for lam in scale_factors]
        E_mit = float(np.dot(coeffs, E))
        denom = ideal - E[0]
        rho = (E_mit - E[0]) / denom if abs(denom) > 1e-12 else float("nan")
        # per-state Richardson negativity (averaged distributions)
        n_neg, p_min = float("nan"), float("nan")
        if all(dist[(method, lam)] for lam in scale_factors):
            D = np.array([np.mean(dist[(method, lam)], axis=0) for lam in scale_factors])
            P_extrap = coeffs @ D
            n_neg = int(np.sum(P_extrap < 0))
            p_min = float(np.min(P_extrap))
        for i, lam in enumerate(scale_factors):
            out[f"E{int(lam)}_{tag}" if float(lam).is_integer() else f"E_{lam}_{tag}"] = round(E[i], 6)
        out[f"E_mit_{tag}"] = round(E_mit, 6)
        out[f"rho_{tag}"] = round(rho, 4)
        out[f"neg_states_{tag}"] = n_neg
        out[f"neg_pmin_{tag}"] = round(p_min, 6) if p_min == p_min else p_min

    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out.keys()))
        w.writeheader()
        w.writerow(out)

    log.info("=" * 64)
    log.info("  AGGREGATE RESULT (EQE1 garbage-folding falsification)")
    log.info("=" * 64)
    for k, v in out.items():
        log.info(f"    {k:14s} = {v}")
    log.info(f"  Saved: {summary_path}")
    return out


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=N_REPS_DEFAULT)
    ap.add_argument("--shots", type=int, default=N_SHOTS_DEFAULT)
    ap.add_argument("--outdir", default=str(RESULTS_DIR_DEFAULT))
    ap.add_argument("--backend", default=BACKEND_NAME)
    ap.add_argument("--local-test", action="store_true",
                    help="Run on a local Aer depolarising simulator (no hardware).")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="Skip collection; only (re)build the summary from raw CSV.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / "horoscope_qexa_raw.csv"
    summary_path = outdir / "horoscope_qexa_summary.csv"

    if args.local_test and args.reps == N_REPS_DEFAULT:
        args.reps, args.shots = 5, 1024  # lightweight defaults for the smoke test

    log.info("=" * 64)
    log.info("  Horoscope Effect on hardware — garbage-folding falsification")
    log.info(f"  backend={'local-sim' if args.local_test else args.backend}  "
             f"reps={args.reps}  shots={args.shots}")
    log.info("=" * 64)

    if not args.aggregate_only:
        backend = None if args.local_test else get_qexa_backend(args.backend)
        circuit_set = prepare_circuits(backend=backend)
        collect(circuit_set, backend, args.reps, args.shots, raw_path,
                local=args.local_test)

    if raw_path.exists():
        aggregate(raw_path, summary_path)
    else:
        log.warning("No raw data found; nothing to aggregate.")


if __name__ == "__main__":
    main()
