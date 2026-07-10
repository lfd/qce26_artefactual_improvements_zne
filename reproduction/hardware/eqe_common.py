"""
Shared Euro-Q-Exa (EQE1) QC-usage conventions — single source of truth.
=======================================================================

Every hardware script in this package goes through this module so that the
**same** conventions are applied everywhere.  The conventions are:

  * ``optimization_level=0`` — no transpiler optimisation, so the circuit we
    submit is the circuit we designed (faithful benchmarking).
  * an explicit ``initial_layout`` onto a vetted, *connected* nearest-neighbour
    physical chain (see :mod:`eqe_topology`).  We never let the transpiler
    pick the layout.
  * ``backend.run(..., no_modify=True)`` — the LRZ MQSS adapter then keeps the
    mapped qubits exactly as submitted (no server-side re-routing onto other,
    possibly worse, qubits).

Rationale.  The earlier study mapped onto physical qubits ``1,2,3,4`` which
are *not* a connected chain (``1`` is not adjacent to ``2``; edge ``(3,4)`` is
the worst on the device), so the adapter had to re-route and the results were
degraded.  Pinning a good chain with ``no_modify=True`` removes that variance.

This module never hardcodes the MQSS token; it is read from the environment
or ``reproduction/.env`` (gitignored).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile

import eqe_topology as topo

# ── path setup (this file lives in reproduction/hardware/) ───────────────
HARDWARE_DIR = Path(__file__).resolve().parent
REPO_DIR = HARDWARE_DIR.parent                # reproduction/

# ── conventions ──────────────────────────────────────────────────────────
BACKEND_NAME = "EQE1"
# EQE1 is an IQM device: native gate set is the phased-rotation gate ``r``
# (a.k.a. PRX) and ``cz``, plus ``id`` and ``measure``.  There is NO native
# ``rz``/``sx``/``x``.  We only use this list for *local* simulation; for a real
# backend we let the backend's own target define the native gates (passing an
# explicit basis alongside a backend is both discouraged and, here, wrong).
EQE_BASIS = ["cz", "r", "id"]
OPTIMIZATION_LEVEL = 0
NO_MODIFY = True

N_QUBITS_DEFAULT = 4
N_REPS_DEFAULT = 30
N_SHOTS_DEFAULT = 4096
MASTER_SEED = 42

# Retry policy for hardware submissions.
MAX_RETRIES = 5
RETRY_BASE_DELAY_S = 30

DEFAULT_LAYOUTS = topo.DEFAULT_LAYOUTS

log = logging.getLogger("eqe")


# ═══════════════════════════════════════════════════════════════════════
#  Credentials & backend
# ═══════════════════════════════════════════════════════════════════════

def read_mqss_token() -> str:
    """Read the MQSS token from ``MQSS_TOKEN`` env-var or ``reproduction/.env``."""
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
    """Connect to the IQM Euro-Q-Exa backend through the LRZ MQSS adapter."""
    token = read_mqss_token()
    from mqss.qiskit_adapter import MQSSQiskitAdapter

    adapter = MQSSQiskitAdapter(token=token)
    backend = adapter.get_backend(backend_name)
    log.info(f"Connected to {backend_name}")
    return backend


# ═══════════════════════════════════════════════════════════════════════
#  Layout selection
# ═══════════════════════════════════════════════════════════════════════

def _live_coupling(backend) -> list[tuple[int, int]] | None:
    """Best-effort extraction of the live coupling map; ``None`` if absent."""
    try:
        cm = backend.coupling_map
        if cm is None:
            return None
        return [tuple(e) for e in cm.get_edges()]
    except Exception:  # noqa: BLE001 — adapters vary; fall back to static map
        return None


def _live_cz_error(backend, a: int, b: int) -> float | None:
    """Best-effort live two-qubit error for edge (a, b)."""
    try:
        target = backend.target
        for name in ("cz", "cx", "ecr"):
            inst = target.get(name)
            if inst and (a, b) in inst:
                props = inst[(a, b)]
                if props is not None and props.error is not None:
                    return float(props.error)
            if inst and (b, a) in inst:
                props = inst[(b, a)]
                if props is not None and props.error is not None:
                    return float(props.error)
    except Exception:  # noqa: BLE001
        return None
    return None


def pick_linear_chain(backend, n: int,
                      prefer: list[int] | None = None) -> dict:
    """Select a length-``n`` connected nearest-neighbour chain.

    Prefers the **live** coupling map and calibration; falls back to the
    static :mod:`eqe_topology` snapshot.  Returns a dict with the chosen
    ``layout`` and the provenance/score, suitable for logging to a manifest.

    If ``prefer`` is supplied and is a valid connected path on the live (or
    static) coupling map, it is used as-is.
    """
    live_edges = _live_coupling(backend) if backend is not None else None
    static_edges = topo.build_coupling_map()
    edges = live_edges if live_edges else static_edges
    source = "live" if live_edges else "static"
    edge_set = {tuple(sorted(e)) for e in edges}

    def connected(path: list[int]) -> bool:
        if len(set(path)) != len(path):
            return False
        return all(tuple(sorted((path[i], path[i + 1]))) in edge_set
                   for i in range(len(path) - 1))

    # 1. honour an explicit preference if it is valid
    if prefer is not None and connected(prefer):
        return {"layout": list(prefer), "source": f"{source}:prefer", "n": n}

    # 2. try the vetted default chain
    default = DEFAULT_LAYOUTS.get(f"chain{n}")
    if default is not None and connected(default):
        return {"layout": list(default), "source": f"{source}:default", "n": n}

    # 3. rank simple paths on the chosen coupling map
    adj: dict[int, list[int]] = {}
    for a, b in edge_set:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    def cost(a: int, b: int) -> float:
        if backend is not None and source == "live":
            e = _live_cz_error(backend, a, b)
            if e is not None:
                return e
        f = topo.edge_fidelity(a, b)
        return 1.0 - f if f is not None else 0.02

    best: tuple[float, list[int]] | None = None

    def dfs(path: list[int], c: float) -> None:
        nonlocal best
        if len(path) == n:
            if best is None or c < best[0]:
                best = (c, list(path))
            return
        for nb in adj.get(path[-1], []):
            if nb in path or nb in topo.DEAD_QUBITS:
                continue
            dfs(path + [nb], c + cost(path[-1], nb))

    for start in adj:
        if start in topo.DEAD_QUBITS:
            continue
        dfs([start], 0.0)

    if best is not None:
        return {"layout": best[1], "source": f"{source}:ranked",
                "n": n, "cost": round(best[0], 6)}

    # 4. last resort: static best chain (ignores connectivity gaps)
    return {"layout": topo.best_static_chain(n),
            "source": "static:fallback", "n": n}


# ═══════════════════════════════════════════════════════════════════════
#  Transpilation (uniform convention)
# ═══════════════════════════════════════════════════════════════════════

def transpile_fixed(circuit: QuantumCircuit, backend, initial_layout: list[int],
                    assert_no_swap: bool = True) -> QuantumCircuit:
    """Transpile ``circuit`` under the uniform EQE convention.

    ``optimization_level=0`` + explicit ``initial_layout`` + EQE basis.  When
    a backend is given its coupling map is respected; for local tests
    (``backend=None``) only the basis is enforced.

    Parameters
    ----------
    assert_no_swap : bool
        If True (default), raise if the transpiler had to insert ``swap``
        gates — i.e. the logical circuit did not match the physical chain.
        Set False for intrinsically non-linear circuits (e.g. Grover) where
        routing is expected.
    """
    n = circuit.num_qubits
    if len(initial_layout) < n:
        raise ValueError(
            f"initial_layout has {len(initial_layout)} qubits but the circuit "
            f"needs {n}.")
    layout = initial_layout[:n]

    if backend is not None:
        # Let the backend's target govern the native gates and coupling map;
        # passing an explicit basis_gates here would invalidate the backend's
        # calibrated gate set (EQE1 natively supports only {cz, id, r}).
        qc_t = transpile(
            circuit, backend=backend,
            initial_layout=layout,
            optimization_level=OPTIMIZATION_LEVEL,
        )
    else:
        qc_t = transpile(
            circuit,
            optimization_level=OPTIMIZATION_LEVEL,
            basis_gates=EQE_BASIS,
        )

    n_swap = qc_t.count_ops().get("swap", 0)
    if assert_no_swap and n_swap > 0:
        raise RuntimeError(
            f"transpile_fixed inserted {n_swap} swap(s): the circuit's "
            f"two-qubit connectivity does not match the linear chain "
            f"{layout}. Choose a matching layout or pass assert_no_swap=False.")
    if n_swap:
        log.warning(f"  transpile_fixed: {n_swap} swap(s) inserted (routing).")
    return qc_t


def two_qubit_count(circuit: QuantumCircuit) -> int:
    """Number of two-qubit gates (cz/cx/ecr) in a circuit."""
    return sum(v for k, v in circuit.count_ops().items()
               if k in ("cz", "cx", "ecr"))


def add_measurements(circuit: QuantumCircuit, layout: list[int],
                     n_logical: int) -> QuantumCircuit:
    """Measure only the *active* qubits (in place) and return the circuit.

    A backend-transpiled circuit is the full device width (e.g. 54 qubits on
    EQE1); using ``measure_all`` would try to measure idle/dead qubits and the
    IQM backend rejects that.  We therefore add an ``n_logical``-bit classical
    register and measure exactly the physical qubits that carry the logical
    circuit:

      * hardware case (``circuit.num_qubits > n_logical``): the physical qubits
        are ``layout[:n_logical]``;
      * local-sim case (``circuit.num_qubits == n_logical``): qubits
        ``0 … n_logical-1``.

    The measured bit order is the layout order; the parity observable
    ``⟨Z^{⊗n}⟩`` is order-independent, so this is unambiguous.
    """
    from qiskit import ClassicalRegister

    if circuit.num_qubits == n_logical:
        phys = list(range(n_logical))
    else:
        phys = list(layout[:n_logical])
    creg = ClassicalRegister(n_logical, "meas")
    circuit.add_register(creg)
    for i, q in enumerate(phys):
        circuit.measure(q, creg[i])
    return circuit


# ═══════════════════════════════════════════════════════════════════════
#  Garbage folding (shared)
# ═══════════════════════════════════════════════════════════════════════

def fold_garbage(circuit: QuantumCircuit, scale_factor: float,
                 rng: np.random.Generator) -> QuantumCircuit:
    """Garbage folding for an *already transpiled* circuit.

    Mirrors the gate-level from-left structure of ``core.zne.fold_circuit`` so
    the total gate count and two-qubit (CZ) count match genuine folding
    exactly.  For every fold copy two-qubit gates are re-applied unchanged on
    the same qubits (preserving the dominant noise channel), while single-qubit
    gates are replaced by a random rotation, so the inserted block no longer
    implements ``G†G = I`` and the coherent signal at ``λ>1`` is destroyed.
    """
    if scale_factor <= 1.0:
        return circuit.copy()

    ops = [(inst.operation, inst.qubits, inst.clbits) for inst in circuit.data]
    n_gates = len(ops)
    n_full = int((scale_factor - 1) // 2)
    total_desired = int(round(scale_factor * n_gates))
    gates_after_full = n_gates * (1 + 2 * n_full)
    n_extra = max(0, (total_desired - gates_after_full) // 2)
    n_extra = min(n_extra, n_gates)

    folded = circuit.copy_empty_like()
    for i, (op, qubits, clbits) in enumerate(ops):
        folded.append(op, qubits, clbits)
        if op.name in ("measure", "barrier"):
            continue
        k = n_full + (1 if i < n_extra else 0)
        for _ in range(k):
            for _ in range(2):  # two garbage gates replacing (G†, G)
                if op.num_qubits >= 2:
                    folded.append(op, qubits, clbits)
                else:
                    # native single-qubit IQM gate r(theta, phi) with random
                    # angles — destroys the coherent signal while staying native.
                    folded.r(float(rng.uniform(0.0, 2.0 * np.pi)),
                             float(rng.uniform(0.0, 2.0 * np.pi)), qubits[0])
    return folded


def fold_identity(circuit: QuantumCircuit, scale_factor: float,
                  rng: np.random.Generator) -> QuantumCircuit:
    """Null model: insert *logical-identity* fold copies.

    The exact controlled counterpart of :func:`fold_garbage`: every fold copy
    has the **same gate budget** — two-qubit gates are re-applied unchanged
    (``CZ·CZ = I``) and each single-qubit fold copy contributes *two* native
    ``r`` pulses — but here the single-qubit pair is chosen to *compose to the
    identity* (``r(θ,φ)·r(-θ,φ) = I``) instead of two random rotations.  The
    inserted block therefore adds the **same real gate noise** as garbage
    folding while preserving the coherent signal, isolating *signal
    destruction* (rather than mere added depth/noise) as the cause of the
    artefact ``rho_garb``.

    .. note::
       We deliberately avoid the ``id`` instruction.  Although EQE1's target
       lists ``id`` as native, the MQSS/IQM submission path re-expands a bare
       :class:`~qiskit...IGate` into a ``u`` gate (which is *not* natively
       supported), cancelling the whole job.  A real ``r(θ,φ)·r(-θ,φ)`` pair is
       genuinely native and survives ``no_modify`` submission unchanged.
    """
    if scale_factor <= 1.0:
        return circuit.copy()
    ops = [(inst.operation, inst.qubits, inst.clbits) for inst in circuit.data]
    n_gates = len(ops)
    n_full = int((scale_factor - 1) // 2)
    total_desired = int(round(scale_factor * n_gates))
    gates_after_full = n_gates * (1 + 2 * n_full)
    n_extra = max(0, (total_desired - gates_after_full) // 2)
    n_extra = min(n_extra, n_gates)

    theta, phi = np.pi / 2.0, 0.0   # fixed identity-composing pair r(θ,φ)·r(-θ,φ)
    folded = circuit.copy_empty_like()
    for i, (op, qubits, clbits) in enumerate(ops):
        folded.append(op, qubits, clbits)
        if op.name in ("measure", "barrier"):
            continue
        k = n_full + (1 if i < n_extra else 0)
        for _ in range(k):
            if op.num_qubits >= 2:
                folded.append(op, qubits, clbits)   # CZ·CZ = I
                folded.append(op, qubits, clbits)
            else:
                folded.r(theta, phi, qubits[0])      # r(θ,φ) ...
                folded.r(-theta, phi, qubits[0])     # ... r(-θ,φ) = I
    return folded


# ═══════════════════════════════════════════════════════════════════════
#  Submission with retry
# ═══════════════════════════════════════════════════════════════════════

def submit_with_retry(backend, circuits, n_shots: int,
                      max_retries: int = MAX_RETRIES,
                      base_delay: int = RETRY_BASE_DELAY_S):
    """Submit a batch to EQE1 with ``no_modify=True`` and exponential backoff."""
    for attempt in range(1, max_retries + 1):
        try:
            log.info(f"  Submitting {len(circuits)} circuits "
                     f"(attempt {attempt}/{max_retries}, no_modify={NO_MODIFY}) ...")
            job = backend.run(circuits, shots=n_shots, no_modify=NO_MODIFY)
            log.info(f"  Job ID: {job.job_id()}")
            result = job.result()
            return result
        except Exception as e:  # noqa: BLE001 — network/QPU faults are expected
            log.warning(f"  Attempt {attempt} failed: {e}")
            if attempt == max_retries:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            log.info(f"  Retrying in {delay}s ...")
            time.sleep(delay)


# ═══════════════════════════════════════════════════════════════════════
#  Measurement helpers
# ═══════════════════════════════════════════════════════════════════════

def expectation_from_counts(counts: dict, n_qubits: int) -> float:
    """⟨Z^{⊗n}⟩ = Σ_x (−1)^popcount(x) P(x)."""
    total = sum(counts.values())
    val = 0.0
    for bitstring, c in counts.items():
        parity = (-1) ** bitstring.replace(" ", "").count("1")
        val += parity * c
    return val / total


def counts_to_probs(counts: dict, n_qubits: int) -> np.ndarray:
    """Dense probability vector (length 2^n) from sparse counts."""
    total = sum(counts.values())
    probs = np.zeros(2 ** n_qubits)
    for bitstring, c in counts.items():
        idx = int(bitstring.replace(" ", ""), 2)
        probs[idx] += c / total
    return probs


# ═══════════════════════════════════════════════════════════════════════
#  CSV resume / append helpers
# ═══════════════════════════════════════════════════════════════════════

def completed_keys(raw_path: Path, key_fields: tuple[str, ...]) -> set:
    """Return the set of already-collected key tuples in a raw CSV."""
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        return set()
    done = set()
    with open(raw_path) as f:
        for row in csv.DictReader(f):
            done.add(tuple(row[k] for k in key_fields))
    return done


def append_rows(raw_path: Path, rows: list[dict]) -> None:
    """Append rows to a CSV, writing the header if the file is new."""
    if not rows:
        return
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    exists = raw_path.exists() and raw_path.stat().st_size > 0
    with open(raw_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
