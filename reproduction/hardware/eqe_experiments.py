#!/usr/bin/env python3
"""
Euro-Q-Exa hardware experiment registry (QuBench workshop paper).
=================================================================

A registry of self-contained, crash-resumable experiment *jobs* that collect
the IQM Euro-Q-Exa (backend ``EQE1``) hardware evidence for the paper
"Benchmarking Error Mitigation: Artefactual Improvements in Zero-Noise
Extrapolation".  The runner ``run_hardware_experiments.py`` executes these jobs
back-to-back, advancing only when the current job has finished
(completion-gated, not clock-gated).

Every job:

  * goes through :mod:`eqe_common` so the **uniform QC convention** is applied
    everywhere — ``optimization_level=0`` + a fixed, connected, low-error
    physical chain (:mod:`eqe_topology`) + ``no_modify=True``;
  * writes its raw per-shot data to ``<outdir>/<job>_raw.csv`` incrementally
    (one batch at a time) and is **resumable** — re-running skips already
    collected ``(method, rep, scale_factor)`` rows;
  * writes an aggregated ``<outdir>/<job>_summary.csv`` and returns a small
    result dict.

Jobs (in scheduling order)
--------------------------
  j0_health        connectivity + calibration probe; selects chain4 / chain5.
  j1_regime_scan   find a *retained-signal* and a *destroyed-signal* TC config.
  j4_depth         TC1 / TC3 / TC5 genuine vs garbage (retained -> destroyed);
                   feeds the rescaling-collapse figure.
  j6_nullmodels    genuine / garbage / identity null-model control.
  j8_drift_design  blocked vs interleaved A/B acquisition (drift bias).
  j9_allan_probe   single-circuit drift probe -> Allan deviation.
  j12_drift_design_long  j8 over a long window (delay-spaced blocked vs interleaved).
  j13_allan_probe_long   j9 over a long window (reaches larger Allan windows m).
  j11_czne_dataset       genuine/garbage/identity regime sweep (retained -> mid
                   -> destroyed); feeds the negative-probability diagnostic.
  j14_czne_dataset_big   larger regime sweep for the negative-probability weight.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit

# ── path setup (this file lives in reproduction/hardware/) ───────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_DIR))

import eqe_common as eqe
import eqe_topology as topo
from core.circuits import (
    build_khan_trotter, compute_ideal_expectation,
)
from core.zne import fold_circuit, lagrange_coefficients, sigma_ci


# ═══════════════════════════════════════════════════════════════════════
#  TC angle configs (calibrated for EQE1; ideal <ZZZZ> noted)
# ═══════════════════════════════════════════════════════════════════════
# "destroyed":  the calibrated TC1 used in the drift study — on hardware the
#               signal is already destroyed at lambda=5 (E(5)<0).
# "retained":   a shallower / smaller-angle config expected to retain signal
#               at lambda=5; the actual choice is confirmed by j1_regime_scan.
TC_DESTROYED = {"rx": 0.097344, "rz": 0.133849, "steps": 1}
TC_RETAINED_CANDIDATES = [
    {"rx": 0.05, "rz": 0.05, "steps": 1},
    {"rx": 0.097344, "rz": 0.05, "steps": 1},
    {"rx": 0.05, "rz": 0.133849, "steps": 1},
]

# Richardson coefficient spectra (scale-factor sets) studied in the paper.
SPECTRA = {
    "kim": [1.0, 3.0, 5.0],
    "hour": [1.0, 1.5, 2.0, 2.5],
    "kandala": [1.0, 1.1, 1.25, 1.5],
}

N_QUBITS_TC = 4


# ── long-duration follow-ups (j12–j14) ───────────────────────────
# These deliberately span a *long wall-clock window* so slow device drift has
# time to manifest (the short j8/j9 runs sampled ~18 min and saw little drift).
# A small inter-sample delay (hardware only) stretches the acquisition without
# burning extra shots; the jobs stay fully resumable.
LONG_DRIFT_REPS = 60          # blocked+interleaved A/B reps  (j12)
LONG_PROBE_SAMPLES = 220      # Allan-probe time-series length (j13)
LONG_SAMPLE_DELAY_S = 6       # inter-sample spacing on hardware (s)
BIG_DATASET_REPS = 30         # per regime/method reps for the big regime sweep (j14)


# ═══════════════════════════════════════════════════════════════════════
#  Run context
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Context:
    """Shared state passed to every job."""
    backend: object | None         # None => local Aer simulator
    outdir: Path
    log: object
    reps: int = eqe.N_REPS_DEFAULT
    shots: int = eqe.N_SHOTS_DEFAULT
    local: bool = False
    seed: int = eqe.MASTER_SEED
    layouts: dict = field(default_factory=dict)   # filled by j0_health
    regime: dict = field(default_factory=dict)    # filled by j1_regime_scan
    _sim: object = None

    def sim(self):
        """Lazily build a local Aer depolarising simulator (for --local-test)."""
        if self._sim is None:
            from qiskit_aer import AerSimulator
            from core.noise import make_noise_model
            nm = make_noise_model("depolarizing", p_1q=1e-3 / 10, p_2q=1e-3,
                                  n_qubits=8)
            self._sim = AerSimulator(noise_model=nm)
        return self._sim

    def chain(self, n: int) -> list[int]:
        key = f"chain{n}"
        if key in self.layouts:
            return self.layouts[key]["layout"]
        return topo.DEFAULT_LAYOUTS.get(key, topo.CHAIN4)


# ═══════════════════════════════════════════════════════════════════════
#  Folding dispatch
# ═══════════════════════════════════════════════════════════════════════

def _fold(method: str, qc_t: QuantumCircuit, lam: float,
          rng: np.random.Generator) -> QuantumCircuit:
    if lam <= 1.0:
        return qc_t.copy()
    if method == "genuine":
        return fold_circuit(qc_t, scale_factor=lam, strategy="from_left")
    if method == "garbage":
        return eqe.fold_garbage(qc_t, lam, rng)
    if method == "identity":
        return eqe.fold_identity(qc_t, lam, rng)
    raise ValueError(f"unknown method {method!r}")


# ═══════════════════════════════════════════════════════════════════════
#  Generic genuine/garbage/identity runner (resumable)
# ═══════════════════════════════════════════════════════════════════════

def run_gg(ctx: Context, name: str, base: QuantumCircuit,
           scale_factors: list[float], layout: list[int],
           methods=("genuine", "garbage"),
           assert_no_swap: bool = True) -> dict:
    """Collect genuine/garbage(/identity) data for one circuit, resumably.

    Returns the aggregated summary dict and writes ``<name>_raw.csv`` and
    ``<name>_summary.csv`` under ``ctx.outdir``.
    """
    n_qubits = base.num_qubits
    ideal = compute_ideal_expectation(base)
    qc_t = eqe.transpile_fixed(base, ctx.backend, initial_layout=layout,
                               assert_no_swap=assert_no_swap)
    n2q = eqe.two_qubit_count(qc_t)
    ctx.log.info(f"[{name}] layout={layout[:n_qubits]} ideal={ideal:+.4f} "
                 f"{qc_t.size()}g/{n2q}cz depth={qc_t.depth()}")

    # Build folded circuit set, asserting genuine/garbage gate parity.
    circuits: dict[tuple[str, float], QuantumCircuit] = {}
    for method in methods:
        for lam in scale_factors:
            rng = np.random.default_rng(ctx.seed + int(round(lam * 1000)))
            qc_f = _fold(method, qc_t, lam, rng)
            eqe.add_measurements(qc_f, layout, n_qubits)
            circuits[(method, lam)] = qc_f
    for lam in scale_factors:
        sizes = {m: circuits[(m, lam)].size() for m in methods}
        czs = {m: eqe.two_qubit_count(circuits[(m, lam)]) for m in methods}
        ok = len(set(czs.values())) == 1
        ctx.log.info(f"  lambda={lam}: " +
                     " ".join(f"{m}={sizes[m]}g/{czs[m]}cz" for m in methods) +
                     ("  [CZ OK]" if ok else "  [CZ MISMATCH]"))

    raw_path = ctx.outdir / f"{name}_raw.csv"
    summary_path = ctx.outdir / f"{name}_summary.csv"
    key_fields = ("method", "rep", "scale_factor")
    done = eqe.completed_keys(raw_path, key_fields)
    if done:
        ctx.log.info(f"  resuming: {len(done)} rows already present")

    for method in methods:
        for rep in range(ctx.reps):
            todo = [lam for lam in scale_factors
                    if not _is_done(done, method, rep, lam)]
            if not todo:
                continue
            batch = [circuits[(method, lam)] for lam in todo]
            ts = datetime.now(timezone.utc).isoformat()
            if ctx.local:
                seed = ctx.seed + (hash((name, method, rep)) % 10_000)
                result = ctx.sim().run(batch, shots=ctx.shots,
                                       seed_simulator=seed).result()
            else:
                result = eqe.submit_with_retry(ctx.backend, batch, ctx.shots)
            rows = []
            for j, lam in enumerate(todo):
                counts = (result.get_counts(j) if len(todo) > 1
                          else result.get_counts())
                ev = eqe.expectation_from_counts(counts, n_qubits)
                rows.append({
                    "job": name, "method": method, "backend": _bname(ctx),
                    "rep": rep, "scale_factor": lam, "n_qubits": n_qubits,
                    "exp_val": f"{ev:.6f}", "n_shots": ctx.shots,
                    "ideal": f"{ideal:.6f}", "layout": json.dumps(layout[:n_qubits]),
                    "timestamp": ts, "counts": json.dumps(counts),
                })
            eqe.append_rows(raw_path, rows)
            evs = {r["scale_factor"]: float(r["exp_val"]) for r in rows}
            ctx.log.info(f"  [{method:8s} rep {rep + 1:>3d}/{ctx.reps}] " +
                         " ".join(f"E({l})={evs.get(l, float('nan')):+.4f}"
                                  for l in scale_factors))

    return aggregate(ctx, raw_path, summary_path, scale_factors, methods,
                     n_qubits)


def _is_done(done: set, method: str, rep: int, lam: float) -> bool:
    return ((method, str(rep), str(lam)) in done or
            (method, str(rep), str(float(lam))) in done)


def _bname(ctx: Context) -> str:
    if ctx.local or ctx.backend is None:
        return "local_simulator"
    return getattr(ctx.backend, "name", "EQE1")


def _run_batch(ctx: Context, tag: str, sub: str, rep: int,
               batch: list, shots: int) -> list[dict]:
    """Submit one batch (local sim or EQE1) and return a list of counts dicts.

    Mirrors the submission logic of :func:`run_gg` so the drift and regime jobs
    below share identical QC conventions (``no_modify`` on hardware, seeded
    depolarising Aer locally).
    """
    if ctx.local:
        seed = ctx.seed + (hash((tag, sub, rep)) % 10_000)
        result = ctx.sim().run(batch, shots=shots, seed_simulator=seed).result()
    else:
        result = eqe.submit_with_retry(ctx.backend, batch, shots)
    return [result.get_counts(j) if len(batch) > 1 else result.get_counts()
            for j in range(len(batch))]


# ═══════════════════════════════════════════════════════════════════════
#  Aggregation & diagnostics
# ═══════════════════════════════════════════════════════════════════════

def aggregate(ctx: Context, raw_path: Path, summary_path: Path,
              scale_factors: list[float], methods, n_qubits: int) -> dict:
    """Aggregate raw rows into E(lambda), Richardson rho, and negativity."""
    import collections
    import csv
    ev = collections.defaultdict(list)
    dist = collections.defaultdict(list)
    ideal_vals = []
    with open(raw_path) as f:
        for row in csv.DictReader(f):
            key = (row["method"], float(row["scale_factor"]))
            ev[key].append(float(row["exp_val"]))
            ideal_vals.append(float(row["ideal"]))
            if row.get("counts"):
                dist[key].append(eqe.counts_to_probs(json.loads(row["counts"]),
                                                     n_qubits))
    ideal = float(np.mean(ideal_vals)) if ideal_vals else float("nan")
    coeffs = lagrange_coefficients(scale_factors)

    out = {"backend": _bname(ctx), "n_qubits": n_qubits,
           "E_ideal": round(ideal, 6), "sigma_ci": round(sigma_ci(scale_factors), 3),
           "scale_factors": json.dumps(scale_factors)}

    for method in methods:
        E = [float(np.mean(ev[(method, lam)])) if ev[(method, lam)] else float("nan")
             for lam in scale_factors]
        E_mit = float(np.dot(coeffs, E))
        denom = ideal - E[0]
        rho = (E_mit - E[0]) / denom if abs(denom) > 1e-12 else float("nan")
        n_neg, p_min = float("nan"), float("nan")
        if all(dist[(method, lam)] for lam in scale_factors):
            D = np.array([np.mean(dist[(method, lam)], axis=0)
                          for lam in scale_factors])
            P_extrap = coeffs @ D
            n_neg = int(np.sum(P_extrap < 0))
            p_min = float(np.min(P_extrap))
        tag = method[:4]
        out[f"E_lambda1_{tag}"] = round(E[0], 6)
        out[f"E_mit_{tag}"] = round(E_mit, 6)
        out[f"rho_{tag}"] = round(rho, 4)
        out[f"neg_states_{tag}"] = n_neg
        out[f"neg_pmin_{tag}"] = round(p_min, 6) if p_min == p_min else p_min

    # genuine-minus-garbage effect, if both present
    rg = out.get("rho_genu", float("nan"))
    rb = out.get("rho_garb", float("nan"))
    if rg == rg and rb == rb:
        out["delta_rho"] = round(rg - rb, 4)

    eqe.write_json(summary_path.with_suffix(".json"), out)
    import csv as _csv
    with open(summary_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(out.keys()))
        w.writeheader()
        w.writerow(out)
    ctx.log.info(f"  aggregate -> {summary_path.name}: " +
                 " ".join(f"{k}={out[k]}" for k in out if k.startswith("rho_")))
    return out


# ═══════════════════════════════════════════════════════════════════════
#  Circuit builders
# ═══════════════════════════════════════════════════════════════════════

def _tc(cfg: dict, n_qubits: int = N_QUBITS_TC) -> QuantumCircuit:
    return build_khan_trotter(n_qubits=n_qubits, n_steps=cfg["steps"],
                              rx_angle=cfg["rx"], rz_angle=cfg["rz"])


# ═══════════════════════════════════════════════════════════════════════
#  Jobs
# ═══════════════════════════════════════════════════════════════════════

def j0_health(ctx: Context) -> dict:
    """Connectivity + calibration probe; pick and persist chain4 / chain5."""
    info = {"timestamp": datetime.now(timezone.utc).isoformat()}
    chain4 = eqe.pick_linear_chain(ctx.backend, 4)
    chain5 = eqe.pick_linear_chain(ctx.backend, 5)
    ctx.layouts = {"chain4": chain4, "chain5": chain5}
    info["chain4"] = chain4
    info["chain5"] = chain5

    # connectivity sanity (static map is always available)
    info["chain4_connected"] = topo.is_connected_path(chain4["layout"])
    info["chain5_connected"] = topo.is_connected_path(chain5["layout"])
    if not info["chain4_connected"]:
        ctx.log.warning(f"  chain4 {chain4['layout']} NOT connected on static map!")

    # optional: a 1-rep TC1 ping at lambda=1 to confirm the device is alive
    try:
        base = _tc(TC_DESTROYED)
        qc_t = eqe.transpile_fixed(base, ctx.backend,
                                   initial_layout=chain4["layout"])
        eqe.add_measurements(qc_t, chain4["layout"], base.num_qubits)
        if ctx.local:
            res = ctx.sim().run([qc_t], shots=ctx.shots,
                                seed_simulator=ctx.seed).result()
        else:
            res = eqe.submit_with_retry(ctx.backend, [qc_t],
                                        min(ctx.shots, 1024))
        ev = eqe.expectation_from_counts(res.get_counts(), base.num_qubits)
        info["ping_E_lambda1"] = round(ev, 4)
        ctx.log.info(f"  ping E(lambda=1) = {ev:+.4f}")
    except Exception as e:  # noqa: BLE001
        info["ping_error"] = str(e)
        ctx.log.warning(f"  ping failed: {e}")

    eqe.write_json(ctx.outdir / "selected_layout.json", info)
    return info


def j1_regime_scan(ctx: Context) -> dict:
    """Find a retained-signal and a destroyed-signal TC config at {1,3,5}.

    Runs genuine folding only (cheap) for a handful of candidate angle sets
    and classifies each by E(lambda=5): retained if E(5) stays clearly
    positive and monotone, destroyed if E(5) <= 0.  Writes ``regime_choice``.
    """
    layout = ctx.chain(4)
    sf = [1.0, 3.0, 5.0]
    candidates = [("destroyed", TC_DESTROYED)] + [
        (f"cand{i}", c) for i, c in enumerate(TC_RETAINED_CANDIDATES)
    ]
    scan = {}
    reps = max(3, min(ctx.reps, 8))   # keep the scan cheap
    for tag, cfg in candidates:
        base = _tc(cfg)
        ideal = compute_ideal_expectation(base)
        qc_t = eqe.transpile_fixed(base, ctx.backend, initial_layout=layout)
        E = {}
        for lam in sf:
            rng = np.random.default_rng(ctx.seed + int(lam))
            qc_f = _fold("genuine", qc_t, lam, rng)
            eqe.add_measurements(qc_f, layout, base.num_qubits)
            vals = []
            for rep in range(reps):
                if ctx.local:
                    res = ctx.sim().run([qc_f], shots=ctx.shots,
                                        seed_simulator=ctx.seed + rep).result()
                else:
                    res = eqe.submit_with_retry(ctx.backend, [qc_f], ctx.shots)
                vals.append(eqe.expectation_from_counts(res.get_counts(),
                                                        base.num_qubits))
            E[lam] = float(np.mean(vals))
        retained = E[5.0] > 0.10 and E[1.0] > E[3.0] > E[5.0]
        scan[tag] = {"cfg": cfg, "ideal": round(ideal, 4),
                     "E": {str(k): round(v, 4) for k, v in E.items()},
                     "retained": bool(retained)}
        ctx.log.info(f"  [{tag}] ideal={ideal:+.3f} "
                     f"E1={E[1.0]:+.3f} E3={E[3.0]:+.3f} E5={E[5.0]:+.3f} "
                     f"{'RETAINED' if retained else 'destroyed'}")

    retained_cfg = next((v["cfg"] for k, v in scan.items()
                         if v["retained"] and k != "destroyed"), None)
    choice = {
        "destroyed": TC_DESTROYED,
        "retained": retained_cfg,        # may be None if none retained
        "retained_found": retained_cfg is not None,
        "scan": scan,
    }
    ctx.regime = choice
    eqe.write_json(ctx.outdir / "regime_choice.json", choice)
    return choice


def j4_depth(ctx: Context) -> dict:
    """TC1 / TC3 / TC5 genuine vs garbage on the retained config (if found)."""
    base_cfg = (ctx.regime or {}).get("retained") or TC_RETAINED_CANDIDATES[0]
    results = {}
    for steps in (1, 3, 5):
        cfg = dict(base_cfg, steps=steps)
        base = _tc(cfg)
        res = run_gg(ctx, f"j4_depth_tc{steps}", base, SPECTRA["kim"],
                     ctx.chain(4))
        results[f"tc{steps}"] = {"rho_genu": res.get("rho_genu"),
                                 "rho_garb": res.get("rho_garb"),
                                 "E_ideal": res.get("E_ideal")}
    eqe.write_json(ctx.outdir / "j4_depth_overview.json", results)
    return results


def j6_nullmodels(ctx: Context) -> dict:
    """Genuine / garbage / identity null-model control."""
    cfg = (ctx.regime or {}).get("destroyed", TC_DESTROYED)
    base = _tc(cfg)
    res = run_gg(ctx, "j6_nullmodels", base, SPECTRA["kim"], ctx.chain(4),
                 methods=("genuine", "garbage", "identity"))
    return res


# ═══════════════════════════════════════════════════════════════════════
#  Drift and timing controls (j8/j9 short window, j12/j13 long window)
# ═══════════════════════════════════════════════════════════════════════

def _run_drift_design(ctx: Context, name: str, reps: int,
                      delay_s: float) -> dict:
    """Core of the blocked-vs-interleaved A/B drift study (shared by j8/j12)."""
    layout = ctx.chain(4)
    cfg = (ctx.regime or {}).get("destroyed", TC_DESTROYED)
    base = _tc(cfg)
    ideal = compute_ideal_expectation(base)
    qc_t = eqe.transpile_fixed(base, ctx.backend, initial_layout=layout)
    qc = qc_t.copy()
    eqe.add_measurements(qc, layout, base.num_qubits)
    n = base.num_qubits
    ctx.log.info(f"[{name}] layout={layout[:n]} ideal={ideal:+.4f} "
                 f"reps={reps} delay={delay_s}s (blocked + interleaved)")

    raw_path = ctx.outdir / f"{name}_raw.csv"
    done = eqe.completed_keys(raw_path, ("schedule", "label", "rep"))

    def record(schedule: str, label: str, rep: int) -> float:
        counts = _run_batch(ctx, name, f"{schedule}_{label}", rep, [qc],
                            ctx.shots)[0]
        ev = eqe.expectation_from_counts(counts, n)
        eqe.append_rows(raw_path, [{
            "job": name, "schedule": schedule, "label": label,
            "rep": rep, "n_qubits": n, "exp_val": f"{ev:.6f}",
            "n_shots": ctx.shots, "ideal": f"{ideal:.6f}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }])
        if delay_s and not ctx.local:
            time.sleep(delay_s)
        return ev

    # blocked: all A, then all B (maximally time-separated)
    for label in ("A", "B"):
        for rep in range(reps):
            if ("blocked", label, str(rep)) in done:
                continue
            ev = record("blocked", label, rep)
            ctx.log.info(f"  [blocked {label} {rep + 1:>3d}/{reps}] E={ev:+.4f}")
    # interleaved: A,B adjacent within each rep
    for rep in range(reps):
        for label in ("A", "B"):
            if ("interleaved", label, str(rep)) in done:
                continue
            ev = record("interleaved", label, rep)
            ctx.log.info(f"  [interlv {label} {rep + 1:>3d}/{reps}] E={ev:+.4f}")

    return _aggregate_drift_design(ctx, raw_path)


def j8_drift_design(ctx: Context) -> dict:
    """A/B acquisition-order study — *blocked* vs *interleaved*.

    Two nominally-identical genuine TC1 ``lambda=1`` measurements (labels
    ``A`` and ``B``) are collected under two schedules:

      * **blocked**     — all ``A`` reps first, then all ``B`` reps (the two
        blocks are separated by minutes of wall-clock time);
      * **interleaved** — ``A, B, A, B, ...`` so each pair is adjacent in time.

    Because ``A`` and ``B`` are the *same* circuit, any ``E_A - E_B`` is pure
    bias (slow device drift + shot noise).  Under drift the blocked schedule
    shows a non-zero mean / inflated spread of the difference while the
    interleaved schedule cancels it — a direct hardware demonstration that the
    *acquisition order* alone can bias an A/B method comparison.
    """
    return _run_drift_design(ctx, "j8_drift_design", max(ctx.reps, 40), 0.0)


def _aggregate_drift_design(ctx: Context, raw_path: Path) -> dict:
    import collections
    import csv as _csv
    name = raw_path.name.replace("_raw.csv", "")
    ev = collections.defaultdict(dict)   # (schedule) -> {(label,rep): E}
    for row in _csv.DictReader(open(raw_path)):
        ev[row["schedule"]][(row["label"], int(row["rep"]))] = float(row["exp_val"])

    out = {"backend": _bname(ctx), "metric": "E_A - E_B (pure bias)"}
    for schedule in ("blocked", "interleaved"):
        d = ev.get(schedule, {})
        reps = sorted({r for (_, r) in d})
        deltas = [d[("A", r)] - d[("B", r)] for r in reps
                  if ("A", r) in d and ("B", r) in d]
        A = [d[("A", r)] for r in reps if ("A", r) in d]
        B = [d[("B", r)] for r in reps if ("B", r) in d]
        if deltas:
            out[f"{schedule}_mean_delta"] = round(float(np.mean(deltas)), 5)
            out[f"{schedule}_std_delta"] = round(float(np.std(deltas, ddof=1)), 5)
            out[f"{schedule}_abs_mean_delta"] = round(float(np.mean(np.abs(deltas))), 5)
        if A and B:
            out[f"{schedule}_blockmean_A"] = round(float(np.mean(A)), 5)
            out[f"{schedule}_blockmean_B"] = round(float(np.mean(B)), 5)
            out[f"{schedule}_blockbias"] = round(float(np.mean(A) - np.mean(B)), 5)
    eqe.write_json(ctx.outdir / f"{name}_summary.json", out)
    ctx.log.info(f"  drift bias: blocked blockbias="
                 f"{out.get('blocked_blockbias')} vs interleaved "
                 f"{out.get('interleaved_blockbias')}")
    return out


def _run_allan_probe(ctx: Context, name: str, samples: int, shots: int,
                     delay_s: float) -> dict:
    """Core of the single-circuit drift probe (shared by j9/j13)."""
    layout = ctx.chain(4)
    cfg = (ctx.regime or {}).get("destroyed", TC_DESTROYED)
    base = _tc(cfg)
    ideal = compute_ideal_expectation(base)
    qc_t = eqe.transpile_fixed(base, ctx.backend, initial_layout=layout)
    eqe.add_measurements(qc_t, layout, base.num_qubits)
    n = base.num_qubits
    ctx.log.info(f"[{name}] layout={layout[:n]} ideal={ideal:+.4f} "
                 f"samples={samples} shots={shots} delay={delay_s}s")

    raw_path = ctx.outdir / f"{name}_raw.csv"
    done = eqe.completed_keys(raw_path, ("rep",))
    for rep in range(samples):
        if (str(rep),) in done:
            continue
        counts = _run_batch(ctx, name, "probe", rep, [qc_t], shots)[0]
        ev = eqe.expectation_from_counts(counts, n)
        eqe.append_rows(raw_path, [{
            "job": name, "rep": rep, "n_qubits": n,
            "exp_val": f"{ev:.6f}", "n_shots": shots, "ideal": f"{ideal:.6f}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }])
        if (rep + 1) % 20 == 0:
            ctx.log.info(f"  [probe {rep + 1:>3d}/{samples}] E={ev:+.4f}")
        if delay_s and not ctx.local:
            time.sleep(delay_s)

    return _aggregate_allan(ctx, raw_path)


def j9_allan_probe(ctx: Context) -> dict:
    """Single-circuit drift probe → Allan deviation & autocorrelation.

    A fixed genuine TC1 ``lambda=1`` circuit is measured many times in quick
    succession, each sample time-stamped.  From the resulting ``E(t)`` time
    series we estimate the **overlapping Allan deviation** ``sigma_A(tau)`` and
    the lag-1 autocorrelation, which quantify the device's drift timescale and
    hence a principled lower bound on the number of repetitions / the
    re-randomisation interval needed for an unbiased benchmark.
    """
    return _run_allan_probe(ctx, "j9_allan_probe",
                            max(ctx.reps * 5, 150), min(ctx.shots, 2048), 0.0)


def _aggregate_allan(ctx: Context, raw_path: Path) -> dict:
    import csv as _csv
    name = raw_path.name.replace("_raw.csv", "")
    rows = sorted((int(r["rep"]), float(r["exp_val"]))
                  for r in _csv.DictReader(open(raw_path)))
    y = np.array([v for _, v in rows])
    out = {"backend": _bname(ctx), "n_samples": int(len(y))}
    if len(y) >= 4:
        out["mean"] = round(float(np.mean(y)), 5)
        out["std"] = round(float(np.std(y, ddof=1)), 5)
        # lag-1 autocorrelation
        yc = y - y.mean()
        denom = float(np.dot(yc, yc))
        out["autocorr_lag1"] = (round(float(np.dot(yc[:-1], yc[1:]) / denom), 4)
                                if denom > 0 else float("nan"))
        # overlapping Allan deviation over a range of averaging windows m
        allan = {}
        for m in (1, 2, 4, 8, 16, 32, 64):
            if len(y) < 2 * m + 1:
                break
            # block-averaged series at window m
            k = len(y) // m
            ybar = y[:k * m].reshape(k, m).mean(axis=1)
            d = np.diff(ybar)
            if len(d) > 0:
                allan[str(m)] = round(float(np.sqrt(0.5 * np.mean(d ** 2))), 6)
        out["allan_dev"] = allan
    eqe.write_json(ctx.outdir / f"{name}_summary.json", out)
    ctx.log.info(f"  Allan: std={out.get('std')} "
                 f"autocorr_lag1={out.get('autocorr_lag1')} "
                 f"allan_dev={out.get('allan_dev')}")
    return out


def j11_czne_dataset(ctx: Context) -> dict:
    """Genuine / garbage / identity regime sweep for the negative-probability
    diagnostic (15 reps per regime/method).

    Runs genuine / garbage / identity folding over a taxonomy grid
    (retained → mid → destroyed) at ``{1,3,5}``.  Each :func:`run_gg` call
    stores per-rep ``exp_val`` *and* full counts, from which the
    negative-probability weight W_neg and the E(lambda) retention regime are
    computed offline.  Provides the per-regime hardware evidence for the
    negative-probability diagnostic in the paper.
    """
    grid = {
        "retained":  (ctx.regime or {}).get("retained") or TC_RETAINED_CANDIDATES[0],
        "mid":       {"rx": 0.07, "rz": 0.09, "steps": 1},
        "destroyed": (ctx.regime or {}).get("destroyed", TC_DESTROYED),
    }
    ds_ctx = Context(backend=ctx.backend, outdir=ctx.outdir, log=ctx.log,
                     reps=max(ctx.reps // 2, 15), shots=ctx.shots,
                     local=ctx.local, seed=ctx.seed,
                     layouts=ctx.layouts, regime=ctx.regime, _sim=ctx._sim)
    results = {}
    for tag, cfg in grid.items():
        ctx.log.info(f"[j11_czne_dataset] regime={tag} cfg={cfg}")
        base = _tc(cfg)
        res = run_gg(ds_ctx, f"j11_czne_{tag}", base, SPECTRA["kim"],
                     ctx.chain(4), methods=("genuine", "garbage", "identity"))
        results[tag] = {"E_ideal": res.get("E_ideal"),
                        "rho_genu": res.get("rho_genu"),
                        "rho_iden": res.get("rho_iden"),
                        "rho_garb": res.get("rho_garb")}
    eqe.write_json(ctx.outdir / "j11_czne_overview.json", results)
    return results


# ═══════════════════════════════════════════════════════════════════════
#  Long-duration follow-ups (j12–j14): drift over hours + big regime sweep
# ═══════════════════════════════════════════════════════════════════════

def j12_drift_design_long(ctx: Context) -> dict:
    """Long-window rerun of the blocked/interleaved A/B drift study (j8).

    Identical design to :func:`j8_drift_design` but with more reps and a small
    inter-sample delay so the *blocked* A and B blocks are separated by tens of
    minutes — giving slow device drift the time it needs to bias the blocked
    schedule (the short j8 run sampled only ~18 min and saw little drift).
    """
    return _run_drift_design(ctx, "j12_drift_design_long",
                             max(ctx.reps, LONG_DRIFT_REPS), LONG_SAMPLE_DELAY_S)


def j13_allan_probe_long(ctx: Context) -> dict:
    """Long-window rerun of the Allan-deviation drift probe (j9).

    A longer, more widely-spaced time series than :func:`j9_allan_probe`,
    reaching larger averaging windows ``m``.  If slow drift (a random-walk
    component) is present, ``sigma_A(m)`` stops falling like ``1/sqrt(m)`` and
    flattens / rises at large ``m`` — the signature that distinguishes drift
    from white shot noise and sets the re-randomisation timescale.
    """
    return _run_allan_probe(ctx, "j13_allan_probe_long",
                            max(ctx.reps * 7, LONG_PROBE_SAMPLES),
                            min(ctx.shots, 2048), LONG_SAMPLE_DELAY_S)


def j14_czne_dataset_big(ctx: Context) -> dict:
    """Larger genuine/garbage/identity regime sweep (45 reps per regime/method).

    Same taxonomy grid as :func:`j11_czne_dataset` but with more reps per
    (regime, method), giving the 405-run dataset behind the negative-probability
    weight statistic and the AUC = 1.0 separation reported in the paper.
    """
    grid = {
        "retained":  (ctx.regime or {}).get("retained") or TC_RETAINED_CANDIDATES[0],
        "mid":       {"rx": 0.07, "rz": 0.09, "steps": 1},
        "destroyed": (ctx.regime or {}).get("destroyed", TC_DESTROYED),
    }
    ds_ctx = Context(backend=ctx.backend, outdir=ctx.outdir, log=ctx.log,
                     reps=max(ctx.reps, BIG_DATASET_REPS), shots=ctx.shots,
                     local=ctx.local, seed=ctx.seed + 1,   # fresh seed = new reps
                     layouts=ctx.layouts, regime=ctx.regime, _sim=ctx._sim)
    results = {}
    for tag, cfg in grid.items():
        ctx.log.info(f"[j14_czne_dataset_big] regime={tag} cfg={cfg}")
        base = _tc(cfg)
        res = run_gg(ds_ctx, f"j14_czne_{tag}", base, SPECTRA["kim"],
                     ctx.chain(4), methods=("genuine", "garbage", "identity"))
        results[tag] = {"E_ideal": res.get("E_ideal"),
                        "rho_genu": res.get("rho_genu"),
                        "rho_iden": res.get("rho_iden"),
                        "rho_garb": res.get("rho_garb")}
    eqe.write_json(ctx.outdir / "j14_czne_overview.json", results)
    return results


# ═══════════════════════════════════════════════════════════════════════
#  Registry (scheduling order)
# ═══════════════════════════════════════════════════════════════════════

JOBS = [
    # setup: pick the qubit chain and the retained/destroyed TC configs
    ("j0_health", j0_health),
    ("j1_regime_scan", j1_regime_scan),
    # depth sweep -> rescaling-collapse figure
    ("j4_depth", j4_depth),
    # genuine/garbage/identity null-model control
    ("j6_nullmodels", j6_nullmodels),
    # drift and timing controls (short + long window)
    ("j8_drift_design", j8_drift_design),
    ("j9_allan_probe", j9_allan_probe),
    ("j12_drift_design_long", j12_drift_design_long),
    ("j13_allan_probe_long", j13_allan_probe_long),
    # genuine/garbage/identity regime sweep -> negative-probability diagnostic
    ("j11_czne_dataset", j11_czne_dataset),
    ("j14_czne_dataset_big", j14_czne_dataset_big),
]

JOB_MAP = dict(JOBS)
