#!/usr/bin/env python3
"""
Hardware experiment runner — IQM Euro-Q-Exa (EQE1).
===================================================

Runs the experiment jobs in :mod:`eqe_experiments` back-to-back on the IQM
Euro-Q-Exa machine (backend ``EQE1``) to collect the paper's hardware evidence.

Design
------
  * **Completion-gated, not clock-gated.**  The next job starts only once the
    previous one has *finished* (completed or failed-after-retries).
  * **Failsafe.**  Every job runs inside its own ``try/except``.  A failing job
    is logged, marked ``failed`` in the run manifest, and the runner
    **continues** to the next job rather than aborting the run.
  * **Periodic persistence.**  Each job streams its raw data to CSV one batch at
    a time (see :func:`eqe_experiments.run_gg`), and the runner rewrites the
    ``run_manifest.json`` after every job.
  * **Resumable.**  Re-running skips jobs already marked ``done`` in the
    manifest and lets partially-finished jobs resume from their per-job CSV.

Usage
-----
  # Local end-to-end smoke test (Aer, no hardware):
  python hardware/run_hardware_experiments.py --local-test

  # Production run on EQE1 (MQSS_TOKEN via env/.env):
  MQSS_TOKEN=... python hardware/run_hardware_experiments.py --reps 30 --shots 4096

  # Resume after an interruption (same --outdir):
  python hardware/run_hardware_experiments.py --reps 30 --shots 4096

  # Run a subset:
  python hardware/run_hardware_experiments.py --only j4_depth j11_czne_dataset
  python hardware/run_hardware_experiments.py --from j4_depth
  python hardware/run_hardware_experiments.py --skip j14_czne_dataset_big
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# ── path setup ──
SCRIPT_DIR = Path(__file__).resolve().parent      # reproduction/hardware/
REPO_DIR = SCRIPT_DIR.parent                       # reproduction/
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_DIR))

import eqe_common as eqe
import eqe_experiments as ex

RESULTS_DIR_DEFAULT = REPO_DIR.parent / "build" / "results" / "eqe_day"
LOGS_DIR = REPO_DIR / "logs"


# ═══════════════════════════════════════════════════════════════════════
#  Logging
# ═══════════════════════════════════════════════════════════════════════

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("eqe")          # shared with eqe_common
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fh = logging.FileHandler(log_dir / "hardware_experiments.log", mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)
    return logger


# ═══════════════════════════════════════════════════════════════════════
#  Manifest
# ═══════════════════════════════════════════════════════════════════════

def load_manifest(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:  # noqa: BLE001 — corrupt manifest: start fresh
            pass
    return {"created": datetime.now(timezone.utc).isoformat(), "jobs": {}}


def save_manifest(path: Path, manifest: dict) -> None:
    manifest["updated"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str))


# ═══════════════════════════════════════════════════════════════════════
#  Scheduler
# ═══════════════════════════════════════════════════════════════════════

def select_jobs(args) -> list[tuple[str, object]]:
    jobs = list(ex.JOBS)
    names = [n for n, _ in jobs]
    if args.only:
        return [(n, ex.JOB_MAP[n]) for n in args.only if n in ex.JOB_MAP]
    start = 0
    if args.from_job and args.from_job in names:
        start = names.index(args.from_job)
    jobs = jobs[start:]
    if args.skip:
        jobs = [(n, fn) for n, fn in jobs if n not in args.skip]
    return jobs


def run_experiments(args) -> None:
    log = setup_logging(LOGS_DIR)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / "run_manifest.json"
    manifest = load_manifest(manifest_path)
    manifest.setdefault("jobs", {})

    if args.local_test and args.reps == eqe.N_REPS_DEFAULT:
        args.reps, args.shots = 4, 512   # light defaults for the smoke test

    jobs = select_jobs(args)

    log.info("=" * 68)
    log.info("  EURO-Q-EXA HARDWARE EXPERIMENTS — runner")
    log.info(f"  backend   : {'LOCAL-SIM' if args.local_test else args.backend}")
    log.info(f"  outdir    : {outdir}")
    log.info(f"  reps/shots: {args.reps}/{args.shots}")
    log.info(f"  queue     : {', '.join(n for n, _ in jobs)}")
    log.info(f"  gating    : completion-based (not clock-based)")
    if args.max_hours:
        log.info(f"  safety cap: {args.max_hours} h (soft)")
    log.info("=" * 68)

    # Connect once; jobs share the backend + a single Context.
    backend = None if args.local_test else eqe.get_qexa_backend(args.backend)
    ctx = ex.Context(backend=backend, outdir=outdir, log=log,
                     reps=args.reps, shots=args.shots, local=args.local_test,
                     seed=args.seed)

    # Re-hydrate layout / regime choices if a previous run persisted them
    _rehydrate(ctx, outdir)

    t_start = time.time()
    for name, fn in jobs:
        rec = manifest["jobs"].get(name, {})
        if rec.get("status") == "done" and not args.force:
            log.info(f"[{name}] already done — skipping (use --force to redo).")
            # still make sure downstream jobs can read layout/regime
            _capture_side_effects(ctx, name, rec)
            continue

        if args.max_hours and (time.time() - t_start) / 3600 >= args.max_hours:
            log.warning(f"Soft time cap {args.max_hours}h reached — stopping "
                        f"before {name}. Re-run to resume.")
            break

        log.info("-" * 68)
        log.info(f"▶ START {name}   (elapsed {(time.time()-t_start)/3600:.2f} h)")
        manifest["jobs"][name] = {"status": "running",
                                  "start": datetime.now(timezone.utc).isoformat()}
        save_manifest(manifest_path, manifest)

        t0 = time.time()
        try:
            result = fn(ctx)
            dt = time.time() - t0
            manifest["jobs"][name].update({
                "status": "done",
                "end": datetime.now(timezone.utc).isoformat(),
                "elapsed_s": round(dt, 1),
                "result": _trim(result),
            })
            log.info(f"✓ DONE  {name}  ({dt/60:.1f} min)")
        except Exception as e:  # noqa: BLE001 — failsafe: isolate each job
            dt = time.time() - t0
            manifest["jobs"][name].update({
                "status": "failed",
                "end": datetime.now(timezone.utc).isoformat(),
                "elapsed_s": round(dt, 1),
                "error": str(e),
            })
            log.error(f"✗ FAIL  {name}: {e}")
            log.debug(traceback.format_exc())
        finally:
            save_manifest(manifest_path, manifest)

    # Final summary
    done = [n for n, r in manifest["jobs"].items() if r.get("status") == "done"]
    failed = [n for n, r in manifest["jobs"].items() if r.get("status") == "failed"]
    log.info("=" * 68)
    log.info(f"RUN COMPLETE — {len(done)} done, {len(failed)} failed, "
             f"{(time.time()-t_start)/3600:.2f} h elapsed")
    if failed:
        log.info(f"  failed: {', '.join(failed)} (re-run to retry)")
    log.info(f"  manifest: {manifest_path}")
    log.info("=" * 68)


def _rehydrate(ctx: ex.Context, outdir: Path) -> None:
    """Reload layout / regime choices from disk so a resumed run can skip j0/j1."""
    lay = outdir / "selected_layout.json"
    if lay.exists():
        try:
            info = json.loads(lay.read_text())
            ctx.layouts = {"chain4": info["chain4"], "chain5": info["chain5"]}
            ctx.log.info(f"  rehydrated layouts: chain4={ctx.layouts['chain4']['layout']}")
        except Exception:  # noqa: BLE001
            pass
    reg = outdir / "regime_choice.json"
    if reg.exists():
        try:
            ctx.regime = json.loads(reg.read_text())
            ctx.log.info("  rehydrated regime choice")
        except Exception:  # noqa: BLE001
            pass


def _capture_side_effects(ctx: ex.Context, name: str, rec: dict) -> None:
    """For already-done j0/j1, make sure ctx still has layouts/regime."""
    if name == "j0_health" and not ctx.layouts:
        _rehydrate(ctx, ctx.outdir)
    if name == "j1_regime_scan" and not ctx.regime:
        _rehydrate(ctx, ctx.outdir)


def _trim(result):
    """Keep the manifest small: store only scalar summary fields."""
    if not isinstance(result, dict):
        return result
    out = {}
    for k, v in result.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        elif isinstance(v, dict):
            out[k] = {kk: vv for kk, vv in v.items()
                      if isinstance(vv, (int, float, str, bool)) or vv is None}
    return out


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default=eqe.BACKEND_NAME)
    p.add_argument("--outdir", default=str(RESULTS_DIR_DEFAULT))
    p.add_argument("--reps", type=int, default=eqe.N_REPS_DEFAULT)
    p.add_argument("--shots", type=int, default=eqe.N_SHOTS_DEFAULT)
    p.add_argument("--seed", type=int, default=eqe.MASTER_SEED)
    p.add_argument("--local-test", action="store_true",
                   help="Run all jobs on a local Aer simulator (no hardware).")
    p.add_argument("--max-hours", type=float, default=None,
                   help="Soft safety cap: stop before a new job once exceeded.")
    p.add_argument("--from", dest="from_job", default=None,
                   help="Start from this job name (inclusive).")
    p.add_argument("--only", nargs="+", default=None,
                   help="Run only these job names.")
    p.add_argument("--skip", nargs="+", default=None,
                   help="Skip these job names.")
    p.add_argument("--force", action="store_true",
                   help="Re-run jobs already marked done in the manifest.")
    args = p.parse_args()
    run_experiments(args)


if __name__ == "__main__":
    main()
