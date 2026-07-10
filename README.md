# Reproduction Package: Benchmarking Error Mitigation

This repository contains all scripts, data, and plotting code to reproduce the
experiments and figures presented in the paper:

> **Benchmarking Error Mitigation:
> Artefactual Improvements in Zero-Noise Extrapolation** (QuBench 2026 @ IEEE QCE 2026)

The paper exposes a failure mode of Richardson zero-noise extrapolation (ZNE):
once noise amplification destroys the signal, the extrapolation collapses into a
fixed rescaling of a single noisy measurement and manufactures an apparent
"improvement" that is independent of how the noise was amplified. The claim is
demonstrated in simulation and on the IQM Euro-Q-Exa QC at the LRZ.

## Quick Start

### Docker (recommended)

Build the container, run the full reproduction pipeline, and compile the paper PDF:

```bash
make repro_docker
```

This runs the pipeline inside an isolated Python + R environment and then compiles
the LaTeX source. The final PDF is written to `build/paper/`.

### Local (venv + R)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r reproduction/requirements.txt   # Python: qiskit, qiskit-aer, numpy, scipy, pandas
Rscript -e "install.packages(c('tidyverse', 'scales', 'patchwork', 'tikzDevice'))"
make repro
```

`make repro` regenerates the simulator sweep, reads the checked-in hardware data,
rebuilds both figures, and compiles the paper. A LaTeX install with the Times
fonts (`texlive-fonts-recommended`), LuaLaTeX, and `texlive-publishers`
(IEEEtran) is required.

### Development shell

```bash
make dev   # drops into the Docker dev container with all dependencies available
```

## Structure

```
amplification_error/
├── Makefile
├── docker/                      # Docker image and compose file
├── paper/                       # LaTeX source, bibliography, precompiled plots
└── reproduction/
    ├── scripts/                 # Simulator experiment scripts (called by make)
    │   ├── horoscope_mechanism.py         # depolarising-noise sweep: E(λ) retention
    │   └── make_horoscope_fallback.py      # regenerate the precompiled-plot fallback
    ├── hardware/                # Real-hardware collection scripts (not in Makefile)
    │   ├── run_hardware_experiments.py     # runner for the Euro-Q-Exa jobs
    │   ├── eqe_experiments.py              # job registry (j0,j1,j4,j6,j8,j9,j11–j14)
    │   ├── eqe_common.py                   # folding / transpile / submit helpers
    │   ├── eqe_topology.py                 # low-error linear-chain selection
    │   └── horoscope_qexa.py               # standalone garbage-folding falsification
    ├── R/                       # R plotting scripts (called by make)
    ├── core/                    # Shared Python library (circuits, noise, stats, zne)
    ├── data/                    # Input datasets (git-tracked) — see data/README.md
    │   ├── qexa_hardware.csv               # Euro-Q-Exa hardware point (Fig. horoscope)
    │   ├── horoscope_*.csv                 # cached simulator-sweep outputs
    │   ├── eqe_day/                        # per-job raw + summary hardware data
    │   └── logs/                           # provenance run logs
    ├── img.tex                  # standalone TikZ wrapper (paper-consistent fonts)
    ├── gen_img.sh               # compiles TikZ fragments to standalone PDFs
    └── requirements.txt

build/                           # Generated output — not git-tracked, created by make
├── results/                     # Computed CSV files (input to R plots)
├── plots/                       # Generated TikZ + PDF plots
└── paper/                       # Compiled LaTeX PDF
```

## Make Targets

| Target                | Description                                                   |
|-----------------------|---------------------------------------------------------------|
| `make`                | Compile the paper PDF (uses precompiled plots if present)     |
| `make repro`          | Run full pipeline locally: simulator sweep + plots + paper    |
| `make repro_docker`   | Same as above, but run inside Docker                          |
| `make dev`            | Start a development environment inside Docker                 |
| `make hardware-local` | Run every hardware job on a local simulator (no token)        |
| `make hardware-run`   | Collect the real Euro-Q-Exa data (requires `MQSS_TOKEN`)      |
| `make hardware_qexa`  | Standalone garbage-folding falsification run on `EQE1`        |
| `make clean`          | Remove `build/`                                               |

## Figures

Both figures are inlined into the paper as TikZ (via `\includetikz`), falling
back to `paper/plots_precompiled/` when `build/plots/` is absent.

| Figure               | Script                              | Input                                      |
|----------------------|-------------------------------------|--------------------------------------------|
| `horoscope_sweep`    | `R/plot_horoscope_extrapolation.R`  | simulator sweep + `data/qexa_hardware.csv` |
| `rescaling_collapse` | `R/plot_rescaling_collapse.R`       | `data/eqe_day/j4_depth_tc{1,3,5}_raw.csv`  |

## Hardware Experiments

The scripts used to collect the real-hardware data are provided under
`reproduction/hardware/` for transparency. They are **not** invoked by the
Makefile — the resulting data is already checked into `reproduction/data/eqe_day/`,
and the run logs are in `reproduction/data/logs/`.

All jobs run on the 54-qubit IQM Euro-Q-Exa (`EQE1`) machine at the LRZ via the
[MQSS Qiskit adapter](https://munich-quantum-software-stack.github.io/MQSS-Interfaces/qiskit/index.html),
pinned to a connected, low-error qubit chain, transpiled at
`optimization_level=0`, `4096` shots per circuit.

| Job(s)                              | Experiment                                                     |
|-------------------------------------|----------------------------------------------------------------|
| `j0_health`, `j1_regime_scan`       | chain selection + retained/destroyed regime scan (setup)       |
| `j4_depth`                          | genuine `{1,3,5}` depth sweep → rescaling-collapse figure      |
| `j6_nullmodels`                     | genuine / garbage / identity null-model control                |
| `j8_drift_design`, `j12_..._long`   | blocked-vs-interleaved acquisition (drift bias)                |
| `j9_allan_probe`, `j13_..._long`    | single-circuit drift probe → Allan deviation                   |
| `j11_czne_dataset`, `j14_..._big`   | genuine/garbage/identity regime sweep → negative-prob. weight  |

Set the token and run either the whole batch or a single job:

```bash
export MQSS_TOKEN="<your-token>"        # or put it in a .env file

# Whole batch on real hardware (resumable):
python reproduction/hardware/run_hardware_experiments.py --reps 30 --shots 4096

# A single job:
python reproduction/hardware/run_hardware_experiments.py --only j4_depth

# Local simulator smoke test (no token needed):
python reproduction/hardware/run_hardware_experiments.py --local-test --only j6_nullmodels
```

The standalone garbage-folding falsification (Fig. horoscope hardware point) can
also be run on its own:

```bash
python reproduction/hardware/horoscope_qexa.py --reps 30 --shots 4096   # or --local-test
```

The broader Euro-Q-Exa "exclusive day" batch (all jobs, including experiments not
used in this paper) is archived separately in `../eqe_exclusive_day/`.
