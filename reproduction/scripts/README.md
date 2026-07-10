# Reproduction Scripts

## Simulation (no credentials required)

- `horoscope_mechanism.py` — garbage-folding falsification in simulation
  (Grover / QFT mirror / Trotter). Produces `horoscope_circuits.csv`,
  `horoscope_spectrum.csv`, `horoscope_sweep.csv`, `horoscope_shots.csv`.
- `make_horoscope_fallback.py` — renders a matplotlib preview of the main
  figure to `paper/plots_precompiled/horoscope_sweep.pdf` (local fallback used
  when the canonical R/tikz figure has not been built).

## Hardware (`../hardware/`)

- `horoscope_qexa.py` — garbage-folding falsification on the IQM Euro-Q-Exa
  machine (`EQE1`) via the MQSS adapter. Requires `MQSS_TOKEN` (see
  `../.env.example`). Crash-resumable; supports `--local-test` for a local
  Aer smoke test.

## Usage

```bash
# Full simulation reproduction + figure (canonical figure needs R + tikzDevice):
make repro

# Or directly:
python reproduction/scripts/horoscope_mechanism.py --backend simulator --outdir build/results

# Hardware falsification on EQE1 (deploy in tmux; never commit your token):
MQSS_TOKEN=... python reproduction/hardware/horoscope_qexa.py --reps 30 --shots 4096

# Local smoke test of the hardware script (no credentials):
python reproduction/hardware/horoscope_qexa.py --local-test
```
