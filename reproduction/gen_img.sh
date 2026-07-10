#!/usr/bin/env bash
# Compile all TikZ .tex fragments into standalone PDFs with the IEEEtran document style.
# Called from the project root (do_we_measure_what_we_claim/) by `make compile_plots`.
# Usage: gen_img.sh [plots-dir]   (default: build/plots)

set -e

PLOTS_DIR="${1:-build/plots}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMG_TEX="$(realpath "$SCRIPT_DIR/img.tex")"
ABS_PLOTS="$(realpath "$PLOTS_DIR")"

for file in "$ABS_PLOTS"/*.tex; do
    [ -f "$file" ] || continue
    job_name="$(basename "$file" .tex)"
    echo "  Compiling: ${job_name}.tex → ${job_name}.pdf"
    printf '\\newcommand{\\path}{%s}\\input{%s}\n' "$file" "$IMG_TEX" | \
        lualatex -interaction=batchmode -output-directory="$ABS_PLOTS" -jobname="$job_name"
    rm -f "$ABS_PLOTS/${job_name}.aux" "$ABS_PLOTS/${job_name}.log"
done
