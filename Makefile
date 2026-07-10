OUTPUT = build
JOB = amplification_error
.PHONY: all clean repro plots compile_plots \
        hardware_qexa hardware-local hardware-run

# Programs
COMPOSE = docker/docker-compose.yml
DC      = docker compose
PY      = python
R       = Rscript

# Directories
D_RESULTS      = $(OUTPUT)/results
D_PLOTS        = $(OUTPUT)/plots
D_PAPER        = $(OUTPUT)/paper

D_REPRODUCTION = ./reproduction
D_R            = $(D_REPRODUCTION)/R
D_SCRIPTS      = $(D_REPRODUCTION)/scripts
D_DATA         = $(D_REPRODUCTION)/data

OUTDIRS = $(D_RESULTS) $(D_PLOTS) $(D_PAPER)

# ------------------------------------------------------------------ #
# Paper dependencies                                                   #
# Add generated .tex plot targets to RAW_PLOTS as the paper grows.    #
# ------------------------------------------------------------------ #
RAW_PLOTS = horoscope_sweep rescaling_collapse
PLOTS     = $(addprefix $(D_PLOTS)/,$(addsuffix .tex,$(RAW_PLOTS)))

HOROSCOPE_RESULTS = $(D_RESULTS)/horoscope_circuits.csv \
                    $(D_RESULTS)/horoscope_shots.csv \
                    $(D_RESULTS)/horoscope_spectrum.csv \
                    $(D_RESULTS)/horoscope_sweep.csv

# ------------------------------------------------------------------ #
# Top-level rules                                                      #
# ------------------------------------------------------------------ #
# NOTE: the paper PDF depends only on main.tex (figures are inlined via
# \includetikz, falling back to paper/plots_precompiled/ when build/plots/
# is absent).  Regenerate figures/data explicitly with `make repro`.
all: $(D_PAPER)/$(JOB).pdf

$(D_PAPER)/$(JOB).pdf: paper/main.tex | $(OUTDIRS)
	BIBINPUTS=paper:$$BIBINPUTS latexmk -shell-escape -lualatex \
		-interaction=nonstopmode \
	    -output-directory=$(D_PAPER) -jobname=$(JOB) $<

dev: $(COMPOSE)
	$(DC) -f $^ run --rm $@

repro_docker: $(COMPOSE)
	$(DC) -f $^ build
	$(DC) -f $^ run --rm repro
	make

repro: plots
	@echo "Reproduction up to date!"

$(OUTDIRS):
	mkdir -p $@

plots: $(PLOTS) compile_plots

compile_plots: $(PLOTS) | $(D_PLOTS)
	@if [ -f "$(D_REPRODUCTION)/gen_img.sh" ]; then \
	    $(D_REPRODUCTION)/gen_img.sh $(D_PLOTS); \
	fi

# ------------------------------------------------------------------ #
# Layer-3 (Horoscope Effect) experiment + figure                       #
# ------------------------------------------------------------------ #
# Simulator sweep (Grover / QFT mirror / Trotter): produces all four CSVs.
$(HOROSCOPE_RESULTS) &: $(D_SCRIPTS)/horoscope_mechanism.py | $(OUTDIRS)
	$(PY) $< --backend simulator --outdir $(D_RESULTS)

# Main Layer-3 figure: E(lambda) retention taxonomy + garbage Grover +
# real EQE1 hardware point (data/qexa_hardware.csv).
$(D_PLOTS)/horoscope_sweep.tex &: $(D_R)/plot_horoscope_extrapolation.R \
                                  $(HOROSCOPE_RESULTS) \
                                  $(D_DATA)/qexa_hardware.csv | $(OUTDIRS)
	$(R) $<

# Genuine-folding rescaling-collapse figure: genuine {1,3,5} depth sweep showing
# Richardson -> 15/8 E(lambda1) as the signal reaches the parity floor (j4).
$(D_PLOTS)/rescaling_collapse.tex: $(D_R)/plot_rescaling_collapse.R \
                                   $(D_DATA)/eqe_day/j4_depth_tc1_raw.csv | $(OUTDIRS)
	$(R) $<

# ------------------------------------------------------------------ #
# Optional: garbage-folding falsification on real hardware (EQE1).      #
# Requires MQSS_TOKEN; not part of the default `repro` target.          #
# ------------------------------------------------------------------ #
hardware_qexa: | $(OUTDIRS)
	$(PY) $(D_REPRODUCTION)/hardware/horoscope_qexa.py --reps 30 --shots 4096 \
	    --outdir $(D_RESULTS)

# ------------------------------------------------------------------ #
# Full hardware experiment run on Euro-Q-Exa (EQE1).                    #
# hardware-local runs every job on a local simulator (no token);       #
# hardware-run collects the real data (needs MQSS_TOKEN via env/.env). #
# ------------------------------------------------------------------ #
hardware-local: | $(OUTDIRS)
	$(PY) $(D_REPRODUCTION)/hardware/run_hardware_experiments.py --local-test \
	    --outdir $(D_RESULTS)/eqe_day

hardware-run: | $(OUTDIRS)
	$(PY) $(D_REPRODUCTION)/hardware/run_hardware_experiments.py --reps 30 --shots 4096 \
	    --outdir $(D_RESULTS)/eqe_day

clean:
	rm -rf build
