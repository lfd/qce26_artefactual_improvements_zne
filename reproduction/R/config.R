#!/usr/bin/env Rscript
# =============================================================================
# Shared Configuration — LfD Colour Scheme, Packages, Theme & TikZ Export
# =============================================================================
# Source this from every plotting script:
#   .script_dir <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) {
#     args <- commandArgs(trailingOnly = FALSE)
#     m <- grep("^--file=", args, value = TRUE)
#     if (length(m)) dirname(normalizePath(sub("^--file=", "", m))) else "R"
#   })
#   source(file.path(.script_dir, "config.R"))
# =============================================================================

# ── Packages ──
library(tidyverse)
library(scales)
if (!require(patchwork, quietly = TRUE)) {
  message("patchwork not installed — multi-panel plots will be skipped")
}
library(tikzDevice)
options(tikzDefaultEngine = "luatex",
	tikzLualatexPackages = c(
		getOption("tikzLualatexPackages"),
		"\\usepackage{amsmath}"),
	tikzDocumentDeclaration = "\\documentclass[10pt,conference]{IEEEtran}"
)

# ── Layout constants (IEEEtran column/text widths) ──
INCH.PER.CM <- 1/2.54
TEXTWIDTH <- 18.13275*INCH.PER.CM
COLWIDTH <- 8.85553*INCH.PER.CM
HEIGHT <- 23.61475*INCH.PER.CM
BASE.SIZE <- 9
SMALL.SIZE <- 7
SYM.SIZE <- 1.2 ## Symol size in legends
LINE.SIZE <- 1
POINT.SIZE <- 0.5

# ── Output directories ──
COLOURS.LIST <- c("black", "#E69F00", "#999999", "#009371", "#ed665a", "#1f78b4", "#009371", "#beaed4")

results_dir <- "./build/results"

# ── LfD Colour Scheme ──
LFD <- list(
  black  = "#000000",
  orange = "#E69F00",
  grey   = "#999999",
  teal   = "#009371",
  red    = "#ED665A",
  blue   = "#1F78B4",
  purple = "#BEAED4"
)
COLOURS.LIST <- c(LFD$black, LFD$orange, LFD$grey, LFD$teal,
                  LFD$red, LFD$blue, LFD$purple)

# ── Paper-quality ggplot theme (matches IEEEtran styling) ──
theme_paper <- function(base_size = BASE.SIZE) {
  theme_bw(base_size = base_size) +
    theme(
      strip.background = element_rect(colour = "black", fill = "white"),
      axis.title.x = element_text(size = base_size),
      axis.title.y = element_text(size = base_size),
      # axis.text = element_text(size = base_size - 1),
      legend.title = element_text(size = base_size),
      # legend.text = element_text(size = base_size - 1),
      legend.position = "top",
      # panel.grid.minor = element_blank(),
      # panel.grid.major = element_line(colour = "grey90", linewidth = 0.3),
      # strip.text = element_text(face = "bold", size = base_size),
      plot.margin = margin(t = 2, r = 1, b = 0, l = 0, unit = "mm")
    )
}

shrink_legend <- function(boxc=-5) {
    return(theme(legend.margin=margin(0,0,0,0),
                 legend.box.margin=margin(boxc,boxc,boxc,boxc)))
}


# ── Helper: LaTeX-safe percent labels (for tikz with sanitize=FALSE) ──
latex_percent <- function(x) paste0(x, "\\%")

# ── Helper: save plot as TikZ (standalone PDFs compiled separately by `make compile_plots`) ──
save_plot <- function(g, name, width = COLWIDTH, height = 0.7 * COLWIDTH) {
  dir <- 'build/plots/'
  # PDF preview disabled — proper standalone PDFs are built from TikZ by `make compile_plots`
  # pdf(paste0(dir, name, ".pdf"), width = width, height = height)
  # print(g)
  # dev.off()
  # TikZ export (for paper-consistent fonts)
  tikz(paste0(dir, name, ".tex"), width = width, height = height, sanitize = FALSE)
  print(g)
  dev.off()
}
