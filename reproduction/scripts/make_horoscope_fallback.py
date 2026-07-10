#!/usr/bin/env python3
"""
Local fallback renderer for the figure ``horoscope_sweep``.
==================================================================

The *canonical* figure is produced by the R/tikzDevice pipeline
(``reproduction/R/plot_horoscope_extrapolation.R`` -> ``build/plots/horoscope_sweep.tex``)
and compiled with LuaLaTeX by ``make repro``.  That toolchain (tidyverse +
tikzDevice) runs inside the project's Docker image.

This script renders an equivalent matplotlib version so the paper still
compiles in environments without the R toolchain: the paper's ``\\includetikz``
macro falls back to ``paper/plots_precompiled/horoscope_sweep.pdf`` whenever
``build/plots/horoscope_sweep.tex`` is absent.

Input :  build/results/horoscope_circuits.csv, reproduction/data/qexa_hardware.csv
Output:  paper/plots_precompiled/horoscope_sweep.pdf
"""

from __future__ import annotations

import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.dirname(HERE)
ROOT = os.path.dirname(REPRO)
CIRC = os.path.join(ROOT, "build", "results", "horoscope_circuits.csv")
QEXA = os.path.join(REPRO, "data", "qexa_hardware.csv")
OUT = os.path.join(ROOT, "paper", "plots_precompiled", "horoscope_sweep.pdf")

LFD = dict(black="#000000", orange="#E69F00", grey="#999999",
           teal="#009371", red="#ED665A", purple="#BEAED4")
LAM = np.array([1.0, 3.0, 5.0])


def lagrange_coeffs(lams, target=0.0):
    K = len(lams)
    c = np.ones(K)
    for i in range(K):
        for j in range(K):
            if i != j:
                c[i] *= (target - lams[j]) / (lams[i] - lams[j])
    return c


def poly_eval(E, lams, xs):
    out = np.zeros_like(xs)
    for i in range(len(lams)):
        li = np.ones_like(xs)
        for j in range(len(lams)):
            if i != j:
                li *= (xs - lams[j]) / (lams[i] - lams[j])
        out += E[i] * li
    return out


def load():
    rows = {}
    with open(CIRC) as f:
        for r in csv.DictReader(f):
            rows[r["circuit"]] = r
    curves = []
    g = rows["Grover 6q"]
    q = rows["QFT Mirror 6q"]
    t = rows["Trotter 4q"]
    curves.append(("QFT Mirror 6q", [float(q[k]) for k in ("E1", "E3", "E5")],
                   float(q["E_ideal"]), LFD["black"], "solid", "^"))
    curves.append(("Grover 6q (genuine)", [float(g[k]) for k in ("E1", "E3", "E5")],
                   float(g["E_ideal"]), LFD["orange"], "solid", "o"))
    curves.append(("Grover 6q (garbage)", [float(g[k]) for k in ("E1_garb", "E3_garb", "E5_garb")],
                   float(g["E_ideal"]), LFD["grey"], "dotted", "o"))
    curves.append(("Trotter 4q", [float(t[k]) for k in ("E1", "E3", "E5")],
                   float(t["E_ideal"]), LFD["teal"], "solid", "s"))
    if os.path.exists(QEXA):
        with open(QEXA) as f:
            hw = next(csv.DictReader(f))
        curves.append(("QExa hardware", [float(hw[k]) for k in ("E1", "E3", "E5")],
                       float(hw["E_ideal"]), LFD["red"], "solid", "D"))
    return curves


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    curves = load()
    c_rich = lagrange_coeffs(LAM)

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    xs_in = np.linspace(1.0, 5.0, 200)
    xs_ex = np.linspace(-0.3, 1.0, 80)

    ax.axhline(1 / 64, color=LFD["grey"], ls=":", lw=0.7)
    ax.axhline(0.0, color="0.8", lw=0.5)
    ax.axvline(0.0, color="0.85", lw=0.5)

    handles = []
    for name, E, ideal, col, ls, mk in curves:
        E = np.array(E)
        ax.plot(xs_in, poly_eval(E, LAM, xs_in), color=col, ls=ls, lw=1.3)
        ax.plot(xs_ex, poly_eval(E, LAM, xs_ex), color=col, ls="--", lw=0.9, alpha=0.6)
        ax.plot(LAM, E, marker=mk, color=col, ls="none", ms=4)
        ax.plot(0.0, float(np.dot(c_rich, E)), marker="*", color=col, ms=8, ls="none")
        if "garbage" not in name:
            ax.plot(-0.22, ideal, marker="x", color=col, ms=5, ls="none")
        handles.append(Line2D([0], [0], color=col, ls=ls, marker=mk, ms=4, label=name))

    ax.text(5.05, 1 / 64, r"$1/2^n$", fontsize=6, color=LFD["grey"], va="bottom")
    ax.set_xlabel(r"Scale factor $\lambda$")
    ax.set_ylabel(r"$E(\lambda)$")
    ax.set_xlim(-0.45, 5.5)
    ax.set_ylim(-0.2, 1.12)
    ax.set_xticks(range(0, 6))
    ax.legend(handles=handles, fontsize=5.2, loc="upper right",
              frameon=False, ncol=1, handlelength=1.6, labelspacing=0.25)
    ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
