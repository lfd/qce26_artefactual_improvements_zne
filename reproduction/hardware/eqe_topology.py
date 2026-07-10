"""
Static topology snapshot — IQM Euro-Q-Exa (EQE1).
=================================================

Offline fallback description of the 54-qubit EQE1 lattice, transcribed from
the ``qiskit`` calibration map exported on **2026-05-24** (see
``amplification_error/QPU_Euro-Q-Exa_Dominik.png``).

This module is a *fallback*: ``eqe_common.pick_linear_chain`` prefers the
**live** coupling map and calibration pulled from the backend at run time.
The static data here is used (a) for ``--local-test`` runs where no backend
is available, and (b) to validate that a hand-chosen ``initial_layout`` is
actually a connected nearest-neighbour path before submitting a job with
``no_modify=True`` (which disables any server-side re-routing).

Lattice geometry
----------------
The 54 qubits sit on a square grid (drawn as a diamond in the calibration
figure).  Every qubit has integer ``(row, col)`` coordinates; two qubits are
connected iff they are nearest neighbours on that grid (Δrow+Δcol == 1).
The coordinate table below was reconstructed from the figure and verified to
reproduce **every** horizontal edge label, so the generated coupling map is
exact.

Calibration caveats (2026-05-24 snapshot)
-----------------------------------------
* Qubit ``33`` is dead — all four of its edges read ``N/A``.  It is listed in
  :data:`DEAD_QUBITS` and excluded from every candidate layout.
* The weakest two-qubit edges are ``(3,4)=0.840``, ``(20,21)=0.883``,
  ``(29,30)=0.893``, ``(48,49)=0.932`` — these should be avoided.
* Physical qubits ``1,2,3,4`` (used in the earlier study) are **not** a
  connected chain: ``1`` is not adjacent to ``2`` and edge ``(3,4)`` is the
  worst on the device.  That mismapping degraded the earlier results.
* The vetted low-error linear chains are :data:`CHAIN4` and :data:`CHAIN5`.
"""

from __future__ import annotations

# ── qubit → (row, col) on the square lattice ─────────────────────────────
# Reconstructed from the 2026-05-24 calibration figure; reproduces every
# horizontal edge label exactly (54 qubits total).
COORDS: dict[int, tuple[int, int]] = {
    # row 0
    51: (0, 3), 52: (0, 4), 53: (0, 5),
    # row 1
    46: (1, 2), 47: (1, 3), 48: (1, 4), 49: (1, 5), 50: (1, 6),
    # row 2
    39: (2, 1), 40: (2, 2), 41: (2, 3), 42: (2, 4), 43: (2, 5), 44: (2, 6), 45: (2, 7),
    # row 3
    31: (3, 1), 32: (3, 2), 33: (3, 3), 34: (3, 4), 35: (3, 5), 36: (3, 6), 37: (3, 7), 38: (3, 8),
    # row 4
    22: (4, 0), 23: (4, 1), 24: (4, 2), 25: (4, 3), 26: (4, 4), 27: (4, 5), 28: (4, 6), 29: (4, 7), 30: (4, 8),
    # row 5
    14: (5, 0), 15: (5, 1), 16: (5, 2), 17: (5, 3), 18: (5, 4), 19: (5, 5), 20: (5, 6), 21: (5, 7),
    # row 6
    7: (6, 1), 8: (6, 2), 9: (6, 3), 10: (6, 4), 11: (6, 5), 12: (6, 6), 13: (6, 7),
    # row 7
    2: (7, 2), 3: (7, 3), 4: (7, 4), 5: (7, 5), 6: (7, 6),
    # row 8
    0: (8, 4), 1: (8, 5),
}

# Dead / unusable qubits (all incident edges N/A in the snapshot).
DEAD_QUBITS: frozenset[int] = frozenset({33})

# ── two-qubit edge fidelities (CZ), 2026-05-24 snapshot ──────────────────
# Keyed by sorted (a, b).  ``None`` marks an N/A (unusable) edge.  Missing
# entries simply have no recorded fidelity; ranking treats them as unknown.
_RAW_EDGE_FIDELITY: dict[tuple[int, int], float | None] = {
    # horizontal
    (0, 1): 0.964,
    (2, 3): 0.940, (3, 4): 0.840, (4, 5): 0.988, (5, 6): 0.993,
    (7, 8): 0.979, (8, 9): 0.990, (9, 10): 0.985, (10, 11): 0.988, (11, 12): 0.989, (12, 13): 0.995,
    (14, 15): 0.991, (15, 16): 0.990, (16, 17): 0.983, (17, 18): 0.967, (18, 19): 0.986, (19, 20): 0.989, (20, 21): 0.883,
    (22, 23): 0.982, (23, 24): 0.992, (24, 25): 0.982, (25, 26): 0.990, (26, 27): 0.958, (27, 28): 0.993, (28, 29): 0.985, (29, 30): 0.893,
    (31, 32): 0.995, (32, 33): None, (33, 34): None, (34, 35): 0.988, (35, 36): 0.993, (36, 37): 0.984, (37, 38): 0.991,
    (39, 40): 0.995, (40, 41): 0.992, (41, 42): 0.985, (42, 43): 0.985, (43, 44): 0.959, (44, 45): 0.960,
    (46, 47): 0.983, (47, 48): 0.985, (48, 49): 0.932, (49, 50): 0.980,
    (51, 52): 0.994, (52, 53): 0.993,
    # vertical
    (47, 51): 0.979, (48, 52): 0.990, (49, 53): 0.995,
    (40, 46): 0.990, (41, 47): 0.985, (42, 48): 0.960, (43, 49): 0.988, (44, 50): 0.980,
    (31, 39): 0.993, (32, 40): 0.997, (33, 41): None, (34, 42): 0.996, (35, 43): 0.992, (36, 44): 0.958, (37, 45): 0.987,
    (25, 33): None,
}


def _norm(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def build_coupling_map() -> list[tuple[int, int]]:
    """Generate the bidirectional coupling map from :data:`COORDS`.

    Two qubits are connected iff they are nearest neighbours on the grid.
    Edges incident to a dead qubit, or explicitly marked ``None``, are
    omitted.  Returns a sorted list of directed ``(a, b)`` pairs (both
    directions), matching Qiskit's ``CouplingMap`` convention.
    """
    pos_to_q = {pos: q for q, pos in COORDS.items()}
    undirected: set[tuple[int, int]] = set()
    for q, (r, c) in COORDS.items():
        if q in DEAD_QUBITS:
            continue
        for dr, dc in ((1, 0), (0, 1)):
            nb = pos_to_q.get((r + dr, c + dc))
            if nb is None or nb in DEAD_QUBITS:
                continue
            edge = _norm(q, nb)
            if _RAW_EDGE_FIDELITY.get(edge, "absent") is None:
                continue  # explicitly N/A
            undirected.add(edge)
    directed: list[tuple[int, int]] = []
    for a, b in sorted(undirected):
        directed.append((a, b))
        directed.append((b, a))
    return directed


def edge_fidelity(a: int, b: int) -> float | None:
    """Recorded CZ fidelity for an edge, or ``None`` if unknown/N/A."""
    return _RAW_EDGE_FIDELITY.get(_norm(a, b))


def neighbours(q: int) -> list[int]:
    """Live (non-dead) nearest neighbours of ``q`` on the static lattice."""
    pos_to_q = {pos: qq for qq, pos in COORDS.items()}
    r, c = COORDS[q]
    out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb = pos_to_q.get((r + dr, c + dc))
        if nb is None or nb in DEAD_QUBITS:
            continue
        if _RAW_EDGE_FIDELITY.get(_norm(q, nb), "absent") is None:
            continue
        out.append(nb)
    return out


# ── vetted default linear chains (low-error, contiguous) ─────────────────
# Row-6 segment: edges (8,9)=0.990 (9,10)=0.985 (10,11)=0.988 (11,12)=0.989.
CHAIN4: list[int] = [8, 9, 10, 11]
CHAIN5: list[int] = [8, 9, 10, 11, 12]

# Alternative chains (used only if CHAIN4/5 unavailable in live calibration).
ALT_CHAINS: dict[int, list[list[int]]] = {
    4: [[8, 9, 10, 11], [40, 41, 42, 43], [23, 24, 25, 26], [14, 15, 16, 17]],
    5: [[8, 9, 10, 11, 12], [39, 40, 41, 42, 43], [22, 23, 24, 25, 26]],
}

DEFAULT_LAYOUTS: dict[str, list[int]] = {
    "chain4": CHAIN4,
    "chain5": CHAIN5,
}


def is_connected_path(layout: list[int],
                      coupling: list[tuple[int, int]] | None = None) -> bool:
    """True iff ``layout`` is a simple nearest-neighbour path (each
    consecutive pair is a live edge)."""
    if coupling is None:
        coupling = build_coupling_map()
    edges = {(_norm(a, b)) for a, b in coupling}
    if len(set(layout)) != len(layout):
        return False
    return all(_norm(layout[i], layout[i + 1]) in edges
               for i in range(len(layout) - 1))


def best_static_chain(n: int) -> list[int]:
    """Best length-``n`` linear chain from the static snapshot.

    Enumerates simple nearest-neighbour paths and ranks them by the sum of
    two-qubit *infidelities* (lower is better), treating unknown fidelities
    as the global mean.  Falls back to :data:`DEFAULT_LAYOUTS`.
    """
    coupling = build_coupling_map()
    adj: dict[int, list[int]] = {}
    for a, b in coupling:
        adj.setdefault(a, []).append(b)

    known = [f for f in _RAW_EDGE_FIDELITY.values() if f is not None]
    mean_fid = sum(known) / len(known) if known else 0.99

    def edge_cost(a: int, b: int) -> float:
        f = edge_fidelity(a, b)
        return 1.0 - (f if f is not None else mean_fid)

    best: tuple[float, list[int]] | None = None

    def dfs(path: list[int], cost: float) -> None:
        nonlocal best
        if len(path) == n:
            if best is None or cost < best[0]:
                best = (cost, list(path))
            return
        for nb in adj.get(path[-1], []):
            if nb in path or nb in DEAD_QUBITS:
                continue
            dfs(path + [nb], cost + edge_cost(path[-1], nb))

    for start in COORDS:
        if start in DEAD_QUBITS:
            continue
        dfs([start], 0.0)

    if best is not None:
        return best[1]
    return DEFAULT_LAYOUTS.get(f"chain{n}", CHAIN4)
