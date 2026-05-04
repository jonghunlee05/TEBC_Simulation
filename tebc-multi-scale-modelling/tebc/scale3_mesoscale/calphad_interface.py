"""
CALPHAD interface using pycalphad.

Gibbs energy model (Redlich-Kister):
  G_m = Σ xᵢ °Gᵢ + RT Σ xᵢ ln(xᵢ) + ᵉˣG_m
  ᵉˣG_m^{AB} = x_A x_B Σ_v  ᵛL^{AB} (x_A - x_B)^v
"""

from __future__ import annotations

import numpy as np

try:
    import pycalphad.variables as v
    from pycalphad import Database, calculate, equilibrium
except ImportError:
    raise ImportError("pip install pycalphad")


def load_database(tdb_path: str):
    """Load thermodynamic database (.tdb format)."""
    return Database(tdb_path)


def compute_equilibrium(dbf, components: list,
                         phases: list, conditions: dict,
                         output: str = "GM"):
    """Compute phase equilibrium via pycalphad."""
    return equilibrium(dbf, components, phases, conditions, output=output)


def gibbs_redlich_kister(x_A: float, x_B: float,
                          L_coeffs: list,
                          T: float) -> float:
    """ᵉˣG = x_A x_B Σ_v (a_v + b_v T)(x_A - x_B)^v"""
    Gex = 0.0
    dx  = x_A - x_B
    for v_order, (a_v, b_v) in enumerate(L_coeffs):
        L_v = a_v + b_v * T
        Gex += L_v * dx**v_order
    return x_A * x_B * Gex


def gibbs_energy_minimum(dbf, components: list,
                          phase: str, T_range: np.ndarray):
    """Min G_m of a single phase across `T_range`, sampled via pycalphad
    `calculate` (does *not* solve a multi-phase equilibrium).

    Returns (T_range, G_min). Use this when you want a single-phase
    Gibbs surface; for actual phase fractions use
    `phase_fractions_vs_T` (which calls pycalphad `equilibrium`).
    """
    g_min = []
    for T in T_range:
        res = calculate(dbf, components, phase, T=float(T), P=101325.0)
        g_min.append(float(res.GM.values.min()))
    return T_range, np.array(g_min)


def phase_fractions_vs_T(dbf, components: list, phases: list,
                          composition: dict, T_range: np.ndarray):
    """True multi-phase equilibrium phase fractions over `T_range`.

    `composition` maps component → mole fraction, e.g.
    ``{"YB": 0.4, "SI": 0.3, "O": 0.3}``.

    Returns a dict mapping phase name → ndarray of mole fractions
    aligned with `T_range`. Phases not present at a given T appear as 0.
    """
    conditions = {v.P: 101325.0, v.T: list(map(float, T_range))}
    for comp, x in composition.items():
        conditions[v.X(comp)] = x
    res = equilibrium(dbf, components, phases, conditions, output="NP")
    fractions = {}
    for ph in phases:
        # `NP` is the per-phase mole-fraction; sum over compset and vertex axes.
        try:
            arr = res.NP.sel(Phase=ph).values
            fractions[ph] = np.nan_to_num(arr).reshape(-1)[: len(T_range)]
        except (KeyError, ValueError):
            fractions[ph] = np.zeros(len(T_range))
    return fractions


# Backward-compatible alias for any caller still importing the old name —
# but with a clear warning that it does NOT compute phase fractions.
def phase_stability_range(*args, **kwargs):
    """DEPRECATED: misnamed. Use `gibbs_energy_minimum` (single-phase G
    surface) or `phase_fractions_vs_T` (true equilibrium fractions).
    """
    import warnings
    warnings.warn(
        "phase_stability_range is misnamed — it returns the per-T minimum "
        "of the single-phase Gibbs surface, not phase fractions. Use "
        "`gibbs_energy_minimum` for that or `phase_fractions_vs_T` for "
        "the actual multi-phase equilibrium.",
        DeprecationWarning, stacklevel=2,
    )
    return gibbs_energy_minimum(*args, **kwargs)
