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
    from pycalphad import Database, equilibrium, calculate
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


def phase_stability_range(dbf, components: list,
                            phase: str, T_range: np.ndarray):
    """Compute phase fraction over T range."""
    fracs = []
    for T in T_range:
        cond = {v.T: float(T), v.P: 101325.0}
        res  = calculate(dbf, components, phase, T=float(T), P=101325.0)
        fracs.append(float(res.GM.values.min()))
    return T_range, np.array(fracs)
