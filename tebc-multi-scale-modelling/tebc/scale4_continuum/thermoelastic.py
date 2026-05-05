"""
Coupled thermoelastic BVP using FEniCSx (dolfinx).

Heat:  ρ c_p ∂T/∂t = ∇·(κ∇T) + Q
Mech:  ∇·σ + b = 0,  σ = C:(ε - α ΔT - ε_p)

CTE mismatch stress (bilayer): σ_f = E_f/(1-ν_f) * (α_s - α_f) * ΔT
"""

from __future__ import annotations


def bilayer_mismatch_stress(E_f: float, nu_f: float,
                             alpha_f: float, alpha_s: float,
                             dT: float) -> float:
    """Biaxial CTE mismatch stress: σ_f = E_f/(1-ν_f)(α_s-α_f)ΔT."""
    return (E_f / (1.0 - nu_f)) * (alpha_s - alpha_f) * dT


def stoney_curvature(sigma_f: float, h_f: float,
                      E_s: float, nu_s: float, h_s: float) -> float:
    """Stoney: κ = 6 σ_f h_f (1-ν_s) / (E_s h_s²)"""
    return 6.0 * sigma_f * h_f * (1.0 - nu_s) / (E_s * h_s**2)


def energy_release_rate_steady_state(sigma0: float, h_f: float,
                                      E_f: float, nu_f: float) -> float:
    """G_ss = (1 - ν_f²) σ₀² h_f / (2 E_f). Hutchinson & Suo 1992."""
    return (1.0 - nu_f**2) * sigma0**2 * h_f / (2.0 * E_f)


def convective_bc_heat_flux(T_surface: float, T_inf: float,
                              h_conv: float, emissivity: float,
                              T_rad: float = None) -> float:
    """q″ = h(T - T∞) + ε σ_SB (T⁴ - T_rad⁴)"""
    from tebc.constants import sigma_SB
    T_rad = T_rad if T_rad is not None else T_inf
    return h_conv*(T_surface - T_inf) + emissivity*sigma_SB*(T_surface**4 - T_rad**4)


#
# `fenics_thermoelastic_setup` was removed: it returned a string of
# FEniCSx code that was never executed and was misleading the README /
# spec into claiming Scale 4 FEA exists. A real coupled thermo-elastic
# FEM driver belongs in its own module (e.g. `tebc/scale4_continuum/
# fem_solver.py`) once dolfinx is wired into the pipeline. See spec
# §7 and KNOWN_LIMITATIONS.md.
