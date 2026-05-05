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
    """Stoney-limit biaxial CTE mismatch stress.

        σ_f = E_f' · (α_s − α_f) · ΔT,    E_f' = E_f / (1 − ν_f)

    Valid only when h_film/h_substrate ≪ 1 (≤ 0.1 in practice). For
    finite thickness ratios use `bilayer_mismatch_stress_hsueh`, which
    reduces to this Stoney form in the thin-film limit.
    """
    return (E_f / (1.0 - nu_f)) * (alpha_s - alpha_f) * dT


def bilayer_mismatch_stress_hsueh(E_f: float, nu_f: float, h_f: float,
                                   E_s: float, nu_s: float, h_s: float,
                                   alpha_f: float, alpha_s: float,
                                   dT: float) -> float:
    """General bilayer biaxial film-stress (uniform-strain bound).

    Returns σ_f only. The companion substrate stress is recoverable
    from force balance σ_s · h_s = −σ_f · h_f. Hsueh's full treatment
    (JAP 91 9652, 2002) also includes a *bending* term for free-standing
    plates that is omitted here — appropriate when the system is
    constrained against curvature (typical for coatings on thick
    components) but adds error of order (h_f/h_s)² for a free plate.

    Force balance + shared in-plane strain give:

        ε_common = (E_f' h_f α_f + E_s' h_s α_s) / (E_f' h_f + E_s' h_s) · ΔT
        σ_f      = E_f' · (ε_common − α_f · ΔT)
                 = E_f' · E_s' · h_s · (α_s − α_f) · ΔT
                   / (E_f' h_f + E_s' h_s)

    Limits:
      h_f/h_s → 0:  σ_f → E_f' (α_s − α_f) ΔT  (Stoney)
      h_f/h_s → ∞:  σ_f → 0
    """
    Ef_p = E_f / (1.0 - nu_f)
    Es_p = E_s / (1.0 - nu_s)
    return Ef_p * Es_p * h_s * (alpha_s - alpha_f) * dT / (Ef_p * h_f + Es_p * h_s)


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
