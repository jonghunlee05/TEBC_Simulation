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


def fenics_thermoelastic_setup():
    """Return FEniCSx skeleton script as string."""
    code = '''
import dolfinx
from dolfinx import fem, mesh, io
from dolfinx.fem.petsc import LinearProblem
import ufl
import numpy as np
from mpi4py import MPI

domain = mesh.create_rectangle(MPI.COMM_WORLD, [[0,0],[1e-2,6e-3]],
                                [200,120], mesh.CellType.triangle)

V_T = fem.functionspace(domain, ("Lagrange", 1))
V_u = fem.functionspace(domain, ("Lagrange", 1, (2,)))

T, theta  = ufl.TrialFunction(V_T), ufl.TestFunction(V_T)
u, v_test = ufl.TrialFunction(V_u), ufl.TestFunction(V_u)

V0     = fem.functionspace(domain, ("DG", 0))
kappa  = fem.Function(V0)
rho_cp = fem.Function(V0)
E_mod  = fem.Function(V0)
nu_mod = fem.Function(V0)
alpha  = fem.Function(V0)

def eps(u):
    return ufl.sym(ufl.grad(u))

def sigma(u, T_field, T_ref=300.0):
    mu    = E_mod / (2*(1+nu_mod))
    lam   = E_mod*nu_mod / ((1+nu_mod)*(1-2*nu_mod))
    strain = eps(u)
    dT     = T_field - T_ref
    return (2*mu*strain + lam*ufl.tr(strain)*ufl.Identity(2)
            - (3*lam + 2*mu)*alpha*dT*ufl.Identity(2))

dt_val = fem.Constant(domain, 1.0)
T_old  = fem.Function(V_T)
T_old.x.array[:] = 300.0

a_T = (rho_cp/dt_val * T * theta * ufl.dx
       + kappa * ufl.dot(ufl.grad(T), ufl.grad(theta)) * ufl.dx)
L_T = rho_cp/dt_val * T_old * theta * ufl.dx

T_field = fem.Function(V_T)
a_u = ufl.inner(sigma(u, T_field), eps(v_test)) * ufl.dx
L_u = ufl.dot(fem.Constant(domain, np.zeros(2)), v_test) * ufl.dx

problem_T = LinearProblem(a_T, L_T, bcs=[], petsc_options={"ksp_type": "cg"})
problem_u = LinearProblem(a_u, L_u, bcs=[], petsc_options={"ksp_type": "gmres"})
'''
    return code
