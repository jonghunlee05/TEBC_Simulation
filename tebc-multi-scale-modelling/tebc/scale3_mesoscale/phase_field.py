"""
Phase-field solver for CMAS attack using FiPy.

Allen-Cahn:    ∂φ/∂t = -M_φ δF/δφ
Cahn-Hilliard: ∂c/∂t = ∇·[M_c ∇(δF/δc)]

Free energy: f(φ, c, T) = h(φ) G_β(c,T) + [1-h(φ)] G_L(c,T) + W g(φ)
Interpolation: h(φ) = φ³(6φ²-15φ+10)
Double well:   g(φ) = φ²(1-φ)²
"""

from __future__ import annotations

import numpy as np

try:
    import fipy as fp
except ImportError:
    raise ImportError("pip install fipy")


def interpolation_h(phi: np.ndarray) -> np.ndarray:
    """h(φ) = φ³(6φ²-15φ+10)"""
    return phi**3 * (6*phi**2 - 15*phi + 10)


def double_well_g(phi: np.ndarray) -> np.ndarray:
    """g(φ) = φ²(1-φ)²"""
    return phi**2 * (1 - phi)**2


def interface_params(sigma: float, ell: float) -> dict:
    """W = 6√2 σ / ℓ; κ_φ = (3/4√2) σ ℓ. Karma & Rappel PRE 1996."""
    W    = 6 * np.sqrt(2) * sigma / ell
    kap  = (3.0 / (4.0 * np.sqrt(2))) * sigma * ell
    return {"W": W, "kappa_phi": kap}


def nucleation_rate(T: float, sigma_nu: float, dG_v: float,
                     theta_contact: float = 0.0,
                     I0: float | None = None) -> float:
    """Heterogeneous nucleation rate I = I₀ exp(-ΔG*/k_B T).

    Default I0 from `tebc.constants.NUCLEATION_PREFACTOR`.
    """
    from tebc.constants import NUCLEATION_PREFACTOR, k_B
    if I0 is None:
        I0 = NUCLEATION_PREFACTOR
    f_theta = 0.25 * (2 - 3*np.cos(theta_contact) + np.cos(theta_contact)**3)
    dG_star = (16 * np.pi * sigma_nu**3 / (3 * dG_v**2)) * f_theta
    return I0 * np.exp(-dG_star / (k_B * T))


class CMASPhaseField:
    """
    2D phase-field model for CMAS infiltration and apatite crystallisation.

    Coupled Allen-Cahn / Cahn-Hilliard with KKS chemical potential matching.
    """

    def __init__(self, nx: int, ny: int, dx: float,
                 sigma: float = 0.3,
                 ell: float   = 1e-7,
                 M_phi: float = 1e-8,
                 kappa_c: float = 1e-18,
                 D_CMAS:  float = 1e-12
                 ):
        params = interface_params(sigma, ell)
        self.W     = params["W"]
        self.kap   = params["kappa_phi"]
        self.M_phi = M_phi
        self.kap_c = kappa_c
        self.D_CMAS = D_CMAS
        self.mesh = fp.Grid2D(nx=nx, ny=ny, dx=dx, dy=dx)
        self.phi  = fp.CellVariable(mesh=self.mesh, value=0.0, hasOld=True)
        self.c    = fp.CellVariable(mesh=self.mesh, value=1.0, hasOld=True)

    def df_dphi(self, phi_v, c_v, G_beta, G_liq):
        """∂f/∂φ = W g'(φ) + (G_β - G_L) h'(φ)"""
        h_prime = 30 * phi_v**2 * (1 - phi_v)**2
        g_prime = 2*phi_v*(1-phi_v)**2 - 2*phi_v**2*(1-phi_v)
        return self.W * g_prime + (G_beta - G_liq) * h_prime

    def step(self, dt: float, G_beta: float, G_liq: float) -> None:
        """Explicit Euler step."""
        phi_v = self.phi.value
        c_v   = self.c.value
        ac_eq = (fp.TransientTerm(var=self.phi)
                 == self.M_phi * (fp.DiffusionTerm(coeff=self.kap, var=self.phi)
                                  - self.W * fp.ImplicitSourceTerm(
                                      coeff=6*phi_v*(1-2*phi_v),var=self.phi)))
        M_c   = self.D_CMAS * c_v * (1-c_v)
        ch_eq = (fp.TransientTerm(var=self.c)
                 == fp.DiffusionTerm(coeff=M_c, var=self.c))
        fp.solve([ac_eq, ch_eq], dt=dt)
        self.phi.updateOld()
        self.c.updateOld()
