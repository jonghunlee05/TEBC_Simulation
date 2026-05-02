"""
Thermally Grown Oxide (SiO₂) kinetics on Si bond coat.

Deal-Grove:   x² + Ax = B(t + τ)
Paralinear:   dx/dt = k_p/(2x) - k_l
PBR stress:   ε_TGO^ox = ⅓(PBR - 1) δ_{ij}
"""

from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from tebc.constants import R_gas
from tebc.utils import arrhenius_eval


def deal_grove_thickness(t: np.ndarray, k_p: float, k_l: float,
                          x0: float = 0.0) -> np.ndarray:
    """
    Thickness x(t) from Deal-Grove: x² + Ax = B(t + τ).

    Convention used here:
      B   = 2·k_p          (so pure-parabolic limit gives x² = 2 k_p t)
      A   = 2·k_p / k_l    (so B/A = k_l, the linear-rate constant)
      τ   = x0·(x0 + A)/B  (virtual time offset for x(0) = x0)

    Closed-form solution:
      x(t) = (A/2)·[ √(1 + 4·B·(t+τ)/A²) − 1 ]

    Limits:
      thin scale  (B(t+τ) ≪ A²/4):  x → k_l (t+τ)
      thick scale (B(t+τ) ≫ A²/4):  x → √(2 k_p (t+τ))
    """
    A = 2.0 * k_p / k_l if k_l > 0 else 1e30
    B = 2.0 * k_p
    tau = x0 * (x0 + A) / B if B > 0 else 0.0
    discriminant = 1.0 + 4.0 * B * (t + tau) / (A**2 + 1e-30)
    x = (A / 2.0) * (np.sqrt(np.maximum(discriminant, 0.0)) - 1.0)
    return x


def parabolic_rate_constant(T_K: float, k_p0: float, Ea_J: float) -> float:
    """k_p(T) = k_p0 * exp(-Ea / RT)  [m²/s]"""
    return arrhenius_eval(T_K, k_p0, Ea_J)


def paralinear_ode(t: float, x: np.ndarray,
                    k_p: float, k_l: float):
    """
    dx/dt = k_p / (2x) - k_l. Steady state x_ss = k_p / (2 k_l).

    Once x ≤ 0 (volatilization outpaces oxidation faster than ODE can
    relax), the protective scale has been consumed; physically dx/dt = 0
    and recession proceeds at k_l on the bare substrate.
    """
    if x[0] <= 0.0:
        return [0.0]
    return [k_p / (2.0 * x[0]) - k_l]


def solve_paralinear(t_span: tuple, k_p: float, k_l: float,
                      x0: float = None,
                      n_points: int = 500) -> dict:
    """
    Solve paralinear ODE for TGO thickness x(t) and substrate recession.

    Numerically robust against the volatilization-dominated regime
    (k_l ≫ √(k_p/Δt)): integrate in y = x² instead of x, where
        dy/dt = k_p - 2 k_l √y,
    which keeps y ≥ 0 by clipping under the square root and asymptotes
    smoothly to y_ss = (k_p/(2 k_l))².

    Default x0 is set adaptively to ½·x_ss so the trajectory starts inside
    the basin of attraction of the steady state.
    """
    t_eval = np.linspace(*t_span, n_points)
    x_ss   = k_p / (2.0 * k_l) if k_l > 0 else np.inf

    if x0 is None:
        x0 = max(min(0.5 * x_ss, 1e-9), 1e-15)

    def rhs_y(t, y, k_p, k_l):
        y_safe = max(y[0], 0.0)
        return [k_p - 2.0 * k_l * np.sqrt(y_safe)]

    y0 = x0 ** 2
    sol = solve_ivp(rhs_y, t_span, [y0], args=(k_p, k_l),
                    t_eval=t_eval, method="LSODA",
                    rtol=1e-8, atol=1e-24)
    x_TGO     = np.sqrt(np.maximum(sol.y[0], 0.0))
    recession = k_l * sol.t
    return {"t": sol.t, "x_TGO": x_TGO, "recession": recession, "x_ss": x_ss}


def tgo_growth_stress(x_TGO: float, E_TGO: float, nu_TGO: float,
                       alpha_TGO: float, alpha_sub: float,
                       dT: float, PBR: float = 2.15) -> float:
    """
    Total TGO biaxial stress = thermal + growth (PBR).
    """
    eps_growth = (PBR - 1.0) / 3.0
    biaxial_mod = E_TGO / (1 - nu_TGO)
    sigma_thermal = biaxial_mod * (alpha_TGO - alpha_sub) * dT
    sigma_growth  = -biaxial_mod * eps_growth
    return sigma_thermal + sigma_growth


def robinson_smialek_recession(T_K: float, P_H2O: float,
                                P_tot: float, v_gas: float,
                                Ea_J: float = 108e3,
                                k0: float   = None) -> float:
    """k_l ∝ v^{0.5} * P_H2O^2 * P_tot^{-0.5} * exp(-ΔQ/RT)."""
    if k0 is None:
        k0 = 2e-9 / (0.044**0.5 * (0.1*101325)**2 * (101325)**(-0.5)
                      * np.exp(-Ea_J/(R_gas*1589)))
    return k0 * v_gas**0.5 * P_H2O**2 * P_tot**(-0.5) * np.exp(-Ea_J/(R_gas*T_K))
