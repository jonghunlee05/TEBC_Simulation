"""
Continuum damage mechanics for thermal cycling.

Lemaitre isotropic damage (JMPS 1979):
  σ̃ = σ / (1 - D)
  Y  = σ_eq² R_v / [2E(1-D)²]
  R_v = ⅔(1+ν) + 3(1-2ν)(σ_H/σ_eq)²
  dD/dp = (Y/S)^s

Mazars tensile damage (IJNME 1984)
Tvergaard-Hutchinson CZM (JMPS 1992)
"""

import numpy as np


def triaxiality_factor(sigma_eq: float, sigma_H: float, nu: float) -> float:
    """R_v = ⅔(1+ν) + 3(1-2ν)(σ_H / σ_eq)²

    σ_eq is by definition the von Mises norm (≥ 0). We treat σ_eq ≤ 0
    as the unloaded / undamaged state and return 0 — both the σ_eq = 0
    case (previously returned ∞ via the +1e-30 guard) and any caller
    that mistakenly hands in a signed value (returns 0 rather than
    silently squaring the wrong sign).
    """
    if sigma_eq <= 0.0:
        return 0.0
    return (2.0/3.0)*(1+nu) + 3*(1-2*nu)*(sigma_H/sigma_eq)**2


def lemaitre_damage_rate(sigma_eq: float, sigma_H: float, nu: float,
                          E: float, D: float,
                          S: float = 1.0e6, s: float = 1.0) -> float:
    """Lemaitre damage evolution dD/dp = (Y/S)^s."""
    Rv = triaxiality_factor(sigma_eq, sigma_H, nu)
    Y  = sigma_eq**2 * Rv / (2.0 * E * (1.0 - D)**2 + 1e-30)
    return (Y / S)**s


def mazars_equivalent_strain(eps_principal: np.ndarray) -> float:
    """ε̃ = √[Σᵢ ⟨εᵢ⟩₊²]"""
    positive = np.maximum(eps_principal, 0.0)
    return np.sqrt(np.sum(positive**2))


def mazars_damage(eps_tilde: float, eps0: float | None = None,
                   A: float | None = None, B: float | None = None,
                   D_max: float = 0.999) -> float:
    """Mazars damage function for quasi-brittle materials.

    Defaults for ε₀, A, B pulled from `tebc.constants`
    (MAZARS_EPS0_DEFAULT, MAZARS_A_TENSION, MAZARS_B_TENSION).

    `D_max` (default 0.999) is the numerical ceiling — leaving a residual
    elastic stiffness E·(1 − D_max) that prevents the constitutive
    tangent from going singular. For a clean transition to fracture,
    drive D_max → 1 and switch to element deletion / cohesive zones at
    that threshold.
    """
    from tebc.constants import (
        MAZARS_A_TENSION,
        MAZARS_B_TENSION,
        MAZARS_EPS0_DEFAULT,
    )
    if eps0 is None: eps0 = MAZARS_EPS0_DEFAULT
    if A    is None: A    = MAZARS_A_TENSION
    if B    is None: B    = MAZARS_B_TENSION
    if eps_tilde <= eps0:
        return 0.0
    D = 1.0 - (1.0-A)*eps0/eps_tilde - A*np.exp(-B*(eps_tilde - eps0))
    return np.clip(D, 0.0, D_max)


class TVHCohesiveZone:
    """
    Tvergaard-Hutchinson cohesive zone model (JMPS 40, 1377, 1992).

    λ = √[(δ_n/δ_n^c)² + (δ_t/δ_t^c)²]
    σ(λ) piecewise: linear → plateau → softening → 0
    G_c = ½ σ̂ δ_n^c [1 - λ₁ + λ₂]
    """
    def __init__(self, sigma_hat: float = 100e6,
                 delta_n_c: float = 1e-6,
                 delta_t_c: float = 3e-6,
                 lambda1: float = 0.15,
                 lambda2: float = 0.50):
        self.sigma_hat = sigma_hat
        self.delta_n_c = delta_n_c
        self.delta_t_c = delta_t_c
        self.l1 = lambda1
        self.l2 = lambda2
        self.G_c = 0.5 * sigma_hat * delta_n_c * (1 - lambda1 + lambda2)

    def effective_opening(self, delta_n: float, delta_t: float) -> float:
        return np.sqrt((delta_n/self.delta_n_c)**2 + (delta_t/self.delta_t_c)**2)

    def sigma_lambda(self, lam: float) -> float:
        if lam <= 0:         return 0.0
        if lam <= self.l1:   return self.sigma_hat * lam / self.l1
        if lam <= self.l2:   return self.sigma_hat
        if lam <= 1.0:       return self.sigma_hat*(1-lam)/(1-self.l2)
        return 0.0

    def tractions(self, delta_n: float, delta_t: float):
        lam = self.effective_opening(delta_n, delta_t)
        if lam < 1e-12:
            return 0.0, 0.0
        sl = self.sigma_lambda(lam)
        T_n = (sl/lam) * (delta_n / self.delta_n_c)
        T_t = (sl/lam) * (delta_t / self.delta_t_c)
        return T_n, T_t

    def benzeggagh_kenane_toughness(self, G_II: float, G_T: float,
                                     eta: float = 1.5,
                                     G_IIc: float | None = None) -> float:
        """G_c(ψ) = G_Ic + (G_IIc - G_Ic)(G_II/G_T)^η

        `G_IIc` defaults to `1.5 · G_Ic` (the previous hardcoded value)
        only when not supplied; pass an experimental ratio for real
        coatings.
        """
        G_Ic  = self.G_c
        if G_IIc is None:
            G_IIc = 1.5 * G_Ic
        if G_T <= 0.0:
            return G_Ic
        return G_Ic + (G_IIc - G_Ic) * (G_II / G_T)**eta
