"""
Multi-scale homogenization: Voigt/Reuss/Hill, Hashin-Shtrikman,
Mori-Tanaka, Maxwell-Eucken, Phani-Niyogi, Cahill-Pohl.
"""

import numpy as np


def voigt_average(C_list: list, f_list: list) -> np.ndarray:
    """Voigt (iso-strain, upper): C_V = Σ_r f_r C_r"""
    assert abs(sum(f_list) - 1.0) < 1e-6
    return sum(f * C for f, C in zip(f_list, C_list))


def reuss_average(C_list: list, f_list: list) -> np.ndarray:
    """Reuss (iso-stress, lower): C_R = [Σ_r f_r C_r⁻¹]⁻¹"""
    S_avg = sum(f * np.linalg.inv(C) for f, C in zip(f_list, C_list))
    return np.linalg.inv(S_avg)


def hill_average(C_list, f_list):
    """Hill (VRH): C_H = ½(C_V + C_R)."""
    return 0.5*(voigt_average(C_list, f_list) + reuss_average(C_list, f_list))


def hashin_shtrikman_bulk_modulus(K1: float, K2: float, G1: float, G2: float,
                                   f1: float, f2: float):
    """Hashin-Shtrikman bounds on bulk modulus. K1 ≤ K2, G1 ≤ G2."""
    K_lo = K1 + f2 / (1/(K2-K1+1e-30) + 3*f1/(3*K1+4*G1))
    K_hi = K2 + f1 / (1/(K1-K2+1e-30) + 3*f2/(3*K2+4*G2))
    return K_lo, K_hi


def mori_tanaka_spheres(K_m: float, G_m: float,
                         K_i: float, G_i: float,
                         f_i: float) -> dict:
    """Mori-Tanaka effective moduli for spherical inclusions."""
    f_m = 1 - f_i
    alpha0 = 3*K_m / (3*K_m + 4*G_m)
    beta0  = 6*(K_m + 2*G_m) / (5*(3*K_m + 4*G_m))

    K_eff = K_m + f_i*(K_i-K_m) / (1 + f_m*alpha0*(K_i-K_m)/(K_m+1e-30))
    G_eff = G_m + f_i*(G_i-G_m) / (1 + f_m*beta0*(G_i-G_m)/(G_m+1e-30))

    E_eff = 9*K_eff*G_eff / (3*K_eff + G_eff)
    nu_eff = (3*K_eff - 2*G_eff) / (2*(3*K_eff + G_eff))
    return {"K_eff": K_eff, "G_eff": G_eff, "E_eff": E_eff, "nu_eff": nu_eff}


def maxwell_eucken_kappa(kappa_s: float, kappa_p: float, phi: float) -> float:
    """Maxwell-Eucken: closed-pore composite κ_eff."""
    num = 2*kappa_s + kappa_p - 2*phi*(kappa_s - kappa_p)
    den = 2*kappa_s + kappa_p + phi*(kappa_s - kappa_p)
    return kappa_s * num / (den + 1e-30)


def phani_niyogi_modulus(E0: float, phi: float,
                          phi_c: float = 0.6, n: float = 2.0) -> float:
    """E(φ) = E0 * (1 - φ/φ_c)^n"""
    return E0 * max(1.0 - phi/phi_c, 0.0)**n


def cahill_pohl_kappa_min(kappa_s: float, n: float,
                           v_speeds: np.ndarray,
                           T: float, theta_D: float) -> float:
    """Cahill-Pohl minimum thermal conductivity (PRB 46, 6131, 1992)."""
    from scipy.integrate import quad

    from tebc.constants import k_B
    prefactor = (np.pi/6)**(1/3) * k_B * n**(2/3)
    total = 0.0
    for v_i in v_speeds:
        Theta_i = theta_D
        ratio   = T / Theta_i if Theta_i > 0 else 1.0
        def integrand(x):
            return x**3 * np.exp(x) / (np.expm1(x) + 1e-300)**2
        I, _ = quad(integrand, 0, 1.0/ratio, limit=100)
        total += v_i * ratio**2 * I
    return prefactor * total
