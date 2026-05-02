"""
Shared mathematical utilities used across all scales.
"""

import numpy as np
from scipy.optimize import curve_fit

from tebc.constants import R_gas, k_B

# ── Voigt notation helpers ────────────────────────────────────────────────────
VOIGT_MAP = {(0,0):0,(1,1):1,(2,2):2,(1,2):3,(0,2):4,(0,1):5}

def tensor_to_voigt(C_full: np.ndarray) -> np.ndarray:
    """Convert 3x3x3x3 stiffness tensor → 6x6 Voigt matrix."""
    C6 = np.zeros((6, 6))
    for (i,j), m in VOIGT_MAP.items():
        for (k,l), n in VOIGT_MAP.items():
            C6[m, n] = C_full[i, j, k, l]
    return C6


def voigt_to_engineering(C6: np.ndarray) -> dict:
    """
    Convert 6×6 Voigt stiffness → engineering moduli.
    Inverts S = C⁻¹ and reads off E, nu, G.
    Valid for orthorhombic/monoclinic symmetry.
    """
    S = np.linalg.inv(C6)
    return {
        "E": np.array([1/S[i,i] for i in range(3)]),
        "nu_12": -S[0,1] / S[0,0],
        "nu_13": -S[0,2] / S[0,0],
        "nu_23": -S[1,2] / S[1,1],
        "G":  np.array([1/S[3+i,3+i] for i in range(3)]),
    }


# ── Arrhenius fitting ─────────────────────────────────────────────────────────
def arrhenius_fit(T_K: np.ndarray, rate: np.ndarray) -> tuple[float, float]:
    """Fit rate = A * exp(-Ea/RT). Returns (A, Ea_J_per_mol)."""
    def model(T, lnA, Ea):
        return lnA - Ea / (R_gas * T)
    popt, _ = curve_fit(model, T_K, np.log(rate), p0=[1.0, 1e5])
    return np.exp(popt[0]), popt[1]


def arrhenius_eval(T_K, A: float, Ea: float) -> np.ndarray:
    """Evaluate Arrhenius: A * exp(-Ea / (R * T))."""
    return A * np.exp(-Ea / (R_gas * np.asarray(T_K)))


# ── Bose-Einstein occupation ──────────────────────────────────────────────────
def bose_einstein(omega: np.ndarray, T: float) -> np.ndarray:
    """n_BE(ω, T) = 1 / (exp(ℏω / k_B T) - 1). Returns 0 at omega=0."""
    from tebc.constants import hbar
    with np.errstate(over='ignore', invalid='ignore'):
        x = hbar * omega / (k_B * T)
        n = np.where(x > 1e-10, 1.0 / (np.expm1(x)), 0.0)
    return n


# ── Mode heat capacity ────────────────────────────────────────────────────────
def mode_heat_capacity(omega: np.ndarray, T: float) -> np.ndarray:
    """
    C_{qs} = k_B * (ℏω/k_BT)² * exp(ℏω/k_BT) / (exp(ℏω/k_BT) - 1)²
    Units: J/K per mode.
    """
    from tebc.constants import hbar
    x = hbar * omega / (k_B * T)
    ex = np.exp(x)
    return k_B * x**2 * ex / (ex - 1.0)**2


# ── Birch-Murnaghan EOS ───────────────────────────────────────────────────────
def birch_murnaghan_energy(V: np.ndarray, E0: float, V0: float,
                            B0: float, B0p: float) -> np.ndarray:
    """3rd-order Birch-Murnaghan EOS."""
    eta = (V0 / V) ** (2.0/3.0)
    f   = 0.5 * (eta - 1.0)
    return E0 + 9.0*V0*B0/16.0 * (f**3 * B0p + f**2 * (6.0 - 4.0*eta))


def fit_eos(V: np.ndarray, E: np.ndarray) -> dict:
    """Fit Birch-Murnaghan to (V, E) data. Returns {E0, V0, B0, B0p}."""
    from scipy.optimize import minimize
    idx   = np.argmin(E)
    p0    = [E[idx], V[idx], 150e9, 4.0]
    bounds = [(None,None),(0,None),(0,None),(1,10)]
    res = minimize(
        lambda p: np.sum((birch_murnaghan_energy(V, *p) - E)**2),
        p0, method='L-BFGS-B', bounds=bounds
    )
    E0, V0, B0, B0p = res.x
    return {"E0": E0, "V0": V0, "B0": B0, "B0p": B0p}
