"""
Physical constants and material database.
All values in SI unless noted.
"""

import numpy as np

# ── Physical constants ────────────────────────────────────────────────────────
k_B   = 1.380649e-23   # J/K  Boltzmann
hbar  = 1.054571817e-34 # J·s  reduced Planck
eV    = 1.602176634e-19 # J    electron-volt
N_A   = 6.02214076e23  # mol⁻¹ Avogadro
R_gas = 8.314462       # J/(mol·K)
sigma_SB = 5.670374419e-8  # W/(m²·K⁴) Stefan-Boltzmann
atm_Pa   = 101325.0    # Pa     standard atmosphere

# ── Mazars damage model defaults (IJNME 1984) ────────────────────────────────
MAZARS_EPS0_DEFAULT = 1.0e-4   # damage threshold strain
MAZARS_A_TENSION    = 0.96     # A coefficient, tension
MAZARS_B_TENSION    = 15000.0  # B coefficient, tension

# ── Phase-field nucleation defaults ──────────────────────────────────────────
NUCLEATION_PREFACTOR = 1.0e36  # I0 [s⁻¹·m⁻³] heterogeneous CNT prefactor

# ── Robinson–Smialek SiO2 recession calibration anchors ──────────────────────
# (Opila & Hann 1997; Robinson & Smialek 1999)
RS_T_REF_K       = 1589.0          # K
RS_P_H2O_REF_PA  = 0.1 * atm_Pa    # 0.1 atm partial pressure
RS_V_GAS_REF     = 0.044           # m/s reference gas velocity
RS_K_L_REF       = 2.0e-9          # m/s recession at the reference state

# ── Material database (RT unless stated) ─────────────────────────────────────
MATERIALS = {
    "beta_Yb2Si2O7": {
        "rho":   6180.0,
        "E":     185e9,
        "nu":    0.275,
        "alpha": 4.05e-6,
        "alpha_aniso": np.array([3.57e-6, 2.49e-6, 1.48e-6]),
        "kappa": 2.5,
        "cp":    450.0,
        "KIC":   1.75e6,
        "Gamma_interface": 30.0,
        "T_melt": 2123.0,
        "k_p_TGO": 1e-14 / 3600,
        "Ea_kp":  101e3,
        "k_l":   2.78e-11,
        "Ea_kl":  108e3,
    },
    "beta_Y2Si2O7": {
        "rho":   4040.0,
        "E":     165e9,
        "nu":    0.27,
        "alpha": 4.0e-6,
        "alpha_aniso": np.array([3.5e-6, 2.4e-6, 2.1e-6]),
        "kappa": 3.0,
        "cp":    460.0,
        "KIC":   2.0e6,
        "Gamma_interface": 35.0,
        "T_melt": 2048.0,
        "k_p_TGO": 8e-15 / 3600,
        "Ea_kp":  108e3,
        "k_l":   1.5e-11,
        "Ea_kl":  108e3,
    },
    "7YSZ": {
        "rho":   6050.0,
        "E":     210e9,
        "E_APS": 50e9,
        "nu":    0.23,
        "alpha": 10.5e-6,
        "kappa": 2.2,
        "kappa_APS": 1.0,
        "cp":    505.0,
        "KIC":   2.0e6,
        "Gamma_interface": 25.0,
        "T_melt": 2983.0,
        "D0_O":  1.3e-6,
        "Ea_DO": 0.95 * eV,
    },
    "Si_bondcoat": {
        "rho":   2329.0,
        "E":     162e9,
        "nu":    0.225,
        "alpha": 2.6e-6,
        "kappa": 156.0,
        "cp":    712.0,
        "T_melt": 1687.0,
        "k_p_dry": 4e-14 / 3600,
        "Ea_kp_dry": 119e3,
        "k_p_wet": 4e-13 / 3600,
        "Ea_kp_wet": 68e3,
    },
    "SiC_SiC_CMC": {
        "rho":   2800.0,
        "E":     230e9,
        "nu":    0.15,
        "alpha": 4.8e-6,
        "kappa": 15.0,
        "cp":    750.0,
        "UTS":   400e6,
    },
    "SiO2_TGO": {
        "rho":   2200.0,
        "E":     70e9,
        "nu":    0.17,
        "alpha": 0.55e-6,
        "kappa": 1.4,
        "cp":    740.0,
        "PBR":   2.15,
    },
}
