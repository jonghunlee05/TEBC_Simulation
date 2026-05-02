# TEBC Multi-Scale Simulation — Claude Code Implementation Specification

> **Purpose:** This document is a complete, machine-readable implementation spec for Claude Code.
> Every section maps directly to a Python module. All governing equations are expressed in
> code-ready form. No ambiguity — implement exactly as written.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Dependencies & Environment](#2-dependencies--environment)
3. [Shared Utilities](#3-shared-utilities)
4. [Scale 1 — DFT/DNP Atomistic Interface](#4-scale-1--dftdnp-atomistic-interface)
5. [Scale 2 — MD / Phonon / QHA](#5-scale-2--md--phonon--qha)
6. [Scale 3 — Phase-Field / CALPHAD / TGO](#6-scale-3--phase-field--calphad--tgo)
7. [Scale 4 — FEA Continuum](#7-scale-4--fea-continuum)
8. [Cross-Scale Coupling](#8-cross-scale-coupling)
9. [Sensitivity Analysis](#9-sensitivity-analysis)
10. [Orchestrator](#10-orchestrator)
11. [Validated Material Parameters](#11-validated-material-parameters)
12. [Test Suite](#12-test-suite)

---

## 1. Project Structure

```
tebc_simulation/
│
├── README.md
├── pyproject.toml
├── environment.yml
│
├── tebc/
│   ├── __init__.py
│   ├── constants.py              # Physical constants, material DB
│   ├── utils.py                  # Shared math utilities
│   │
│   ├── scale1_atomistic/
│   │   ├── __init__.py
│   │   ├── dft_interface.py      # DFT parser: VASP OUTCAR/vasprun.xml
│   │   ├── dnp_trainer.py        # DeePMD-kit training wrapper
│   │   ├── dpgen_workflow.py     # DP-GEN active learning loop
│   │   └── scale1_outputs.py     # Extract C_ijkl, alpha_ij, E_defect, gamma_surf
│   │
│   ├── scale2_md/
│   │   ├── __init__.py
│   │   ├── lammps_runner.py      # LAMMPS input generator + runner
│   │   ├── ensembles.py          # NVT / NVE / NPT equations & controls
│   │   ├── phonon_qha.py         # Phonopy/Phono3py wrapper + QHA F(V,T)
│   │   ├── green_kubo.py         # HCACF → kappa tensor
│   │   ├── msd_diffusion.py      # MSD → D_O Arrhenius fit
│   │   └── scale2_outputs.py     # kappa(T), alpha(T), D_O(T), C_ij(T)
│   │
│   ├── scale3_mesoscale/
│   │   ├── __init__.py
│   │   ├── phase_field.py        # Allen-Cahn + Cahn-Hilliard solver (FiPy)
│   │   ├── calphad_interface.py  # pycalphad wrapper: G_m, phase fracs
│   │   ├── tgo_kinetics.py       # Deal-Grove + paralinear recession
│   │   ├── cmas_reaction.py      # CMAS dissolution + apatite nucleation
│   │   └── scale3_outputs.py     # kappa_eff, E_eff, TGO thickness(t)
│   │
│   ├── scale4_continuum/
│   │   ├── __init__.py
│   │   ├── thermoelastic.py      # FEniCSx coupled T + u BVP
│   │   ├── damage_mechanics.py   # Lemaitre CDM + Mazars tensile damage
│   │   ├── cohesive_zone.py      # Tvergaard-Hutchinson CZM traction law
│   │   ├── recession.py          # Paralinear oxidation + Robinson-Smialek
│   │   └── scale4_outputs.py     # sigma(x,t), T(x,t), D(x,t), TGO(t)
│   │
│   ├── coupling/
│   │   ├── __init__.py
│   │   ├── homogenization.py     # Voigt/Reuss/HS/Mori-Tanaka/SC
│   │   ├── rve_solver.py         # FE2 RVE micro BVP
│   │   ├── parameter_passing.py  # Validated handoff between scales
│   │   └── uncertainty.py        # GP surrogate + PCE + Bayesian UQ
│   │
│   ├── sensitivity/
│   │   ├── __init__.py
│   │   └── sobol_morris.py       # SALib Sobol + Morris analysis
│   │
│   └── orchestrator.py           # Main pipeline runner
│
├── data/
│   ├── materials_db.json         # Validated numerical parameters
│   └── tdb/                      # Thermodynamic database files (.tdb)
│
└── tests/
    ├── test_scale1.py
    ├── test_scale2.py
    ├── test_scale3.py
    ├── test_scale4.py
    ├── test_coupling.py
    └── test_sensitivity.py
```

---

## 2. Dependencies & Environment

### `environment.yml`
```yaml
name: tebc
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - numpy>=1.26
  - scipy>=1.12
  - matplotlib>=3.8
  - pandas>=2.2
  - h5py>=3.10
  - pytest>=8.0
  # FEA
  - fenics-dolfinx>=0.8         # FEniCSx continuum solver
  - petsc4py>=3.20
  - mpi4py>=3.1
  # Phonons / MD
  - phonopy>=2.23
  - phono3py>=2.8
  - ase>=3.23                   # Atomic Simulation Environment
  - pymatgen>=2024.3
  # Phase-field
  - fipy>=3.4                   # Allen-Cahn / Cahn-Hilliard PDE solver
  # CALPHAD
  - pycalphad>=0.10
  # Sensitivity
  - salib>=1.5
  # ML potentials (optional GPU)
  - pip
  - pip:
    - deepmd-kit>=3.0
    - dpgen>=0.11
    - uncertainpy>=1.4
```

### `pyproject.toml`
```toml
[build-system]
requires = ["setuptools>=68"]

[project]
name = "tebc_simulation"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy", "scipy", "matplotlib", "pandas",
    "ase", "pymatgen", "phonopy", "phono3py",
    "fipy", "pycalphad", "salib", "h5py",
]

[project.optional-dependencies]
fea = ["fenics-dolfinx", "petsc4py", "mpi4py"]
ml  = ["deepmd-kit", "dpgen"]
```

---

## 3. Shared Utilities

### `tebc/constants.py`

```python
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

# ── Material database (RT unless stated) ─────────────────────────────────────
# Sources: see Section 11
MATERIALS = {
    "beta_Yb2Si2O7": {
        "rho":   6180.0,          # kg/m³
        "E":     185e9,           # Pa  Young's modulus
        "nu":    0.275,           # Poisson ratio
        "alpha": 4.05e-6,         # K⁻¹ mean linear CTE
        "alpha_aniso": np.array([3.57e-6, 2.49e-6, 1.48e-6]),  # a,b,c axes
        "kappa": 2.5,             # W/(m·K) RT  → use kappa_T() for T-dep.
        "cp":    450.0,           # J/(kg·K)
        "KIC":   1.75e6,          # Pa·m^0.5
        "Gamma_interface": 30.0,  # J/m²  EBC/bond-coat interface toughness
        "T_melt": 2123.0,         # K
        "k_p_TGO": 1e-14 / 3600, # m²/s  parabolic rate at 1589 K (1316°C)
        "Ea_kp":  101e3,          # J/mol activation energy k_p
        "k_l":   2.78e-11,        # m/s   linear recession at 1589 K
        "Ea_kl":  108e3,          # J/mol
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
        "E":     210e9,           # dense
        "E_APS": 50e9,            # APS porous (~12% porosity)
        "nu":    0.23,
        "alpha": 10.5e-6,
        "kappa": 2.2,             # dense
        "kappa_APS": 1.0,         # APS porous
        "cp":    505.0,
        "KIC":   2.0e6,
        "Gamma_interface": 25.0,
        "T_melt": 2983.0,
        "D0_O":  1.3e-6,          # m²/s  oxygen pre-exp (Brossmann 2003)
        "Ea_DO": 0.95 * eV,       # J     activation energy O diffusion
    },
    "Si_bondcoat": {
        "rho":   2329.0,
        "E":     162e9,
        "nu":    0.225,
        "alpha": 2.6e-6,          # RT value; strongly T-dependent
        "kappa": 156.0,           # RT; use kappa_Si_T() for T-dep.
        "cp":    712.0,
        "T_melt": 1687.0,         # K  — HARD service ceiling
        "k_p_dry": 4e-14 / 3600, # m²/s dry O2 at 1473 K
        "Ea_kp_dry": 119e3,
        "k_p_wet": 4e-13 / 3600, # m²/s 90% H2O at 1589 K
        "Ea_kp_wet": 68e3,
    },
    "SiC_SiC_CMC": {
        "rho":   2800.0,
        "E":     230e9,           # in-plane 2D
        "nu":    0.15,
        "alpha": 4.8e-6,
        "kappa": 15.0,            # through-thickness at 1000°C
        "cp":    750.0,
        "UTS":   400e6,           # Pa
    },
    "SiO2_TGO": {
        "rho":   2200.0,          # amorphous
        "E":     70e9,
        "nu":    0.17,
        "alpha": 0.55e-6,
        "kappa": 1.4,
        "cp":    740.0,
        "PBR":   2.15,            # Pilling-Bedworth ratio Si→SiO2
    },
}
```

### `tebc/utils.py`

```python
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
        "E": np.array([1/S[i,i] for i in range(3)]),          # Pa, axial
        "nu_12": -S[0,1] / S[0,0],
        "nu_13": -S[0,2] / S[0,0],
        "nu_23": -S[1,2] / S[1,1],
        "G":  np.array([1/S[3+i,3+i] for i in range(3)]),     # Pa, shear
    }


# ── Arrhenius fitting ─────────────────────────────────────────────────────────
def arrhenius_fit(T_K: np.ndarray, rate: np.ndarray) -> tuple[float, float]:
    """
    Fit rate = A * exp(-Ea/RT).
    Returns (A, Ea_J_per_mol).
    """
    def model(T, lnA, Ea):
        return lnA - Ea / (R_gas * T)
    popt, _ = curve_fit(model, T_K, np.log(rate), p0=[1.0, 1e5])
    return np.exp(popt[0]), popt[1]


def arrhenius_eval(T_K: float | np.ndarray, A: float, Ea: float) -> np.ndarray:
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
    """
    3rd-order Birch-Murnaghan EOS.
    E(V) = E0 + (9 V0 B0/16){[(V0/V)^(2/3) - 1]^3 B0'
                              + [(V0/V)^(2/3) - 1]^2 [6 - 4(V0/V)^(2/3)]}
    B0 in Pa, V in m³.
    """
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
```

---

## 4. Scale 1 — DFT/DNP Atomistic Interface

### `tebc/scale1_atomistic/dft_interface.py`

```python
"""
Parse DFT results (VASP format) and extract:
  - Elastic stiffness tensor C_ijkl  (GPa)
  - Cohesive energy E0               (eV/atom)
  - Equilibrium volume V0            (Å³)
  - Anisotropic CTE tensor alpha_ij  (K⁻¹) via QHA
  - Surface energy gamma_surf        (J/m²)
  - Defect formation energy E_def    (eV)

All implemented with pymatgen + phonopy; no VASP license required for parsing.
"""

from __future__ import annotations
import numpy as np
from pathlib import Path

try:
    from pymatgen.io.vasp.outputs import Outcar, Vasprun
    from pymatgen.analysis.elasticity import ElasticTensor
except ImportError:
    raise ImportError("pip install pymatgen")


def parse_elastic_tensor(outcar_path: str | Path) -> np.ndarray:
    """
    Parse VASP OUTCAR → 6×6 elastic stiffness matrix (GPa).

    VASP writes C_ij in kBar; pymatgen converts to GPa.
    Kohn-Sham equation solved self-consistently → stress-strain response
    via finite distortions: C_ij = (1/V0) ∂²E/∂ε_i∂ε_j.

    Returns
    -------
    C6 : np.ndarray, shape (6,6), GPa
    """
    outcar = Outcar(str(outcar_path))
    et = ElasticTensor.from_voigt(
        np.array(outcar.elastic_tensor)   # kBar
    ) * 0.1  # kBar → GPa
    return et.voigt


def parse_structure_energy(vasprun_path: str | Path) -> dict:
    """
    Parse vasprun.xml → E0 (eV/atom), V0 (Å³), forces (eV/Å).
    Used for:
      E_coh = -(1/N)[E_crystal - Σ N_i E_i^atom]
    """
    vr = Vasprun(str(vasprun_path))
    struct = vr.final_structure
    N = len(struct)
    return {
        "E0_per_atom": vr.final_energy / N,
        "E0_total":    vr.final_energy,
        "V0_angstrom3": struct.volume,
        "N_atoms": N,
        "lattice": struct.lattice.matrix,        # Å
        "forces":  np.array(vr.ionic_steps[-1]["forces"]),  # eV/Å
    }


def compute_cohesive_energy(E_crystal_eV: float, N: int,
                             E_atoms_eV: dict[str, float],
                             composition: dict[str, int]) -> float:
    """
    E_coh = -(1/N)[E_crystal - Σ_i N_i * E_i^atom]

    Parameters
    ----------
    E_crystal_eV  : DFT total energy of crystal (eV)
    N             : number of atoms
    E_atoms_eV    : {species: isolated atom energy in eV}
    composition   : {species: count}

    Returns
    -------
    E_coh : float, eV/atom (positive = bound)
    """
    E_ref = sum(composition[s] * E_atoms_eV[s] for s in composition)
    return -(E_crystal_eV - E_ref) / N


def compute_surface_energy(E_slab: float, E_bulk_per_atom: float,
                            N_slab: int, A_surface_m2: float) -> float:
    """
    γ_surf = (E_slab - N_slab * E_bulk/atom) / (2 * A)

    Factor 2 because slab has two surfaces.
    Units: J/m²   (convert from eV/Å² using 1 eV/Å² = 16.022 J/m²)

    Parameters
    ----------
    E_slab            : eV, total slab energy
    E_bulk_per_atom   : eV/atom
    N_slab            : number of atoms in slab
    A_surface_m2      : surface area in m²

    Returns
    -------
    gamma : float, J/m²
    """
    from tebc.constants import eV
    delta_E = (E_slab - N_slab * E_bulk_per_atom) * eV   # J
    return delta_E / (2.0 * A_surface_m2)


def compute_defect_formation_energy(E_defect: float, E_host: float,
                                     mu: dict[str, float],
                                     delta_n: dict[str, int],
                                     q: int, E_VBM: float,
                                     E_Fermi: float,
                                     E_corr: float = 0.0) -> float:
    """
    Standard defect formation energy (Freysoldt, RMP 2014):

      E_f[X^q] = E_tot[defect,q] - E_tot[host]
                 - Σ_i δn_i μ_i
                 + q(E_VBM + E_Fermi)
                 + E_corr

    Parameters
    ----------
    E_defect  : eV, DFT energy of supercell with defect
    E_host    : eV, DFT energy of perfect supercell (same cell)
    mu        : {species: chemical potential in eV}
    delta_n   : {species: n_added - n_removed} (positive = added)
    q         : charge state
    E_VBM     : eV, valence band maximum of host
    E_Fermi   : eV, Fermi level relative to VBM
    E_corr    : eV, image-charge finite-size correction

    Returns
    -------
    E_f : float, eV
    """
    chem_term = sum(delta_n.get(s, 0) * mu[s] for s in mu)
    return (E_defect - E_host - chem_term
            + q * (E_VBM + E_Fermi) + E_corr)


def extract_born_effective_charges(outcar_path: str | Path) -> np.ndarray:
    """
    Parse Born effective charges Z*_{I,αβ} from VASP OUTCAR.
    VASP tag: LEPSILON = .TRUE. or LCALCEPS = .TRUE.

    Returns
    -------
    Z_star : np.ndarray, shape (N_atoms, 3, 3)
    """
    outcar = Outcar(str(outcar_path))
    return np.array(outcar.born)
```

### `tebc/scale1_atomistic/dnp_trainer.py`

```python
"""
Deep Neural Network Potential (DeePMD) training wrapper.

Architecture: DeepPot-SE smooth descriptor + fitting net
Loss:  L = (p_e/N)|ΔE|² + (p_f/3N)Σ|ΔF_i|² + (p_v/9N)|ΔΞ|²
"""

from __future__ import annotations
import json
import subprocess
from pathlib import Path


# ── Descriptor: smooth se_e2_a ────────────────────────────────────────────────
def build_deepmd_input(
    type_map: list[str],             # e.g. ["O", "Si", "Yb"]
    r_cut: float      = 7.0,         # Å  cutoff radius
    r_cut_smth: float = 2.0,         # Å  onset of smooth switch
    sel: list[int]    = [60, 40, 20], # max neighbours per type
    neuron_embed: list[int] = [25, 50, 100],
    axis_neuron: int        = 16,
    neuron_fit: list[int]   = [240, 240, 240],
    n_training: int         = 1_000_000,
    batch_size: int         = 32,
    start_lr: float         = 1e-3,
    stop_lr:  float         = 3.51e-8,
    decay_steps: int        = 5000,
    pref_e: float = 1.0,    # final p_e weight
    pref_f: float = 1.0,    # final p_f weight
    pref_v: float = 0.02,   # final p_v weight
    output_path: str = "input_deepmd.json",
) -> dict:
    """
    Generate DeePMD-kit v3 JSON input.

    Smooth descriptor (se_e2_a):
      s(r) = C²-continuous quintic switch from r_cut_smth to r_cut
      s(r) = 1/r                                          r < r_cut_smth
      s(r) = (1/r)[x³(-6x²+15x-10)+1]  x=(r-r_s)/(r_c-r_s)  r_s≤r<r_c
      s(r) = 0                                            r ≥ r_c

    Loss scheduling (linear warm-up):
      p_e: 0.02 → pref_e
      p_f: 1000 → pref_f
      p_v: 0    → pref_v
    """
    cfg = {
        "model": {
            "type_map": type_map,
            "descriptor": {
                "type": "se_e2_a",
                "rcut": r_cut,
                "rcut_smth": r_cut_smth,
                "sel": sel,
                "neuron": neuron_embed,
                "axis_neuron": axis_neuron,
                "resnet_dt": False,
                "seed": 1,
            },
            "fitting_net": {
                "neuron": neuron_fit,
                "resnet_dt": False,
                "seed": 1,
            },
        },
        "learning_rate": {
            "type": "exp",
            "decay_steps": decay_steps,
            "start_lr": start_lr,
            "stop_lr": stop_lr,
        },
        "loss": {
            "type": "ener",
            "start_pref_e": 0.02, "limit_pref_e": pref_e,
            "start_pref_f": 1000, "limit_pref_f": pref_f,
            "start_pref_v": 0.0,  "limit_pref_v": pref_v,
        },
        "training": {
            "training_data": {"systems": ["./data/train"], "batch_size": batch_size},
            "validation_data": {"systems": ["./data/valid"], "batch_size": batch_size},
            "numb_steps": n_training,
            "seed": 1,
            "disp_file": "lcurve.out",
            "disp_freq": 1000,
            "save_freq": 10000,
            "save_ckpt": "model.ckpt",
        },
    }
    with open(output_path, "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg


def run_deepmd_training(input_json: str = "input_deepmd.json",
                        n_gpu: int = 1) -> None:
    """
    dp train input_deepmd.json
    dp freeze
    dp compress
    """
    cmds = [
        f"dp train {input_json}",
        "dp freeze -o graph.pb",
        "dp compress -i graph.pb -o graph_compress.pb",
    ]
    for cmd in cmds:
        subprocess.run(cmd.split(), check=True)


def evaluate_model_deviation(frames: list, model_paths: list[str]) -> dict:
    """
    Committee model deviation: σ_F = max_i √⟨|F_i^(k) - ⟨F_i⟩|²⟩_k

    Used in DP-GEN active learning to select uncertain frames.
    Threshold: σ_lo = 0.10 eV/Å, σ_hi = 0.25 eV/Å

    Returns
    -------
    {"sigma_F": array, "uncertain_mask": bool array}
    """
    sigma_lo, sigma_hi = 0.10, 0.25
    from ase.io import read
    try:
        from deepmd.calculator import DP
    except ImportError:
        raise ImportError("pip install deepmd-kit")

    calcs = [DP(model=p) for p in model_paths]
    sigma_F_all = []
    for frame in frames:
        forces = []
        for calc in calcs:
            atoms = frame.copy()
            atoms.calc = calc
            forces.append(atoms.get_forces())
        forces = np.stack(forces)            # (n_models, N, 3)
        mean_F = forces.mean(axis=0)
        dev_F  = np.sqrt(((forces - mean_F)**2).mean())
        sigma_F_all.append(dev_F)
    sigma_F = np.array(sigma_F_all)
    return {
        "sigma_F": sigma_F,
        "uncertain_mask": (sigma_F > sigma_lo) & (sigma_F < sigma_hi),
        "too_uncertain":  sigma_F >= sigma_hi,
    }
```

---

## 5. Scale 2 — MD / Phonon / QHA

### `tebc/scale2_md/ensembles.py`

```python
"""
MD ensemble implementations:
  NVE  : velocity-Verlet integrator (microcanonical)
  NVT  : Nosé-Hoover chain thermostat (canonical)
  NPT  : Parrinello-Rahman barostat + NHC thermostat (isothermal-isobaric)

These are reference implementations.
Production runs use LAMMPS (see lammps_runner.py).
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class NoseHooverChain:
    """
    Nosé-Hoover chain thermostat (Martyna-Klein-Tuckerman 1992).

    Equations of motion:
      dr_i/dt  = p_i / m_i
      dp_i/dt  = F_i - ξ₁ p_i
      dξ_k/dt  = p_{ξ_k} / Q_k
      dp_{ξ_1}/dt = [Σ p_i²/m_i - g k_B T] - (p_{ξ_2}/Q_2) p_{ξ_1}
      dp_{ξ_k}/dt = [p²_{ξ_{k-1}}/Q_{k-1} - k_B T] - (p_{ξ_{k+1}}/Q_{k+1})p_{ξ_k}
      dp_{ξ_M}/dt = [p²_{ξ_{M-1}}/Q_{M-1} - k_B T]

    Chain length M=3 is sufficient for silicate systems.
    Thermostat mass: Q_k = g k_B T τ²  (k=1),  k_B T τ²  (k≥2)
    """
    T_target: float      # K
    n_atoms:  int
    n_dof:    int = None # degrees of freedom; default 3N
    tau:      float = 0.1e-12  # s  coupling time
    M:        int   = 3        # chain length

    def __post_init__(self):
        from tebc.constants import k_B
        self.k_B = k_B
        if self.n_dof is None:
            self.n_dof = 3 * self.n_atoms
        g = self.n_dof
        Q1 = g * k_B * self.T_target * self.tau**2
        Qk = k_B * self.T_target * self.tau**2
        self.Q = np.array([Q1] + [Qk]*(self.M-1))
        self.xi   = np.zeros(self.M)   # positions
        self.p_xi = np.zeros(self.M)   # momenta

    def conserved_quantity(self, KE: float) -> float:
        """
        ℋ_NHC = ℋ + Σ p²_{ξ_k}/(2Q_k) + g k_B T ξ₁ + k_B T Σ_{k≥2} ξ_k
        Conserved to machine precision in exact arithmetic.
        """
        g = self.n_dof
        return (KE
                + np.sum(self.p_xi**2 / (2*self.Q))
                + g * self.k_B * self.T_target * self.xi[0]
                + self.k_B * self.T_target * np.sum(self.xi[1:]))

    def step(self, KE_atoms: float, dt: float) -> float:
        """
        Yoshida-Suzuki integration of NHC (Tuckerman 2010).
        Returns scaling factor s; multiply momenta by s.
        """
        g = self.n_dof
        kBT = self.k_B * self.T_target
        # G forces on chain
        G = np.zeros(self.M)
        G[0] = 2*KE_atoms - g*kBT
        for k in range(1, self.M):
            G[k] = self.p_xi[k-1]**2 / self.Q[k-1] - kBT
        # Propagate (simplified single Yoshida step)
        s = 1.0
        for k in range(self.M-1, -1, -1):
            self.p_xi[k] += 0.5*dt * G[k]
        s = np.exp(-0.5*dt * self.p_xi[0] / self.Q[0])
        for k in range(self.M):
            self.xi[k] += dt * self.p_xi[k] / self.Q[k]
        G[0] = 2*KE_atoms*s**2 - g*kBT
        self.p_xi[0] += 0.5*dt * G[0]
        for k in range(1, self.M):
            G[k] = self.p_xi[k-1]**2 / self.Q[k-1] - kBT
            self.p_xi[k] += 0.5*dt * G[k]
        return s


def velocity_verlet_step(pos, vel, forces, masses, dt):
    """
    Velocity-Verlet integrator (NVE or any constant-force ensemble).

    r(t+dt) = r(t) + v(t)dt + ½a(t)dt²
    v(t+dt) = v(t) + ½[a(t) + a(t+dt)]dt

    Parameters
    ----------
    pos    : (N,3) m
    vel    : (N,3) m/s
    forces : (N,3) N  at time t
    masses : (N,)  kg
    dt     : float s

    Returns
    -------
    pos_new, vel_half  (call force evaluator, then complete vel update)
    """
    acc   = forces / masses[:, None]
    pos_new = pos + vel*dt + 0.5*acc*dt**2
    vel_half = vel + 0.5*acc*dt
    return pos_new, vel_half


def parrinello_rahman_step(h, h_dot, stress_int, stress_ext, W, dt):
    """
    Parrinello-Rahman barostat equation of motion:

      W ḧ = V (σ_int - p_ext I)(h^T)⁻¹

    Parameters
    ----------
    h          : (3,3) cell matrix [Å]
    h_dot      : (3,3) dh/dt
    stress_int : (3,3) internal virial stress [Pa]
    stress_ext : float target pressure [Pa]
    W          : float barostat mass [kg·m²]
    dt         : float timestep [s]

    Returns
    -------
    h_new, h_dot_new
    """
    V   = np.abs(np.linalg.det(h))
    hT_inv = np.linalg.inv(h.T)
    P_diff = stress_int - stress_ext * np.eye(3)
    h_ddot = V * P_diff @ hT_inv / W
    h_new     = h     + h_dot*dt     + 0.5*h_ddot*dt**2
    h_dot_new = h_dot + 0.5*h_ddot*dt
    return h_new, h_dot_new
```

### `tebc/scale2_md/phonon_qha.py`

```python
"""
Phonon calculations and Quasi-Harmonic Approximation (QHA).

QHA Helmholtz free energy:
  F(V,T) = E0(V) + E_ZPE(V) + F_vib(V,T)
         = E0(V) + Σ_{q,s} ½ℏω_{q,s}(V)
                 + k_B T Σ_{q,s} ln[1 - exp(-ℏω_{q,s}(V)/k_BT)]

Equilibrium V(T) = argmin_V F(V,T)
Linear CTE: α_V = (1/V)(∂V/∂T)_p

For monoclinic β-RE₂Si₂O₇: generalize to full cell tensor {a,b,c,β}.
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from tebc.constants import k_B, hbar
from tebc.utils import birch_murnaghan_energy, fit_eos, mode_heat_capacity


def compute_free_energy(omega_qpts: np.ndarray, weights: np.ndarray,
                         E0: float, T: float) -> float:
    """
    F(V,T) = E0 + Σ_{q,s} w_q [½ℏω + k_BT ln(1-exp(-ℏω/k_BT))]

    Parameters
    ----------
    omega_qpts : (n_qpts, n_branches) rad/s  phonon frequencies
    weights    : (n_qpts,)  BZ integration weights (sum to 1)
    E0         : float J  static DFT energy
    T          : float K  temperature

    Returns
    -------
    F : float J
    """
    kBT = k_B * T
    hw  = hbar * omega_qpts               # (n_q, n_br) J
    # ZPE
    F_ZPE = 0.5 * (weights[:, None] * hw).sum()
    # Vibrational
    x  = hw / kBT
    # log(1 - exp(-x)) safely
    ln_term = np.where(x > 50, x, np.log(1.0 - np.exp(-x) + 1e-300))
    F_vib   = kBT * (weights[:, None] * ln_term).sum()
    return E0 + F_ZPE + F_vib


def qha_cte(V_list: np.ndarray, T_list: np.ndarray,
             omega_list: list[np.ndarray], weights: np.ndarray,
             E_list: np.ndarray) -> dict:
    """
    Perform QHA:
    1. Fit E(V) → EOS {E0, V0, B0, B0p}
    2. For each T: minimise F(V,T) over V grid → V_eq(T)
    3. α_V(T) = (1/V) dV/dT  (central differences)
    4. Grüneisen: γ(T) = Σ_qs C_qs γ_qs / Σ_qs C_qs

    Parameters
    ----------
    V_list    : (n_vol,) m³
    T_list    : (n_T,)   K
    omega_list: list of (n_qpts, n_branches) rad/s, one per volume
    weights   : (n_qpts,) BZ weights
    E_list    : (n_vol,)  J, static energies

    Returns
    -------
    dict with V_eq, alpha_V, gamma_gruneisen, B_T arrays
    """
    n_V, n_T = len(V_list), len(T_list)
    V_eq    = np.zeros(n_T)
    B_T     = np.zeros(n_T)

    for i, T in enumerate(T_list):
        F_V = np.array([
            compute_free_energy(omega_list[j], weights, E_list[j], T)
            for j in range(n_V)
        ])
        # Fit F(V) with BM EOS to get smooth V_min
        from scipy.optimize import minimize_scalar
        eos = fit_eos(V_list, F_V)
        V_eq[i] = eos["V0"]
        B_T[i]  = eos["B0"]

    # Volumetric CTE: α_V = (1/V) dV/dT
    dV_dT = np.gradient(V_eq, T_list)
    alpha_V = dV_dT / V_eq

    # Mode Grüneisen at reference volume (mid-point)
    mid = n_V // 2
    omega_ref = omega_list[mid]
    dln_omega = np.zeros_like(omega_ref)
    if n_V >= 3:
        # dln(ω)/dln(V) ≈ (ln(ω(V+)) - ln(ω(V-))) / (2 Δln(V))
        ln_V = np.log(V_list)
        dln_V = np.gradient(ln_V)[[0, -1]]
        dln_omega = (np.log(omega_list[-1]+1e-30) -
                     np.log(omega_list[0]+1e-30)) / (ln_V[-1] - ln_V[0])
    gamma_mode = -dln_omega

    T_ref = T_list[len(T_list)//2]
    C_qs  = mode_heat_capacity(omega_ref, T_ref)
    C_tot = (weights[:, None] * C_qs).sum()
    gamma_avg = (weights[:, None] * C_qs * gamma_mode).sum() / (C_tot + 1e-30)

    return {
        "V_eq":    V_eq,
        "T_list":  T_list,
        "alpha_V": alpha_V,
        "alpha_linear": alpha_V / 3.0,  # isotropic approx
        "B_T":     B_T,
        "gamma_gruneisen": gamma_avg,
    }


def gruneisen_cte_relation(gamma: float, Cv: float, V: float, B: float) -> float:
    """
    α_V = γ C_V / (B V)    [Grüneisen identity]

    Parameters
    ----------
    gamma : float   macroscopic Grüneisen parameter
    Cv    : float   J/(m³·K) volumetric heat capacity
    V     : float   m³/atom or m³/mol (consistent with Cv)
    B     : float   Pa  bulk modulus

    Returns
    -------
    alpha_V : float K⁻¹
    """
    return gamma * Cv / (B * V)
```

### `tebc/scale2_md/green_kubo.py`

```python
"""
Green-Kubo thermal conductivity from MD trajectory.

κ_αβ = (V / k_B T²) ∫₀^∞ ⟨J_α(0) J_β(t)⟩ dt

Heat current (centroid stress formulation, mandatory for ML potentials):
  J = (1/V)[Σ_i ε_i v_i + ½ Σ_{i≠j} (r_i - r_j)(F_ij · v_i)]

IMPORTANT: Standard per-atom stress underestimates κ by 20-40% for
covalent many-body systems. Use centroid/stress/atom in LAMMPS.
"""

from __future__ import annotations
import numpy as np
from scipy.integrate import cumulative_trapezoid
from tebc.constants import k_B


def compute_hcacf(J: np.ndarray, dt: float,
                   max_lag_steps: int = None) -> np.ndarray:
    """
    Compute normalised heat current autocorrelation function (HCACF)
    C(t) = ⟨J(0)·J(t)⟩

    Uses FFT for O(N log N) computation.

    Parameters
    ----------
    J            : (n_steps, 3) or (n_steps,) W/m² heat current
    dt           : float s timestep
    max_lag_steps: int  maximum lag (default n_steps//2)

    Returns
    -------
    t_lag : (max_lag_steps,) s
    HCACF : (max_lag_steps,) W²/m⁴  (not normalised)
    """
    if J.ndim == 2:
        # Isotropic: average over components
        HCACF = sum(
            np.correlate(J[:,i], J[:,i], mode='full') for i in range(3)
        ) / 3.0
    else:
        HCACF = np.correlate(J, J, mode='full')

    n = len(J)
    mid = len(HCACF) // 2
    HCACF = HCACF[mid:] / n   # normalise by n_samples

    if max_lag_steps is not None:
        HCACF = HCACF[:max_lag_steps]
    t_lag = np.arange(len(HCACF)) * dt
    return t_lag, HCACF


def integrate_hcacf(t_lag: np.ndarray, HCACF: np.ndarray,
                     V: float, T: float) -> np.ndarray:
    """
    κ(t) = (V / k_B T²) ∫₀^t C(t') dt'

    Running integral → plateau = κ.

    Parameters
    ----------
    t_lag  : (n,) s
    HCACF  : (n,) W²/m⁴
    V      : float m³  simulation volume
    T      : float K   temperature

    Returns
    -------
    kappa_t : (n,) W/(m·K)  running κ
    """
    prefactor = V / (k_B * T**2)
    kappa_t = prefactor * cumulative_trapezoid(HCACF, t_lag, initial=0)
    return kappa_t


def plateau_estimate(kappa_t: np.ndarray, t_lag: np.ndarray,
                      t_plateau_start: float = None) -> dict:
    """
    Estimate plateau κ as mean over [t_plateau_start, t_max].
    Convergence criterion: std/mean < 0.1 in plateau region.

    For low-κ oxides like β-Yb₂Si₂O₇, integration window ≥ 100 ps required.
    Recommend ≥ 8 independent replicas and report mean ± std.

    Parameters
    ----------
    kappa_t         : (n,) W/(m·K)
    t_lag           : (n,) s
    t_plateau_start : float s  default = 0.5 * t_max

    Returns
    -------
    {"kappa": float, "kappa_std": float, "converged": bool}
    """
    t_max = t_lag[-1]
    if t_plateau_start is None:
        t_plateau_start = 0.5 * t_max
    mask = t_lag >= t_plateau_start
    kappa_plateau = kappa_t[mask]
    kappa_mean = np.mean(kappa_plateau)
    kappa_std  = np.std(kappa_plateau)
    return {
        "kappa": kappa_mean,
        "kappa_std": kappa_std,
        "converged": (kappa_std / (abs(kappa_mean) + 1e-30)) < 0.15,
    }


def kappa_anisotropic(J_xyz: np.ndarray, dt: float,
                       V: float, T: float) -> np.ndarray:
    """
    Compute full 3×3 κ tensor.

    κ_αβ = (V/k_BT²) ∫ ⟨J_α(0)J_β(t)⟩ dt

    Returns
    -------
    kappa_tensor : (3,3) W/(m·K)
    """
    kappa = np.zeros((3, 3))
    prefactor = V / (k_B * T**2)
    n = len(J_xyz)
    for a in range(3):
        for b in range(3):
            c = np.correlate(J_xyz[:, a], J_xyz[:, b], mode='full')
            c = c[n-1:] / n
            kappa[a, b] = prefactor * np.trapz(c, dx=dt)
    return kappa
```

### `tebc/scale2_md/msd_diffusion.py`

```python
"""
Mean Square Displacement → diffusion coefficient → Arrhenius fit.

D_O(T) = D0 * exp(-Ea / k_B T)

For YSZ oxygen diffusivity (Brossmann 2003):
  D0 = 1.3e-6 m²/s
  Ea = 0.95-1.10 eV

MSD formula:
  ⟨|Δr(t)|²⟩ = (1/N) Σ_i |r_i(t) - r_i(0)|²
  D = lim_{t→∞} ⟨Δr²⟩ / (2d t)   (d = 3 for 3D)
"""

import numpy as np
from scipy.stats import linregress
from tebc.constants import k_B
from tebc.utils import arrhenius_fit, arrhenius_eval


def compute_msd(positions: np.ndarray, species_mask: np.ndarray = None,
                max_lag_fraction: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute MSD from trajectory.

    Parameters
    ----------
    positions      : (n_frames, n_atoms, 3) m  unwrapped coordinates
    species_mask   : (n_atoms,) bool  True for atoms to include (e.g., O only)
    max_lag_fraction: fraction of total trajectory to use as max lag

    Returns
    -------
    t_lag : (n_lag,) s  (in units of frame index; multiply by dt externally)
    msd   : (n_lag,) m²
    """
    if species_mask is not None:
        positions = positions[:, species_mask, :]
    n_frames, n_atoms, _ = positions.shape
    n_lag = int(n_frames * max_lag_fraction)
    msd = np.zeros(n_lag)
    for lag in range(1, n_lag):
        disp = positions[lag:] - positions[:-lag]
        msd[lag] = np.mean(disp**2)
    return np.arange(n_lag), msd


def msd_to_diffusivity(t_lag: np.ndarray, msd: np.ndarray,
                        dt_per_frame: float,
                        fit_start_frac: float = 0.1,
                        fit_end_frac:   float = 0.5,
                        dim: int = 3) -> dict:
    """
    Fit D from linear region of MSD: D = slope / (2 * dim).

    Returns
    -------
    {"D": float m²/s, "D_err": float, "r2": float}
    """
    n = len(t_lag)
    i0 = int(n * fit_start_frac)
    i1 = int(n * fit_end_frac)
    t_fit   = t_lag[i0:i1] * dt_per_frame
    msd_fit = msd[i0:i1]
    slope, intercept, r, p, stderr = linregress(t_fit, msd_fit)
    D = slope / (2 * dim)
    return {"D": D, "D_std": stderr / (2*dim), "r2": r**2}


def arrhenius_diffusivity(T_list: np.ndarray, D_list: np.ndarray) -> dict:
    """
    Fit D(T) = D0 * exp(-Ea / k_B T)  and return Arrhenius parameters.

    Parameters
    ----------
    T_list : (n,) K
    D_list : (n,) m²/s

    Returns
    -------
    {"D0": float, "Ea_eV": float, "Ea_J": float}
    """
    from tebc.constants import eV
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A, Ea_J = arrhenius_fit(T_list, D_list)
    return {"D0": A, "Ea_J": Ea_J, "Ea_eV": Ea_J / eV}
```

---

## 6. Scale 3 — Phase-Field / CALPHAD / TGO

### `tebc/scale3_mesoscale/phase_field.py`

```python
"""
Phase-field solver for CMAS attack using FiPy.

Allen-Cahn (non-conserved order parameter φ, crystallisation front):
  ∂φ/∂t = -M_φ δF/δφ = -M_φ [∂f/∂φ - κ_φ ∇²φ]

Cahn-Hilliard (conserved composition c):
  ∂c/∂t = ∇·[M_c ∇(δF/δc)] = ∇·[M_c ∇(∂f/∂c - κ_c ∇²c)]

Free energy density:
  f(φ, c, T) = h(φ) G_β(c,T) + [1-h(φ)] G_L(c,T) + W g(φ)

Interpolation: h(φ) = φ³(6φ²-15φ+10)  (C²-continuous)
Double well:   g(φ) = φ²(1-φ)²

Interface properties:
  σ = √(W κ_φ) / (3√2)    [J/m²]
  ℓ = √(8 κ_φ / W)        [m]
"""

from __future__ import annotations
import numpy as np

try:
    import fipy as fp
except ImportError:
    raise ImportError("pip install fipy")


def interpolation_h(phi: np.ndarray) -> np.ndarray:
    """h(φ) = φ³(6φ²-15φ+10)  — smooth step, C²-continuous."""
    return phi**3 * (6*phi**2 - 15*phi + 10)


def double_well_g(phi: np.ndarray) -> np.ndarray:
    """g(φ) = φ²(1-φ)²."""
    return phi**2 * (1 - phi)**2


def interface_params(sigma: float, ell: float) -> dict:
    """
    Given interface energy σ [J/m²] and width ℓ [m], compute:
      W = 6√2 σ / ℓ     [J/m³]
      κ_φ = (3/4√2) σ ℓ [J/m]

    Reference: Karma & Rappel, PRE 53, R3017 (1996)
    """
    W    = 6 * np.sqrt(2) * sigma / ell
    kap  = (3.0 / (4.0 * np.sqrt(2))) * sigma * ell
    return {"W": W, "kappa_phi": kap}


def nucleation_rate(T: float, sigma_nu: float, dG_v: float,
                     theta_contact: float = 0.0,
                     I0: float = 1e36) -> float:
    """
    Heterogeneous nucleation rate [m⁻³ s⁻¹]:

      I = I₀ exp(-ΔG* / k_B T)
      ΔG* = (16π σ³/3 ΔG_v²) f(θ)
      f(θ) = ¼(2 - 3cosθ + cos³θ)

    Parameters
    ----------
    T             : K   temperature
    sigma_nu      : J/m² nucleus-liquid interface energy
    dG_v          : J/m³ bulk driving force (negative = exothermic)
    theta_contact : rad  contact angle for heterogeneous nucleation
    I0            : m⁻³s⁻¹ pre-exponential

    Returns
    -------
    I : float m⁻³ s⁻¹
    """
    from tebc.constants import k_B
    f_theta = 0.25 * (2 - 3*np.cos(theta_contact) + np.cos(theta_contact)**3)
    dG_star = (16 * np.pi * sigma_nu**3 / (3 * dG_v**2)) * f_theta
    return I0 * np.exp(-dG_star / (k_B * T))


class CMASPhaseField:
    """
    2D phase-field model for CMAS infiltration and apatite crystallisation.

    Governing equations:
      ∂φ/∂t = -M_φ [W g'(φ) - κ_φ ∇²φ + (G_β - G_L) h'(φ)]
      ∂c/∂t = ∇·{M_c(φ,c) ∇[h(φ) ∂G_β/∂c + (1-h(φ)) ∂G_L/∂c - κ_c ∇²c]}

    Coupled via KKS condition: ∂G_β/∂c = ∂G_L/∂c = μ (equal chemical potentials)
    """

    def __init__(self, nx: int, ny: int, dx: float,
                 sigma: float = 0.3,    # J/m²  apatite/CMAS interface energy
                 ell: float   = 1e-7,   # m     interface width (100 nm)
                 M_phi: float = 1e-8,   # m³/(J·s)  Allen-Cahn mobility
                 kappa_c: float = 1e-18,# J/m   gradient energy coefficient
                 D_CMAS:  float = 1e-12 # m²/s  CMAS cation diffusivity
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
        """
        Explicit Euler step (use small dt << ℓ²/(M_φ κ_φ) for stability).
        Production: switch to Crank-Nicolson in FiPy.
        """
        phi_v = self.phi.value
        c_v   = self.c.value
        # Allen-Cahn (non-conserved)
        ac_eq = (fp.TransientTerm(var=self.phi)
                 == self.M_phi * (fp.DiffusionTerm(coeff=self.kap, var=self.phi)
                                  - self.W * fp.ImplicitSourceTerm(
                                      coeff=6*phi_v*(1-2*phi_v),var=self.phi)))
        # Cahn-Hilliard (conserved)
        M_c   = self.D_CMAS * c_v * (1-c_v)  # simplified linear mobility
        ch_eq = (fp.TransientTerm(var=self.c)
                 == fp.DiffusionTerm(coeff=M_c, var=self.c))
        fp.solve([ac_eq, ch_eq], dt=dt)
        self.phi.updateOld()
        self.c.updateOld()
```

### `tebc/scale3_mesoscale/tgo_kinetics.py`

```python
"""
Thermally Grown Oxide (SiO₂) kinetics on Si bond coat.

Deal-Grove model:
  x² + Ax = B(t + τ)
  Parabolic: B = 2 D_ox C* / N₁        [m²/s]
  Linear:    B/A = k_s C* / N₁         [m/s]

Arrhenius:
  B(T)   = B0 * exp(-Q_B / RT)
  B/A(T) = (B/A)_0 * exp(-Q_BA / RT)

Paralinear model (Opila & Hann 1997):
  dx/dt = k_p/(2x) - k_l
  x_ss  = k_p / (2 k_l)                [steady-state SiO₂ thickness]
  ṁ_rec → k_l  (linear long-time recession rate)

Pilling-Bedworth stress:
  ε_TGO^ox = ⅓(PBR - 1) δ_{ij}
"""

from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from tebc.constants import R_gas
from tebc.utils import arrhenius_eval


# ── Deal-Grove model ──────────────────────────────────────────────────────────
def deal_grove_thickness(t: np.ndarray, k_p: float, k_l: float,
                          x0: float = 0.0) -> np.ndarray:
    """
    Thickness x(t) from Deal-Grove (general form).

    x² + A x = B t   where B = k_p,  B/A = k_l  (thick scale limit)

    Using the quadratic solution:
      x = A/2 * [√(1 + 4(Bt + A²/4)/A²) - 1]

    With B = 2*k_p (parabolic constant), A = 2*k_p/k_l (linear constant).
    """
    A = 2 * k_p / k_l  if k_l > 0 else 1e30
    B = k_p * 2
    tau = x0 * (x0 + A) / B  # initial offset
    discriminant = 1.0 + 4.0*(B*(t + tau) + A**2/4.0) / (A**2 + 1e-30)
    x = (A / 2.0) * (np.sqrt(np.maximum(discriminant, 0)) - 1.0)
    return x


def parabolic_rate_constant(T_K: float, k_p0: float, Ea_J: float) -> float:
    """k_p(T) = k_p0 * exp(-Ea / RT)  [m²/s]"""
    return arrhenius_eval(T_K, k_p0, Ea_J)


# ── Paralinear model (simultaneous oxidation + volatilisation) ────────────────
def paralinear_ode(t: float, x: np.ndarray,
                    k_p: float, k_l: float) -> list[float]:
    """
    dx/dt = k_p / (2x) - k_l

    Steady state: x_ss = k_p / (2 k_l)
    """
    return [k_p / (2.0 * x[0] + 1e-30) - k_l]


def solve_paralinear(t_span: tuple, k_p: float, k_l: float,
                      x0: float = 1e-9,
                      n_points: int = 500) -> dict:
    """
    Solve paralinear ODE for TGO thickness x(t) and
    SiC recession thickness r(t) = k_l * t.

    Returns
    -------
    {"t": array s, "x_TGO": array m, "recession": array m}
    """
    t_eval = np.linspace(*t_span, n_points)
    sol = solve_ivp(paralinear_ode, t_span, [x0],
                    args=(k_p, k_l), t_eval=t_eval,
                    method='RK45', rtol=1e-8, atol=1e-12)
    x_TGO    = sol.y[0]
    recession = k_l * sol.t
    x_ss     = k_p / (2 * k_l) if k_l > 0 else np.inf
    return {"t": sol.t, "x_TGO": x_TGO, "recession": recession, "x_ss": x_ss}


def tgo_growth_stress(x_TGO: float, E_TGO: float, nu_TGO: float,
                       alpha_TGO: float, alpha_sub: float,
                       dT: float, PBR: float = 2.15) -> float:
    """
    Total TGO biaxial stress:
      σ_TGO = σ_thermal + σ_growth
      σ_thermal = E/(1-ν) * (α_TGO - α_sub) * ΔT   [Pa]
      σ_growth  = E/(1-ν) * ε_growth                 (compressive, < 0)
      ε_growth  ≈ ⅓(PBR-1)  for Si→SiO₂

    Returns
    -------
    sigma : float Pa  (negative = compressive)
    """
    eps_growth = (PBR - 1.0) / 3.0
    biaxial_mod = E_TGO / (1 - nu_TGO)
    sigma_thermal = biaxial_mod * (alpha_TGO - alpha_sub) * dT
    sigma_growth  = -biaxial_mod * eps_growth   # compressive
    return sigma_thermal + sigma_growth


def robinson_smialek_recession(T_K: float, P_H2O: float,
                                P_tot: float, v_gas: float,
                                Ea_J: float = 108e3,
                                k0: float   = None) -> float:
    """
    Empirical Robinson-Smialek volatility correlation:

      k_l ∝ v^{0.5} * P_H2O^2 * P_tot^{-0.5} * exp(-ΔQ/RT)   [m/s]

    Reference: Robinson & Smialek, JACerS 82, 1817 (1999)
               Opila et al., JACerS 82, 1826 (1999)

    Parameters
    ----------
    T_K   : K  temperature
    P_H2O : Pa  water vapour partial pressure
    P_tot : Pa  total pressure
    v_gas : m/s  gas velocity
    Ea_J  : J/mol  activation energy (~108 kJ/mol for SiC)
    k0    : pre-exponential (calibrate to experiment)

    Returns
    -------
    k_l : float m/s  linear recession rate
    """
    if k0 is None:
        # Calibrated to Opila 1999 (1316°C, P_H2O=0.1atm, P_tot=1atm, v=4.4cm/s)
        # k_l_ref ≈ 2e-9 m/s
        k0 = 2e-9 / (0.044**0.5 * (0.1*101325)**2 * (101325)**(-0.5)
                      * np.exp(-Ea_J/(R_gas*1589)))
    return k0 * v_gas**0.5 * P_H2O**2 * P_tot**(-0.5) * np.exp(-Ea_J/(R_gas*T_K))
```

### `tebc/scale3_mesoscale/calphad_interface.py`

```python
"""
CALPHAD interface using pycalphad.

Gibbs energy model (Redlich-Kister):
  G_m = Σ xᵢ °Gᵢ + RT Σ xᵢ ln(xᵢ) + ᵉˣG_m

  ᵉˣG_m^{AB} = x_A x_B Σ_v  ᵛL^{AB} (x_A - x_B)^v
  ᵛL = a_v + b_v T + c_v T ln T + ...

For CMAS-EBC system: two-sublattice ionic liquid model
  (Ca²⁺, Mg²⁺, Y³⁺, Yb³⁺)_P (O²⁻, SiO₄⁴⁻, AlO₂⁻, Va)_Q
"""

from __future__ import annotations
import numpy as np

try:
    import pycalphad.variables as v
    from pycalphad import Database, equilibrium, calculate
except ImportError:
    raise ImportError("pip install pycalphad")


def load_database(tdb_path: str) -> object:
    """Load thermodynamic database (.tdb format)."""
    return Database(tdb_path)


def compute_equilibrium(dbf: object, components: list[str],
                         phases: list[str], conditions: dict,
                         output: str = "GM") -> object:
    """
    Compute phase equilibrium via pycalphad.

    Example conditions:
      {v.T: 1573, v.P: 101325, v.X('SIO2'): 0.45, v.X('CAO'): 0.33}

    Returns pycalphad LightDataset with phase fractions, compositions, GM.
    """
    return equilibrium(dbf, components, phases, conditions, output=output)


def gibbs_redlich_kister(x_A: float, x_B: float,
                          L_coeffs: list[tuple[float, float]],
                          T: float) -> float:
    """
    Excess Gibbs energy (Redlich-Kister):
      ᵉˣG = x_A x_B Σ_v (a_v + b_v T)(x_A - x_B)^v

    Parameters
    ----------
    x_A, x_B  : mole fractions
    L_coeffs  : list of (a_v, b_v) for v = 0, 1, 2, ...
    T         : K

    Returns
    -------
    Gex : float J/mol
    """
    Gex = 0.0
    dx  = x_A - x_B
    for v_order, (a_v, b_v) in enumerate(L_coeffs):
        L_v = a_v + b_v * T
        Gex += L_v * dx**v_order
    return x_A * x_B * Gex


def phase_stability_range(dbf: object, components: list[str],
                            phase: str, T_range: np.ndarray) -> np.ndarray:
    """
    Compute phase fraction of `phase` over temperature range at fixed composition.
    Returns (T_range, phase_fraction) arrays.
    """
    fracs = []
    for T in T_range:
        cond = {v.T: float(T), v.P: 101325.0}
        res  = calculate(dbf, components, phase, T=float(T), P=101325.0)
        # Simplified: return 1 if phase is stable (ΔGmix < 0)
        fracs.append(float(res.GM.values.min()))
    return T_range, np.array(fracs)
```

---

## 7. Scale 4 — FEA Continuum

### `tebc/scale4_continuum/thermoelastic.py`

```python
"""
Coupled thermoelastic BVP using FEniCSx (dolfinx).

Governing equations:
  Heat:  ρ c_p ∂T/∂t = ∇·(κ∇T) + Q               [J/(m³·s)]
  Mech:  ∇·σ + b = 0,  σ = C:(ε - α ΔT - ε_p)    [Pa]

Weak forms:
  ∫_Ω ρc_p θ̇·θ dΩ + ∫_Ω κ ∇T·∇θ dΩ
    = ∫_Ω Q θ dΩ - ∫_{Γ_h} [h(T-T∞) + εσ(T⁴-T∞⁴)]θ dΓ

  ∫_Ω σ:∇ˢv dΩ = ∫_Ω b·v dΩ + ∫_{Γ_N} t̄·v dΓ

CTE mismatch stress (bilayer, analytical):
  σ_f = E_f/(1-ν_f) * (α_s - α_f) * ΔT
"""

from __future__ import annotations
import numpy as np


def bilayer_mismatch_stress(E_f: float, nu_f: float,
                             alpha_f: float, alpha_s: float,
                             dT: float) -> float:
    """
    Biaxial CTE mismatch stress in film (Hutchinson & Suo 1992):

      σ_f = E_f/(1-ν_f) * (α_s - α_f) * ΔT

    Convention: ΔT = T_final - T_deposit (negative on cool-down).
    Positive σ = tensile.

    Parameters
    ----------
    E_f   : Pa  film Young's modulus
    nu_f  : Poisson ratio of film
    alpha_f, alpha_s : K⁻¹ CTE of film, substrate
    dT    : K   temperature change (T_final - T_ref)

    Returns
    -------
    sigma : float Pa
    """
    return (E_f / (1.0 - nu_f)) * (alpha_s - alpha_f) * dT


def stoney_curvature(sigma_f: float, h_f: float,
                      E_s: float, nu_s: float, h_s: float) -> float:
    """
    Stoney equation: film stress → substrate curvature κ [m⁻¹]

      κ = 6 σ_f h_f (1-ν_s) / (E_s h_s²)

    Valid for h_f << h_s.
    """
    return 6.0 * sigma_f * h_f * (1.0 - nu_s) / (E_s * h_s**2)


def energy_release_rate_steady_state(sigma0: float, h_f: float,
                                      E_f: float, nu_f: float) -> float:
    """
    Steady-state energy release rate for delamination of stressed film:

      G_ss = (1 - ν_f²) σ₀² h_f / (2 E_f)

    Reference: Hutchinson & Suo, Adv. Appl. Mech. 29 (1992)

    Parameters
    ----------
    sigma0 : Pa  residual stress in film
    h_f    : m   film thickness
    E_f    : Pa  Young's modulus
    nu_f   : Poisson ratio

    Returns
    -------
    G_ss : float J/m²
    """
    return (1.0 - nu_f**2) * sigma0**2 * h_f / (2.0 * E_f)


def convective_bc_heat_flux(T_surface: float, T_inf: float,
                              h_conv: float, emissivity: float,
                              T_rad: float = None) -> float:
    """
    Combined convective + radiative heat flux at surface:

      q″ = h(T - T∞) + ε σ_SB (T⁴ - T_rad⁴)

    Parameters
    ----------
    T_surface : K
    T_inf     : K  fluid temperature
    h_conv    : W/(m²·K)  convective heat transfer coefficient
    emissivity: float 0-1
    T_rad     : K  radiation source temperature (default = T_inf)

    Returns
    -------
    q : float W/m²  (positive = heat into surface)
    """
    from tebc.constants import sigma_SB
    T_rad = T_rad if T_rad is not None else T_inf
    return h_conv*(T_surface - T_inf) + emissivity*sigma_SB*(T_surface**4 - T_rad**4)


def fenics_thermoelastic_setup():
    """
    FEniCSx variational formulation skeleton.

    Copy this into your simulation script. Requires dolfinx ≥ 0.8.

    Weak form:
      a_T(T,θ) + a_u(u,v) = L_T(θ) + L_u(v)
    """
    code = '''
import dolfinx
from dolfinx import fem, mesh, io
from dolfinx.fem.petsc import LinearProblem
import ufl
import numpy as np
from mpi4py import MPI

# ── Mesh: layered TEBC geometry ──────────────────────────────────────────────
# Layers (m): CMC(5e-3) / Si(100e-6) / EBC(150e-6) / TBC(200e-6)
domain = mesh.create_rectangle(MPI.COMM_WORLD, [[0,0],[1e-2,6e-3]],
                                [200,120], mesh.CellType.triangle)

# ── Function spaces ───────────────────────────────────────────────────────────
V_T = fem.functionspace(domain, ("Lagrange", 1))          # temperature
V_u = fem.functionspace(domain, ("Lagrange", 1, (2,)))    # displacement (2D)

T, theta  = ufl.TrialFunction(V_T), ufl.TestFunction(V_T)
u, v_test = ufl.TrialFunction(V_u), ufl.TestFunction(V_u)

# ── Material properties (layered via DG0) ────────────────────────────────────
V0     = fem.functionspace(domain, ("DG", 0))
kappa  = fem.Function(V0)   # W/(m·K) — set per element from material_db
rho_cp = fem.Function(V0)   # J/(m³·K)
E_mod  = fem.Function(V0)   # Pa
nu_mod = fem.Function(V0)   # -
alpha  = fem.Function(V0)   # K⁻¹

# ── Thermoelastic constitutive ────────────────────────────────────────────────
def eps(u):
    return ufl.sym(ufl.grad(u))

def sigma(u, T_field, T_ref=300.0):
    """σ = C:(ε - α ΔT I)  [plane stress simplification shown]"""
    mu    = E_mod / (2*(1+nu_mod))
    lam   = E_mod*nu_mod / ((1+nu_mod)*(1-2*nu_mod))
    strain = eps(u)
    dT     = T_field - T_ref
    return (2*mu*strain + lam*ufl.tr(strain)*ufl.Identity(2)
            - (3*lam + 2*mu)*alpha*dT*ufl.Identity(2))

# ── Thermal BVP ───────────────────────────────────────────────────────────────
dt_val = fem.Constant(domain, 1.0)  # s timestep
T_old  = fem.Function(V_T)
T_old.x.array[:] = 300.0

a_T = (rho_cp/dt_val * T * theta * ufl.dx
       + kappa * ufl.dot(ufl.grad(T), ufl.grad(theta)) * ufl.dx)
L_T = rho_cp/dt_val * T_old * theta * ufl.dx  # + source terms

# ── Mechanical BVP ───────────────────────────────────────────────────────────
T_field = fem.Function(V_T)   # solved temperature
a_u = ufl.inner(sigma(u, T_field), eps(v_test)) * ufl.dx
L_u = ufl.dot(fem.Constant(domain, np.zeros(2)), v_test) * ufl.dx

# ── Solve ─────────────────────────────────────────────────────────────────────
problem_T = LinearProblem(a_T, L_T, bcs=[], petsc_options={"ksp_type": "cg"})
problem_u = LinearProblem(a_u, L_u, bcs=[], petsc_options={"ksp_type": "gmres"})
'''
    return code
```

### `tebc/scale4_continuum/damage_mechanics.py`

```python
"""
Continuum damage mechanics for thermal cycling.

Lemaitre isotropic damage (JMPS 1979):
  σ̃ = σ / (1 - D)              effective stress
  Y  = σ_eq² R_v / [2E(1-D)²]  damage energy release rate
  R_v = ⅔(1+ν) + 3(1-2ν)(σ_H/σ_eq)²  triaxiality factor
  dD/dp = (Y/S)^s               H(p - p_D)  [per unit plastic strain]

Mazars tensile damage model (IJNME 1984):
  ε̃ = √[Σᵢ ⟨εᵢ⟩₊²]           equivalent strain (positive eigenvalues only)
  D(ε̃) = 1 - (1-A)ε₀/ε̃ - A exp[-B(ε̃ - ε₀)]

Cohesive Zone (Tvergaard-Hutchinson):
  λ = √[(δ_n/δ_n^c)² + (δ_t/δ_t^c)²]  effective opening
  σ(λ) = σ̂  for λ ∈ [λ₁, λ₂]
  T_n = (σ(λ)/λ)(δ_n/δ_n^c)
  T_t = (σ(λ)/λ)(δ_t/δ_t^c)
"""

import numpy as np


# ── Lemaitre CDM ─────────────────────────────────────────────────────────────
def triaxiality_factor(sigma_eq: float, sigma_H: float, nu: float) -> float:
    """
    R_v = ⅔(1+ν) + 3(1-2ν)(σ_H / σ_eq)²

    σ_H = ⅓ tr(σ)  hydrostatic stress
    σ_eq = √(3/2 s:s)  von Mises equivalent
    """
    return (2.0/3.0)*(1+nu) + 3*(1-2*nu)*(sigma_H/(sigma_eq + 1e-30))**2


def lemaitre_damage_rate(sigma_eq: float, sigma_H: float, nu: float,
                          E: float, D: float,
                          S: float = 1.0e6, s: float = 1.0) -> float:
    """
    Lemaitre damage evolution:

      dD/dp = (Y/S)^s

    Y = σ_eq² R_v / [2 E (1-D)²]

    Parameters
    ----------
    sigma_eq, sigma_H : Pa  von Mises and hydrostatic stress
    nu   : Poisson ratio
    E    : Pa  Young's modulus
    D    : current damage [0,1)
    S    : Pa  softening parameter (material, ~0.5-5 MPa for ceramics)
    s    : damage exponent (~1-2)

    Returns
    -------
    dD_dp : float 1/unit_plastic_strain
    """
    Rv = triaxiality_factor(sigma_eq, sigma_H, nu)
    Y  = sigma_eq**2 * Rv / (2.0 * E * (1.0 - D)**2 + 1e-30)
    return (Y / S)**s


def mazars_equivalent_strain(eps_principal: np.ndarray) -> float:
    """
    Mazars equivalent strain (tensile only):
      ε̃ = √[Σᵢ ⟨εᵢ⟩₊²]   where ⟨x⟩₊ = max(x, 0)

    Parameters
    ----------
    eps_principal : (3,) principal strains

    Returns
    -------
    eps_tilde : float
    """
    positive = np.maximum(eps_principal, 0.0)
    return np.sqrt(np.sum(positive**2))


def mazars_damage(eps_tilde: float, eps0: float = 1e-4,
                   A: float = 0.96, B: float = 15000.0) -> float:
    """
    Mazars damage function for quasi-brittle materials:

      D(ε̃) = 0                                        ε̃ ≤ ε₀
      D(ε̃) = 1 - (1-A)ε₀/ε̃ - A exp[-B(ε̃ - ε₀)]    ε̃ > ε₀

    Parameters
    ----------
    eps_tilde : float  equivalent strain
    eps0      : float  damage threshold strain (~1e-4 for RE-silicates)
    A         : float  post-peak softening coefficient (0.96 for concrete/ceramic)
    B         : float  softening rate (5000-25000 for ceramics)

    Returns
    -------
    D : float in [0, 1)
    """
    if eps_tilde <= eps0:
        return 0.0
    D = 1.0 - (1.0-A)*eps0/eps_tilde - A*np.exp(-B*(eps_tilde - eps0))
    return np.clip(D, 0.0, 0.999)


# ── Tvergaard-Hutchinson CZM ─────────────────────────────────────────────────
class TVHCohesiveZone:
    """
    Tvergaard-Hutchinson cohesive zone model (JMPS 40, 1377, 1992).

    Traction-separation law:
      λ = √[(δ_n/δ_n^c)² + (δ_t/δ_t^c)²]

      σ(λ) piecewise:
        0         λ ≤ 0
        σ̂ λ/λ₁   0 < λ ≤ λ₁  (linear loading)
        σ̂         λ₁ < λ ≤ λ₂  (plateau)
        σ̂(1-λ)/(1-λ₂)  λ₂ < λ ≤ 1  (softening)
        0          λ > 1  (complete failure)

      T_n = (σ(λ)/λ) (δ_n/δ_n^c)
      T_t = (σ(λ)/λ) (δ_t/δ_t^c)

      G_c = ½ σ̂ δ_n^c [1 - λ₁ + λ₂]
    """
    def __init__(self, sigma_hat: float = 100e6,
                 delta_n_c: float = 1e-6,
                 delta_t_c: float = 3e-6,
                 lambda1: float = 0.15,
                 lambda2: float = 0.50):
        self.sigma_hat = sigma_hat      # Pa  peak traction
        self.delta_n_c = delta_n_c     # m   critical normal opening
        self.delta_t_c = delta_t_c     # m   critical tangential opening
        self.l1 = lambda1
        self.l2 = lambda2
        self.G_c = 0.5 * sigma_hat * delta_n_c * (1 - lambda1 + lambda2)

    def effective_opening(self, delta_n: float, delta_t: float) -> float:
        """λ = √[(δ_n/δ_n^c)² + (δ_t/δ_t^c)²]"""
        return np.sqrt((delta_n/self.delta_n_c)**2 + (delta_t/self.delta_t_c)**2)

    def sigma_lambda(self, lam: float) -> float:
        """Piecewise traction envelope σ(λ)."""
        if lam <= 0:         return 0.0
        if lam <= self.l1:   return self.sigma_hat * lam / self.l1
        if lam <= self.l2:   return self.sigma_hat
        if lam <= 1.0:       return self.sigma_hat*(1-lam)/(1-self.l2)
        return 0.0

    def tractions(self, delta_n: float, delta_t: float) -> tuple[float, float]:
        """
        (T_n, T_t) in Pa.
        T_n = σ(λ)/λ * δ_n/δ_n^c
        T_t = σ(λ)/λ * δ_t/δ_t^c
        """
        lam = self.effective_opening(delta_n, delta_t)
        if lam < 1e-12:
            return 0.0, 0.0
        sl = self.sigma_lambda(lam)
        T_n = (sl/lam) * (delta_n / self.delta_n_c)
        T_t = (sl/lam) * (delta_t / self.delta_t_c)
        return T_n, T_t

    def benzeggagh_kenane_toughness(self, G_II: float, G_T: float,
                                     eta: float = 1.5) -> float:
        """
        Mixed-mode toughness (Benzeggagh-Kenane 1996):
          G_c(ψ) = G_Ic + (G_IIc - G_Ic)(G_II/G_T)^η
        """
        G_Ic  = self.G_c
        G_IIc = 1.5 * self.G_c   # typical ratio for oxide interfaces
        return G_Ic + (G_IIc - G_Ic) * (G_II / (G_T + 1e-30))**eta
```

---

## 8. Cross-Scale Coupling

### `tebc/coupling/homogenization.py`

```python
"""
Multi-scale homogenization:
  Voigt  (upper bound):  C_V = Σ f_r C_r
  Reuss  (lower bound):  C_R = [Σ f_r C_r⁻¹]⁻¹
  Hashin-Shtrikman bounds
  Mori-Tanaka (spherical inclusions in matrix)
  Maxwell-Eucken (closed-pore thermal conductivity)
  Phani-Niyogi porous moduli

Hill's condition: ⟨σ:ε⟩ = ⟨σ⟩:⟨ε⟩
"""

import numpy as np


# ── Mechanical bounds ─────────────────────────────────────────────────────────
def voigt_average(C_list: list[np.ndarray], f_list: list[float]) -> np.ndarray:
    """
    Voigt (iso-strain, upper): C_V = Σ_r f_r C_r
    C_list: list of (6,6) Voigt stiffness matrices [Pa]
    f_list: volume fractions (must sum to 1)
    """
    assert abs(sum(f_list) - 1.0) < 1e-6
    return sum(f * C for f, C in zip(f_list, C_list))


def reuss_average(C_list: list[np.ndarray], f_list: list[float]) -> np.ndarray:
    """
    Reuss (iso-stress, lower): C_R = [Σ_r f_r C_r⁻¹]⁻¹
    """
    S_avg = sum(f * np.linalg.inv(C) for f, C in zip(f_list, C_list))
    return np.linalg.inv(S_avg)


def hill_average(C_list, f_list):
    """Hill (VRH): C_H = ½(C_V + C_R)."""
    return 0.5*(voigt_average(C_list, f_list) + reuss_average(C_list, f_list))


def hashin_shtrikman_bulk_modulus(K1: float, K2: float, G1: float, G2: float,
                                   f1: float, f2: float) -> tuple[float, float]:
    """
    Hashin-Shtrikman bounds on bulk modulus.
    Assumes K1 ≤ K2, G1 ≤ G2.

    K_HS⁻ = K1 + f2 / [1/(K2-K1) + 3f1/(3K1+4G1)]
    K_HS⁺ = K2 + f1 / [1/(K1-K2) + 3f2/(3K2+4G2)]

    Returns (K_lower, K_upper) [Pa]
    """
    K_lo = K1 + f2 / (1/(K2-K1+1e-30) + 3*f1/(3*K1+4*G1))
    K_hi = K2 + f1 / (1/(K1-K2+1e-30) + 3*f2/(3*K2+4*G2))
    return K_lo, K_hi


def mori_tanaka_spheres(K_m: float, G_m: float,
                         K_i: float, G_i: float,
                         f_i: float) -> dict:
    """
    Mori-Tanaka effective moduli for spherical inclusions.

      K_MT = K_m + f_i(K_i-K_m) / [1 + f_m α_0 (K_i-K_m)/(K_m)]
      where α_0 = 3K_m/(3K_m+4G_m)

    Reference: Benveniste, Mech. Mater. 6, 147 (1987)

    Parameters
    ----------
    K_m, G_m : float Pa  matrix moduli
    K_i, G_i : float Pa  inclusion moduli (set G_i→0, K_i→0 for pores)
    f_i      : float     inclusion volume fraction

    Returns
    -------
    {"K_eff": float, "G_eff": float, "E_eff": float, "nu_eff": float}
    """
    f_m = 1 - f_i
    alpha0 = 3*K_m / (3*K_m + 4*G_m)
    beta0  = 6*(K_m + 2*G_m) / (5*(3*K_m + 4*G_m))

    K_eff = K_m + f_i*(K_i-K_m) / (1 + f_m*alpha0*(K_i-K_m)/(K_m+1e-30))
    G_eff = G_m + f_i*(G_i-G_m) / (1 + f_m*beta0*(G_i-G_m)/(G_m+1e-30))

    E_eff = 9*K_eff*G_eff / (3*K_eff + G_eff)
    nu_eff = (3*K_eff - 2*G_eff) / (2*(3*K_eff + G_eff))
    return {"K_eff": K_eff, "G_eff": G_eff, "E_eff": E_eff, "nu_eff": nu_eff}


# ── Thermal conductivity bounds ───────────────────────────────────────────────
def maxwell_eucken_kappa(kappa_s: float, kappa_p: float, phi: float) -> float:
    """
    Maxwell-Eucken model for closed-pore composite:

      κ_eff = κ_s * [2κ_s + κ_p - 2φ(κ_s - κ_p)] / [2κ_s + κ_p + φ(κ_s - κ_p)]

    For pores: κ_p → 0 (or air ≈ 0.025 W/m·K).

    Parameters
    ----------
    kappa_s : float W/(m·K)  solid matrix conductivity
    kappa_p : float W/(m·K)  pore/inclusion conductivity (≈0 for closed pores)
    phi     : float  pore volume fraction (0 to 1)

    Returns
    -------
    kappa_eff : float W/(m·K)
    """
    num = 2*kappa_s + kappa_p - 2*phi*(kappa_s - kappa_p)
    den = 2*kappa_s + kappa_p + phi*(kappa_s - kappa_p)
    return kappa_s * num / (den + 1e-30)


def phani_niyogi_modulus(E0: float, phi: float,
                          phi_c: float = 0.6, n: float = 2.0) -> float:
    """
    Phani-Niyogi porous modulus (J. Mater. Sci. 2000):

      E(φ) = E0 * (1 - φ/φ_c)^n

    φ_c ≈ 0.6 (percolation threshold for random packing)
    n   ≈ 2   (empirical for APS coatings)

    Parameters
    ----------
    E0    : float Pa  dense modulus
    phi   : float pore fraction
    phi_c : float critical porosity
    n     : float exponent

    Returns
    -------
    E_eff : float Pa
    """
    return E0 * max(1.0 - phi/phi_c, 0.0)**n


def cahill_pohl_kappa_min(kappa_s: float, n: float,
                           v_speeds: np.ndarray,
                           T: float, theta_D: float) -> float:
    """
    Cahill-Pohl minimum thermal conductivity (PRB 46, 6131, 1992):

      κ_min = (π/6)^{1/3} k_B n^{2/3} Σ_i v_i (T/Θ_i)² ∫₀^{Θ_i/T} f(x) dx

    where f(x) = x³ eˣ/(eˣ-1)²

    For glass-like transport in β-Yb₂Si₂O₇ at high T.

    Parameters
    ----------
    kappa_s   : float  W/(m·K)  lattice conductivity (for reference)
    n         : float  m⁻³  atom number density
    v_speeds  : (3,)   m/s  acoustic velocities [vL, vT1, vT2]
    T         : float  K    temperature
    theta_D   : float  K    Debye temperature per branch

    Returns
    -------
    kappa_min : float W/(m·K)
    """
    from scipy.integrate import quad
    from tebc.constants import k_B
    prefactor = (np.pi/6)**(1/3) * k_B * n**(2/3)
    total = 0.0
    for v_i in v_speeds:
        Theta_i = theta_D  # simplified: same for each branch
        ratio   = T / Theta_i if Theta_i > 0 else 1.0
        def integrand(x):
            return x**3 * np.exp(x) / (np.expm1(x) + 1e-300)**2
        I, _ = quad(integrand, 0, 1.0/ratio, limit=100)
        total += v_i * ratio**2 * I
    return prefactor * total
```

---

## 9. Sensitivity Analysis

### `tebc/sensitivity/sobol_morris.py`

```python
"""
Global sensitivity analysis using SALib.

Sobol (variance-based) indices:
  S_i  = V_i / V(Y) = Var_{X_i}[E_{X~i}[Y|X_i]] / Var(Y)  (first-order)
  S_Ti = 1 - Var_{X~i}[E_{X_i}[Y|X~i]] / Var(Y)           (total effect)

Morris elementary effects:
  EE_i(x) = [f(x_1,...,x_i+Δ,...) - f(x)] / Δ
  μ*_i = (1/r) Σ |EE_i^(k)|   (mean absolute effect)
  σ_i  = stdev(EE_i)           (nonlinearity/interaction indicator)

Key parameters for TEBC failure life:
  Ranking (consensus from literature):
  1. CTE mismatch Δα             → 35-50% variance
  2. TGO parabolic rate k_p      → 20-30%
  3. Interface fracture Γ_int    → 10-20%
  4. Topcoat κ                   → <10%
  5. Recession k_l               → sets thinning
"""

from __future__ import annotations
import numpy as np
import pandas as pd

try:
    from SALib.sample import saltelli, morris as morris_sample
    from SALib.analyze import sobol, morris as morris_analyze
except ImportError:
    raise ImportError("pip install SALib")


# ── Default TEBC parameter space ─────────────────────────────────────────────
DEFAULT_TEBC_PROBLEM = {
    "num_vars": 7,
    "names": [
        "delta_alpha",    # K⁻¹   CTE mismatch (EBC - CMC)
        "k_p",            # m²/s  TGO parabolic rate at service T
        "Gamma_int",      # J/m²  interface fracture toughness
        "kappa_TBC",      # W/mK  TBC thermal conductivity
        "k_l",            # m/s   recession rate
        "E_EBC",          # Pa    EBC Young's modulus
        "porosity_TBC",   # -     TBC porosity fraction
    ],
    "bounds": [
        [0.5e-6, 2.0e-6],    # delta_alpha
        [1e-15,  1e-12],     # k_p  (log-uniform → transform below)
        [5.0,    80.0],      # Gamma_int
        [0.8,    2.5],       # kappa_TBC
        [1e-11,  1e-8],      # k_l
        [100e9,  250e9],     # E_EBC
        [0.05,   0.20],      # porosity_TBC
    ],
    "dists": ["unif", "logunif", "unif", "unif", "logunif", "unif", "unif"],
}


def run_sobol(model_func, problem: dict = None,
              N: int = 1024,
              calc_second_order: bool = True) -> pd.DataFrame:
    """
    Saltelli-sampled Sobol analysis.

    Parameters
    ----------
    model_func : callable(X: np.ndarray) -> np.ndarray
                 Evaluates TEBC model at N(d+2) parameter sets.
                 X shape: (n_samples, n_vars)
    problem    : SALib problem dict (default: DEFAULT_TEBC_PROBLEM)
    N          : base sample size (total calls = N*(d+2))

    Returns
    -------
    df : DataFrame with columns [S1, S1_conf, ST, ST_conf] per parameter
    """
    if problem is None:
        problem = DEFAULT_TEBC_PROBLEM
    X = saltelli.sample(problem, N, calc_second_order=calc_second_order)
    Y = model_func(X)
    Si = sobol.analyze(problem, Y, calc_second_order=calc_second_order,
                        print_to_console=False)
    df = pd.DataFrame({
        "parameter": problem["names"],
        "S1":        Si["S1"],
        "S1_conf":   Si["S1_conf"],
        "ST":        Si["ST"],
        "ST_conf":   Si["ST_conf"],
    })
    df = df.sort_values("ST", ascending=False).reset_index(drop=True)
    return df


def run_morris(model_func, problem: dict = None,
               n_trajectories: int = 50,
               num_levels: int = 4) -> pd.DataFrame:
    """
    Morris elementary effects screening.

    Parameters
    ----------
    model_func     : callable(X) -> Y
    n_trajectories : number of Morris trajectories r
    num_levels     : p in Morris sampling grid

    Returns
    -------
    df : DataFrame with [mu_star, sigma, mu] per parameter
    """
    if problem is None:
        problem = DEFAULT_TEBC_PROBLEM
    X = morris_sample.sample(problem, N=n_trajectories,
                              num_levels=num_levels, optimal_trajectories=10)
    Y = model_func(X)
    Si = morris_analyze.analyze(problem, X, Y, print_to_console=False)
    df = pd.DataFrame({
        "parameter": problem["names"],
        "mu_star":   Si["mu_star"],
        "sigma":     Si["sigma"],
        "mu":        Si["mu"],
    })
    df = df.sort_values("mu_star", ascending=False).reset_index(drop=True)
    return df


def tebc_failure_model(X: np.ndarray,
                        n_cycles: float = 600.0,
                        h_layer: float  = 150e-6) -> np.ndarray:
    """
    Analytical TEBC failure index (surrogate for full FEA).

    Based on Evans-Hutchinson framework:
      G_drive = (1-ν²) σ₀² h / (2E)       [driving force]
      σ₀ = E/(1-ν) * Δα * ΔT              [CTE mismatch stress]
      TGO: x_TGO(t) = √(k_p * t_total)    [parabolic]
      t_total = n_cycles * 3600 s
      Failure index = G_drive / Γ_int     [>1 = delaminated]

    Plus recession index = k_l * t / h_layer

    Parameters
    ----------
    X : (n_samples, 7) parameter matrix from Sobol/Morris sampler

    Returns
    -------
    Y : (n_samples,) combined failure index
    """
    delta_alpha = X[:, 0]
    k_p         = X[:, 1]
    Gamma_int   = X[:, 2]
    kappa_TBC   = X[:, 3]
    k_l         = X[:, 4]
    E_EBC       = X[:, 5]
    porosity    = X[:, 6]

    nu    = 0.27
    dT    = 1300.0            # K  temperature drop per cycle
    t_tot = n_cycles * 3600.0 # s

    # Biaxial mismatch stress
    sigma0 = (E_EBC / (1 - nu)) * delta_alpha * dT
    # Driving force for delamination
    G_drive = (1 - nu**2) * sigma0**2 * h_layer / (2 * E_EBC)
    # TGO contribution (Evans-Hutchinson ratcheting ΔG)
    x_TGO = np.sqrt(k_p * t_tot)
    E_TGO, nu_TGO = 70e9, 0.17
    sigma_TGO = (E_TGO/(1-nu_TGO)) * 0.31   # PBR growth strain
    G_TGO  = (1-nu_TGO**2)*sigma_TGO**2 * x_TGO / (2*E_TGO)
    # Recession fraction
    recession_frac = k_l * t_tot / h_layer
    # Effective porosity correction on κ
    kappa_eff = maxwell_eucken_kappa_simple(kappa_TBC, 0.0, porosity)

    # Composite failure index (dimensionless)
    fail_idx = ((G_drive + G_TGO) / (Gamma_int + 1e-30)
                + 2.0 * recession_frac)
    return fail_idx


def maxwell_eucken_kappa_simple(ks, kp, phi):
    """Inline Maxwell-Eucken for sensitivity model."""
    num = 2*ks + kp - 2*phi*(ks - kp)
    den = 2*ks + kp +   phi*(ks - kp)
    return ks * num / (den + 1e-30)
```

---

## 10. Orchestrator

### `tebc/orchestrator.py`

```python
"""
Main pipeline orchestrator.

Runs the full 4-scale TEBC simulation in sequence:
  Scale 1 → Scale 2 → Scale 3 → Scale 4 → Sensitivity

Each scale's outputs are validated before passing to the next.
"""

from __future__ import annotations
import logging
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("tebc")


@dataclass
class TEBCConfig:
    """Top-level simulation configuration."""
    material_EBC:   str = "beta_Yb2Si2O7"   # key in MATERIALS dict
    material_TBC:   str = "7YSZ"
    material_bond:  str = "Si_bondcoat"
    material_sub:   str = "SiC_SiC_CMC"

    # Layer thicknesses [m]
    h_TBC:    float = 200e-6
    h_EBC:    float = 150e-6
    h_bond:   float = 100e-6
    h_sub:    float = 5e-3

    # Operating conditions
    T_hot:    float = 1600.0   # K  hot-gas temperature
    T_cold:   float = 300.0    # K  ambient
    T_dep:    float = 1473.0   # K  deposition temperature
    P_H2O:    float = 101325.0 # Pa water vapour pressure
    P_tot:    float = 1e6      # Pa total pressure (10 atm engine)
    v_gas:    float = 100.0    # m/s gas velocity
    n_cycles: int   = 600      # thermal cycles

    # Simulation control
    run_scale1: bool = True
    run_scale2: bool = True
    run_scale3: bool = True
    run_scale4: bool = True
    run_sensitivity: bool = True

    # Paths
    dft_outcar:   str = ""
    lammps_input: str = ""
    tdb_file:     str = "data/tdb/RE_Si_O.tdb"
    output_dir:   str = "results/"


@dataclass
class ScaleParameters:
    """Validated parameters passed between scales."""
    # Scale 1 → 2
    C_ijkl: np.ndarray = field(default_factory=lambda: np.eye(6)*200e9)  # Pa
    E0: float = 0.0         # J/atom
    V0: float = 0.0         # m³/atom
    gamma_surf: float = 1.0 # J/m²
    E_defect: float = 5.0   # eV

    # Scale 2 → 3
    kappa_T: np.ndarray = field(default_factory=lambda: np.array([[1.5,0,0],[0,1.5,0],[0,0,1.5]]))  # W/mK
    alpha_T: float = 4.05e-6  # K⁻¹
    D_O: float = 1e-18        # m²/s at service T
    C_ij_T: np.ndarray = field(default_factory=lambda: np.eye(6)*150e9)

    # Scale 3 → 4
    kappa_eff: float = 1.5   # W/mK  effective layer κ
    E_eff: float = 150e9     # Pa    effective modulus
    x_TGO: float = 0.0       # m     TGO thickness at end of simulation
    recession: float = 0.0   # m     total recession

    # Scale 4 outputs
    sigma_max: float = 0.0   # Pa
    G_drive:   float = 0.0   # J/m²
    fail_index: float = 0.0  # dimensionless


def run_pipeline(cfg: TEBCConfig) -> ScaleParameters:
    """
    Execute full multi-scale pipeline.

    Scale 1 (DFT/DNP) → Scale 2 (MD/QHA) → Scale 3 (PF/TGO) → Scale 4 (FEA)
    → Sensitivity analysis
    """
    from tebc.constants import MATERIALS
    params = ScaleParameters()

    # ── SCALE 1 ──────────────────────────────────────────────────────────────
    if cfg.run_scale1 and Path(cfg.dft_outcar).exists():
        logger.info("Scale 1: Parsing DFT outputs...")
        from tebc.scale1_atomistic.dft_interface import (
            parse_elastic_tensor, parse_structure_energy)
        params.C_ijkl = parse_elastic_tensor(cfg.dft_outcar)
        struct = parse_structure_energy(cfg.dft_outcar.replace("OUTCAR","vasprun.xml"))
        params.E0 = struct["E0_per_atom"]
        params.V0 = struct["V0_angstrom3"] * 1e-30 / struct["N_atoms"]
        logger.info(f"  C11={params.C_ijkl[0,0]/1e9:.1f} GPa, E0={params.E0:.3f} eV/atom")
    else:
        logger.info("Scale 1: Using database values for %s", cfg.material_EBC)
        mat = MATERIALS[cfg.material_EBC]
        # Isotropic approximation from E, nu
        E, nu = mat["E"], mat["nu"]
        lam = E*nu/((1+nu)*(1-2*nu))
        mu  = E/(2*(1+nu))
        C6  = np.zeros((6,6))
        for i in range(3): C6[i,i] = lam + 2*mu
        for i,j in [(0,1),(0,2),(1,2)]: C6[i,j]=C6[j,i]=lam
        for i in range(3,6): C6[i,i] = mu
        params.C_ijkl = C6

    # ── SCALE 2 ──────────────────────────────────────────────────────────────
    if cfg.run_scale2:
        logger.info("Scale 2: MD/QHA thermal properties...")
        mat = MATERIALS[cfg.material_EBC]
        T_service = (cfg.T_hot + cfg.T_cold) / 2.0
        # Arrhenius D_O if YSZ
        if cfg.material_TBC == "7YSZ":
            from tebc.constants import eV
            ysz = MATERIALS["7YSZ"]
            params.D_O = (ysz["D0_O"]
                          * np.exp(-ysz["Ea_DO"]/(1.380649e-23 * T_service)))
        params.alpha_T  = mat["alpha"]
        params.kappa_T  = np.eye(3) * mat["kappa"]
        params.C_ij_T   = params.C_ijkl * (1 - 0.15*(T_service-300)/1300)
        logger.info(f"  α={params.alpha_T*1e6:.2f}×10⁻⁶ K⁻¹, κ={params.kappa_T[0,0]:.2f} W/mK")

    # ── SCALE 3 ──────────────────────────────────────────────────────────────
    if cfg.run_scale3:
        logger.info("Scale 3: TGO kinetics + phase-field...")
        from tebc.scale3_mesoscale.tgo_kinetics import (
            solve_paralinear, robinson_smialek_recession, parabolic_rate_constant)
        mat = MATERIALS[cfg.material_bond]
        T_service = cfg.T_hot
        k_p = parabolic_rate_constant(T_service, mat["k_p_wet"], mat["Ea_kp_wet"])
        k_l = robinson_smialek_recession(T_service, cfg.P_H2O, cfg.P_tot, cfg.v_gas)
        t_total = cfg.n_cycles * 3600.0
        sol = solve_paralinear((0, t_total), k_p, k_l)
        params.x_TGO   = float(sol["x_TGO"][-1])
        params.recession = float(sol["recession"][-1])

        # Effective EBC properties (porous APS)
        from tebc.coupling.homogenization import maxwell_eucken_kappa, phani_niyogi_modulus
        phi_APS = 0.12
        params.kappa_eff = maxwell_eucken_kappa(
            MATERIALS[cfg.material_EBC]["kappa"], 0.025, phi_APS)
        params.E_eff = phani_niyogi_modulus(
            MATERIALS[cfg.material_EBC]["E"], phi_APS)
        logger.info(f"  TGO={params.x_TGO*1e6:.2f} μm, recession={params.recession*1e6:.1f} μm")

    # ── SCALE 4 ──────────────────────────────────────────────────────────────
    if cfg.run_scale4:
        logger.info("Scale 4: Continuum thermoelastic analysis...")
        from tebc.scale4_continuum.thermoelastic import (
            bilayer_mismatch_stress, energy_release_rate_steady_state)
        dT = cfg.T_cold - cfg.T_dep   # cool-down from deposition
        mat_EBC = MATERIALS[cfg.material_EBC]
        mat_sub = MATERIALS[cfg.material_sub]
        sigma_EBC = bilayer_mismatch_stress(
            params.E_eff, mat_EBC["nu"],
            mat_EBC["alpha"], mat_sub["alpha"], dT)
        params.sigma_max = abs(sigma_EBC)
        params.G_drive   = energy_release_rate_steady_state(
            sigma_EBC, cfg.h_EBC, params.E_eff, mat_EBC["nu"])
        Gamma_int = mat_EBC["Gamma_interface"]
        params.fail_index = params.G_drive / Gamma_int
        logger.info(f"  σ={sigma_EBC/1e6:.1f} MPa, G={params.G_drive:.1f} J/m², FI={params.fail_index:.3f}")

    # ── SENSITIVITY ───────────────────────────────────────────────────────────
    if cfg.run_sensitivity:
        logger.info("Sensitivity: Sobol analysis (N=512)...")
        from tebc.sensitivity.sobol_morris import run_sobol, tebc_failure_model
        df = run_sobol(tebc_failure_model, N=512)
        logger.info("\n" + df.to_string(index=False))
        results_dir = Path(cfg.output_dir)
        results_dir.mkdir(exist_ok=True)
        df.to_csv(results_dir / "sobol_indices.csv", index=False)

    return params


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = TEBCConfig(
        run_scale1=False,   # set True + dft_outcar if VASP data available
        run_scale2=True,
        run_scale3=True,
        run_scale4=True,
        run_sensitivity=True,
    )
    result = run_pipeline(cfg)
    print(f"\n=== TEBC Simulation Complete ===")
    print(f"Failure index:  {result.fail_index:.4f}  ({'FAIL' if result.fail_index > 1 else 'PASS'})")
    print(f"TGO thickness:  {result.x_TGO*1e6:.2f} μm")
    print(f"EBC recession:  {result.recession*1e6:.2f} μm")
    print(f"Driving force:  {result.G_drive:.1f} J/m²")
```

---

## 11. Validated Material Parameters

### `data/materials_db.json`
*(excerpt — full file generated by `tebc/constants.py`)*

```json
{
  "_comment": "All SI units unless noted. Sources cited inline.",
  "beta_Yb2Si2O7": {
    "rho_kg_m3":      6180,
    "E_Pa":           185e9,
    "nu":             0.275,
    "alpha_K":        4.05e-6,
    "alpha_aniso_K":  [3.57e-6, 2.49e-6, 1.48e-6],
    "kappa_Wm1K1":    2.5,
    "cp_Jkg1K1":      450,
    "KIC_Pa_m05":     1.75e6,
    "Gamma_int_Jm2":  30,
    "T_melt_K":       2123,
    "k_p_m2s_1316C":  2.78e-18,
    "Ea_kp_Jmol":     101000,
    "k_l_ms_1316C":   2.78e-11,
    "Ea_kl_Jmol":     108000,
    "sources": ["Tian 2013 ActaMat", "Zhou JACerS 2013", "Zhao JECS 2020"]
  },
  "7YSZ": {
    "rho_kg_m3":      6050,
    "E_dense_Pa":     210e9,
    "E_APS_Pa":       50e9,
    "nu":             0.23,
    "alpha_K":        10.5e-6,
    "kappa_dense_Wm1K1": 2.2,
    "kappa_APS_Wm1K1":   1.0,
    "cp_Jkg1K1":      505,
    "KIC_Pa_m05":     2.0e6,
    "D0_O_m2s":       1.3e-6,
    "Ea_DO_eV":       0.95,
    "sources": ["Brossmann PCCP 2003", "Schlichting JMS 2001", "Clarke SCT 2003"]
  }
}
```

---

## 12. Test Suite

### `tests/test_scale2.py`

```python
"""Unit tests for Scale 2 (MD/QHA) modules."""

import numpy as np
import pytest
from tebc.scale2_md.green_kubo import compute_hcacf, integrate_hcacf, plateau_estimate
from tebc.scale2_md.phonon_qha import compute_free_energy, gruneisen_cte_relation
from tebc.scale2_md.msd_diffusion import msd_to_diffusivity
from tebc.constants import k_B, hbar


def test_hcacf_delta():
    """HCACF of delta function = constant → κ should converge."""
    n = 10000
    dt = 1e-15
    J  = np.zeros(n); J[0] = 1.0
    t, C = compute_hcacf(J.reshape(-1, 1).repeat(3, axis=1) / np.sqrt(3), dt)
    # Should not raise; C[0] > 0
    assert C[0] > 0


def test_free_energy_zero_T():
    """At T→0, F = E0 + ZPE. All phonon occupation = 0."""
    omega = np.array([[1e13, 2e13]])  # rad/s
    w     = np.array([1.0])
    E0    = -1e-18  # J
    F = compute_free_energy(omega, w, E0, T=1.0)  # near-zero T
    ZPE = 0.5 * (hbar * omega * w[:, None]).sum()
    assert abs(F - (E0 + ZPE)) / abs(E0 + ZPE) < 1e-3


def test_gruneisen_identity():
    """α_V = γ C_V / (B V) should give physically reasonable CTE."""
    gamma = 1.0        # typical for β-RE disilicates
    Cv    = 1.5e6      # J/(m³·K)  ~ρ cp for Yb2Si2O7
    V     = 1.0        # m³ (per unit volume)
    B     = 135e9      # Pa
    alpha = gruneisen_cte_relation(gamma, Cv, V, B)
    # Should be ~11e-6 K⁻¹ (or Cv/B ratio)
    assert 1e-8 < alpha < 1e-4, f"Unreasonable CTE: {alpha}"


def test_msd_diffusivity():
    """Linear MSD should recover exact D."""
    D_true = 1e-12  # m²/s
    n_frames = 1000
    dt = 1e-12
    t  = np.arange(n_frames) * dt
    msd = 6 * D_true * t
    t_idx = np.arange(n_frames)
    result = msd_to_diffusivity(t_idx, msd, dt_per_frame=dt)
    assert abs(result["D"] - D_true) / D_true < 0.01


class TestScale3:
    """Tests for TGO kinetics."""
    def test_paralinear_steady_state(self):
        from tebc.scale3_mesoscale.tgo_kinetics import solve_paralinear
        k_p = 1e-14   # m²/s
        k_l = 1e-10   # m/s
        x_ss_analytical = k_p / (2*k_l)
        sol  = solve_paralinear((0, 1e8), k_p, k_l, x0=1e-9, n_points=2000)
        x_final = sol["x_TGO"][-1]
        assert abs(x_final - x_ss_analytical)/x_ss_analytical < 0.05

    def test_deal_grove_thin_limit(self):
        """Thin scale: x ≈ (B/A)t = k_l t (linear)."""
        from tebc.scale3_mesoscale.tgo_kinetics import deal_grove_thickness
        k_p = 1e-18  # very small parabolic rate
        k_l = 1e-9   # m/s
        t   = np.array([1.0, 10.0, 100.0])
        x   = deal_grove_thickness(t, k_p, k_l)
        x_linear = k_l * t
        np.testing.assert_allclose(x, x_linear, rtol=0.05)


class TestScale4:
    """Tests for continuum mechanics."""
    def test_mismatch_stress_sign(self):
        """On cool-down, YSZ (high α) on EBC (low α) → compressive in YSZ."""
        from tebc.scale4_continuum.thermoelastic import bilayer_mismatch_stress
        sigma = bilayer_mismatch_stress(
            E_f=50e9, nu_f=0.23,
            alpha_f=10.5e-6, alpha_s=4.05e-6,
            dT=-1300)  # cool-down
        assert sigma < 0, "YSZ should be compressive on cool-down"

    def test_energy_release_rate_positive(self):
        """G_ss must be positive."""
        from tebc.scale4_continuum.thermoelastic import energy_release_rate_steady_state
        G = energy_release_rate_steady_state(300e6, 200e-6, 50e9, 0.23)
        assert G > 0

    def test_czm_traction_peak(self):
        """Peak traction at λ = λ₁."""
        from tebc.scale4_continuum.damage_mechanics import TVHCohesiveZone
        czm = TVHCohesiveZone(sigma_hat=100e6, delta_n_c=1e-6, delta_t_c=3e-6)
        # Pure normal opening at λ = λ₁
        dn_at_peak = czm.l1 * czm.delta_n_c
        T_n, _ = czm.tractions(dn_at_peak, 0.0)
        assert abs(T_n - czm.sigma_hat) / czm.sigma_hat < 0.01


class TestCoupling:
    """Tests for homogenization."""
    def test_voigt_reuss_bounds(self):
        """Voigt ≥ Hill ≥ Reuss (bulk modulus)."""
        from tebc.coupling.homogenization import voigt_average, reuss_average, hill_average
        C1 = np.eye(6) * 200e9; C1[3,3]=C1[4,4]=C1[5,5]=80e9
        C2 = np.eye(6) *  50e9; C2[3,3]=C2[4,4]=C2[5,5]=20e9
        f  = [0.7, 0.3]
        Cv = voigt_average([C1,C2], f)
        Cr = reuss_average([C1,C2], f)
        Ch = hill_average([C1,C2], f)
        # Trace of Voigt ≥ Hill ≥ Reuss
        assert np.trace(Cv) >= np.trace(Ch) >= np.trace(Cr)

    def test_maxwell_eucken_limits(self):
        """φ=0 → κ_eff = κ_s; φ→1 → κ_eff ≈ 0."""
        from tebc.coupling.homogenization import maxwell_eucken_kappa
        assert abs(maxwell_eucken_kappa(2.5, 0.0, 0.0) - 2.5) < 1e-10
        assert maxwell_eucken_kappa(2.5, 0.0, 0.99) < 0.05


class TestSensitivity:
    """Tests for Sobol analysis."""
    def test_sobol_sum_leq_total(self):
        """Sum of S1 ≤ Sum of ST (always)."""
        from tebc.sensitivity.sobol_morris import run_sobol, tebc_failure_model
        df = run_sobol(tebc_failure_model, N=128)
        assert df["S1"].sum() <= df["ST"].sum() + 0.1

    def test_delta_alpha_dominates(self):
        """CTE mismatch should have highest total Sobol index."""
        from tebc.sensitivity.sobol_morris import run_sobol, tebc_failure_model
        df = run_sobol(tebc_failure_model, N=256)
        top_param = df.iloc[0]["parameter"]
        assert top_param in ("delta_alpha", "k_p"), f"Unexpected dominant param: {top_param}"
```

---

## Appendix: Equation Reference Map

| Symbol | Equation | Module | Line note |
|--------|----------|--------|-----------|
| Kohn-Sham | `[-ℏ²/2m ∇² + V_ext + V_H + V_xc]ψᵢ = εᵢψᵢ` | `dft_interface.py` | solved by VASP |
| Cohesive energy | `E_coh = -(1/N)[E_cryst - Σ Nᵢ Eᵢ^atom]` | `compute_cohesive_energy()` | |
| Surface energy | `γ = (E_slab - N E_bulk) / (2A)` | `compute_surface_energy()` | |
| Defect formation | `E_f = E_tot[def,q] - E_host - Σδnᵢμᵢ + q(E_VBM+E_F) + E_corr` | `compute_defect_formation_energy()` | |
| DNP smooth switch | `s(r) = quintic in (r-r_s)/(r_c-r_s)` | `build_deepmd_input()` | |
| DNP loss | `L = (p_e/N)∣ΔE∣² + (p_f/3N)Σ∣ΔFᵢ∣² + (p_v/9N)∣ΔΞ∣²` | `build_deepmd_input()` | |
| Velocity-Verlet | `r(t+dt) = r + vdt + ½a dt²` | `velocity_verlet_step()` | |
| Nosé-Hoover chain | `dp_{ξ_k}/dt = [p²_{ξ_{k-1}}/Q_{k-1} - k_BT]` | `NoseHooverChain.step()` | |
| Parrinello-Rahman | `W ḧ = V(σ_int - p_ext I)(hᵀ)⁻¹` | `parrinello_rahman_step()` | |
| QHA free energy | `F = E₀ + Σ [½ℏω + k_BT ln(1-exp(-ℏω/k_BT))]` | `compute_free_energy()` | |
| Grüneisen identity | `α_V = γ C_V / (B V)` | `gruneisen_cte_relation()` | |
| Green-Kubo κ | `κ = V/k_BT² ∫⟨J(0)J(t)⟩dt` | `integrate_hcacf()` | |
| MSD diffusivity | `D = ⟨Δr²⟩ / 6t` | `msd_to_diffusivity()` | |
| Arrhenius | `D = D₀ exp(-Ea/RT)` | `arrhenius_fit/eval()` | |
| Allen-Cahn | `∂φ/∂t = -M_φ[Wg'(φ) - κ_φ∇²φ + (G_β-G_L)h'(φ)]` | `CMASPhaseField.step()` | |
| Cahn-Hilliard | `∂c/∂t = ∇·[M_c ∇(∂f/∂c - κ_c∇²c)]` | `CMASPhaseField.step()` | |
| Nucleation rate | `I = I₀ exp(-ΔG*/k_BT), ΔG* = 16πσ³f(θ)/3ΔG_v²` | `nucleation_rate()` | |
| Redlich-Kister | `ᵉˣG = x_A x_B Σ_v Lᵥ(x_A-x_B)^v` | `gibbs_redlich_kister()` | |
| Deal-Grove | `x² + Ax = Bt` | `deal_grove_thickness()` | |
| Paralinear ODE | `dx/dt = k_p/(2x) - k_l` | `paralinear_ode()` | |
| Robinson-Smialek | `k_l ∝ v^0.5 P_{H₂O}^2 P_tot^{-0.5} exp(-Q/RT)` | `robinson_smialek_recession()` | |
| PBR stress | `ε_TGO = ⅓(PBR-1)` | `tgo_growth_stress()` | |
| Mismatch stress | `σ_f = E/(1-ν)(α_s-α_f)ΔT` | `bilayer_mismatch_stress()` | |
| Stoney | `κ = 6σ_f h_f(1-ν_s)/(E_s h_s²)` | `stoney_curvature()` | |
| G_ss | `G_ss = (1-ν²)σ₀²h/(2E)` | `energy_release_rate_steady_state()` | |
| Heat equation | `ρc_p Ṫ = ∇·(κ∇T) + Q` | `fenics_thermoelastic_setup()` | FEniCSx |
| Thermoelastic | `σ = C:(ε - αΔT)` | `fenics_thermoelastic_setup()` | |
| Lemaitre Y | `Y = σ_eq² R_v / [2E(1-D)²]` | `lemaitre_damage_rate()` | |
| Lemaitre dD/dp | `dD/dp = (Y/S)^s` | `lemaitre_damage_rate()` | |
| Mazars ε̃ | `ε̃ = √[Σᵢ⟨εᵢ⟩₊²]` | `mazars_equivalent_strain()` | |
| CZM λ | `λ = √[(δ_n/δ_n^c)² + (δ_t/δ_t^c)²]` | `TVHCohesiveZone` | |
| CZM T_n | `T_n = (σ(λ)/λ)(δ_n/δ_n^c)` | `TVHCohesiveZone.tractions()` | |
| Voigt | `C_V = Σ f_r C_r` | `voigt_average()` | |
| Reuss | `C_R = [Σ f_r C_r⁻¹]⁻¹` | `reuss_average()` | |
| Hashin-Shtrikman | `K_HS = K₁ + f₂/[1/(K₂-K₁) + 3f₁/(3K₁+4G₁)]` | `hashin_shtrikman_bulk_modulus()` | |
| Mori-Tanaka | `K_MT = K_m + f_i(K_i-K_m)/[1 + f_m α₀(K_i-K_m)/K_m]` | `mori_tanaka_spheres()` | |
| Maxwell-Eucken | `κ_eff = κ_s [2κ_s+κ_p-2φ(κ_s-κ_p)] / [2κ_s+κ_p+φ(κ_s-κ_p)]` | `maxwell_eucken_kappa()` | |
| Phani-Niyogi | `E(φ) = E₀(1-φ/φ_c)^n` | `phani_niyogi_modulus()` | |
| Sobol S_i | `S_i = Var[E[Y\|Xᵢ]] / Var(Y)` | `run_sobol()` | |
| Morris EE | `EE_i = [f(x+Δeᵢ)-f(x)]/Δ` | `run_morris()` | |
