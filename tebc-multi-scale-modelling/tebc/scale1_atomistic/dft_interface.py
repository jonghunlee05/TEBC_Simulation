"""
Parse DFT results (VASP format) and extract:
  - Elastic stiffness tensor C_ijkl  (GPa)
  - Cohesive energy E0               (eV/atom)
  - Equilibrium volume V0            (Å³)
  - Anisotropic CTE tensor alpha_ij  (K⁻¹) via QHA
  - Surface energy gamma_surf        (J/m²)
  - Defect formation energy E_def    (eV)
"""

from __future__ import annotations

import numpy as np

try:
    from pymatgen.analysis.elasticity import ElasticTensor
    from pymatgen.io.vasp.outputs import Outcar, Vasprun
except ImportError:
    raise ImportError("pip install pymatgen")


def parse_elastic_tensor(outcar_path) -> np.ndarray:
    """
    Parse VASP OUTCAR → 6×6 elastic stiffness matrix in **Pa**.

    VASP writes C_ij in kBar (1 kBar = 1e8 Pa); we convert at the array
    level so the rest of the pipeline can use SI throughout. Kohn–Sham
    equations are solved self-consistently → stress–strain response via
    finite distortions: C_ij = (1/V0) ∂²E/∂ε_i∂ε_j.

    Note: older versions of pymatgen returned `outcar.elastic_tensor`
    already in GPa. If you see stiffnesses ~10× too large, your pymatgen
    is doing the conversion for you — drop the `* 1e8` scaling here and
    use `* 1e9` (GPa→Pa) instead.
    """
    outcar = Outcar(str(outcar_path))
    C_kBar = np.array(outcar.elastic_tensor)
    C_Pa = C_kBar * 1.0e8                                    # kBar → Pa
    et = ElasticTensor.from_voigt(C_Pa)
    return et.voigt


def parse_structure_energy(vasprun_path) -> dict:
    """Parse vasprun.xml → E0 (eV/atom), V0 (Å³), forces (eV/Å)."""
    vr = Vasprun(str(vasprun_path))
    struct = vr.final_structure
    N = len(struct)
    return {
        "E0_per_atom": vr.final_energy / N,
        "E0_total":    vr.final_energy,
        "V0_angstrom3": struct.volume,
        "N_atoms": N,
        "lattice": struct.lattice.matrix,
        "forces":  np.array(vr.ionic_steps[-1]["forces"]),
    }


def compute_cohesive_energy(E_crystal_eV: float, N: int,
                             E_atoms_eV: dict,
                             composition: dict) -> float:
    """E_coh = -(1/N)[E_crystal - Σ_i N_i * E_i^atom]"""
    E_ref = sum(composition[s] * E_atoms_eV[s] for s in composition)
    return -(E_crystal_eV - E_ref) / N


def compute_surface_energy(E_slab: float, E_bulk_per_atom: float,
                            N_slab: int, A_surface_m2: float) -> float:
    """γ_surf = (E_slab - N_slab * E_bulk/atom) / (2 * A) [J/m²]"""
    from tebc.constants import eV
    delta_E = (E_slab - N_slab * E_bulk_per_atom) * eV
    return delta_E / (2.0 * A_surface_m2)


def compute_defect_formation_energy(E_defect: float, E_host: float,
                                     mu: dict,
                                     delta_n: dict,
                                     q: int, E_VBM: float,
                                     E_Fermi: float,
                                     E_corr: float = 0.0) -> float:
    """
    Standard defect formation energy (Freysoldt, RMP 2014):
      E_f[X^q] = E_tot[defect,q] - E_tot[host]
                 - Σ_i δn_i μ_i + q(E_VBM + E_Fermi) + E_corr
    """
    chem_term = sum(delta_n.get(s, 0) * mu[s] for s in mu)
    return (E_defect - E_host - chem_term
            + q * (E_VBM + E_Fermi) + E_corr)


def extract_born_effective_charges(outcar_path) -> np.ndarray:
    """Parse Born effective charges Z*_{I,αβ} from VASP OUTCAR."""
    outcar = Outcar(str(outcar_path))
    return np.array(outcar.born)
