# TEBC Multi-Scale Simulation

A 4-scale Python framework for simulating **rare-earth silicate Thermal/Environmental Barrier Coatings (T/EBCs)** on SiC/SiC ceramic matrix composites — from first-principles electronic structure all the way to component-level finite-element analysis.

The pipeline targets coating systems such as **β-Yb₂Si₂O₇ / β-Y₂Si₂O₇ EBCs** with **7YSZ TBC**, **Si bond coat**, and a **SiC/SiC CMC substrate** under aero-engine combustor conditions.

---

## The four scales

| Scale | Length / time | Physics | Key modules |
|------|---------------|---------|-------------|
| **1 — Atomistic** | Å, fs | DFT (VASP) → DeePMD neural-network potentials trained via DP-GEN active learning | [`tebc/scale1_atomistic/`](tebc/scale1_atomistic/) |
| **2 — Molecular Dynamics** | nm, ps–ns | LAMMPS NVT/NVE/NPT ensembles, Phonopy/Phono3py QHA, Green-Kubo κ(T), MSD → D_O Arrhenius fits | [`tebc/scale2_md/`](tebc/scale2_md/) |
| **3 — Mesoscale** | μm, hours | Allen-Cahn/Cahn-Hilliard phase field (FiPy), pycalphad equilibria, Deal-Grove + paralinear TGO kinetics, CMAS dissolution | [`tebc/scale3_mesoscale/`](tebc/scale3_mesoscale/) |
| **4 — Continuum FEA** | mm, thousands of cycles | FEniCSx coupled thermo-elasticity, Lemaitre + Mazars damage, Tvergaard-Hutchinson cohesive zones, Robinson-Smialek recession | [`tebc/scale4_continuum/`](tebc/scale4_continuum/) |

Information flows upward: elastic constants Cᵢⱼₖₗ, thermal expansion αᵢⱼ and surface energies from Scale 1 feed MD potentials at Scale 2, which produce κ(T), α(T), D_O(T), Cᵢⱼ(T) for Scale 3, where homogenized κ_eff, E_eff and TGO thickness(t) become inputs to the Scale 4 component model.

---

## Cross-scale coupling and UQ

- **[`tebc/coupling/homogenization.py`](tebc/coupling/homogenization.py)** — Voigt, Reuss, Hashin-Shtrikman bounds, Mori-Tanaka, and self-consistent schemes for effective properties of porous / multi-phase microstructures.
- **[`tebc/sensitivity/sobol_morris.py`](tebc/sensitivity/sobol_morris.py)** — SALib-based Sobol indices and Morris elementary effects for global sensitivity over the full parameter space (results in [`results/sobol_indices.csv`](results/sobol_indices.csv)).
- **[`tebc/orchestrator.py`](tebc/orchestrator.py)** — `TEBCConfig` + pipeline runner that chains all four scales with validated parameter handoff.

---

## Material database

[`data/materials_db.json`](data/materials_db.json) holds literature-validated parameters (with citations) for:

- **β-Yb₂Si₂O₇** and **β-Y₂Si₂O₇** EBC materials — anisotropic CTE, κ, K_IC, parabolic/linear oxidation rates
- **7YSZ** topcoat — dense and APS-sprayed properties, oxygen diffusivity
- **Si bond coat**
- **SiC/SiC CMC** substrate

All values are SI units; sources include Tian 2013 ActaMat, Zhou JACerS 2013, Zhao JECS 2020, Brossmann PCCP 2003, etc.

---

## Layout

```
tebc-multi-scale-modelling/
├── tebc/                          # Main package
│   ├── constants.py
│   ├── utils.py
│   ├── orchestrator.py            # TEBCConfig + pipeline runner
│   ├── scale1_atomistic/          # DFT parsing, DeePMD training
│   ├── scale2_md/                 # LAMMPS, phonons/QHA, Green-Kubo, MSD
│   ├── scale3_mesoscale/          # phase field, CALPHAD, TGO kinetics
│   ├── scale4_continuum/          # FEniCSx thermoelasticity, damage
│   ├── coupling/                  # homogenization schemes
│   └── sensitivity/               # Sobol / Morris analysis
├── data/
│   ├── materials_db.json          # validated material parameters
│   └── tdb/                       # CALPHAD .tdb files
├── results/                       # generated outputs
├── tests/                         # pytest suite with physics-grounded checks
├── TEBC_implementation_spec.md    # full implementation specification
├── pyproject.toml
└── environment.yml
```

---

## Installation

```bash
conda env create -f environment.yml
conda activate tebc
pip install -e .
```

Optional extras:
```bash
pip install -e ".[fea]"   # FEniCSx + PETSc + MPI for Scale 4
pip install -e ".[ml]"    # DeePMD-kit + DP-GEN for Scale 1
```

Requires Python ≥ 3.11. Core dependencies: `numpy`, `scipy`, `ase`, `pymatgen`, `phonopy`, `phono3py`, `fipy`, `pycalphad`, `salib`.

---

## Running

Minimal example:
```python
from tebc.orchestrator import TEBCConfig, run_pipeline

cfg = TEBCConfig(
    material_EBC="beta_Yb2Si2O7",
    material_TBC="7YSZ",
    T_hot=1600.0,    # K
    n_cycles=600,
)
run_pipeline(cfg)
```

Default operating conditions reflect an aero-engine combustor within the Robinson-Smialek calibration window (~1316 °C, P_H₂O ≈ 0.1 atm, v_gas ≈ 4.4 cm/s).

---

## Tests

```bash
pytest tests/
```

Tests assert physical reasonableness of MD-derived κ(T), α(T) and diffusivities — not just numerical agreement.

---

## Reference

See [`TEBC_implementation_spec.md`](TEBC_implementation_spec.md) for the complete implementation specification — every governing equation, module contract, and parameter handoff is documented there.
