import json
from pathlib import Path

import numpy as np
from ase.io import read
from ase.optimize import BFGS
from ase.filters import UnitCellFilter
from mace.calculators import mace_mp


MODEL_NAME = "MACE-MP-0"
MODEL_VARIANT = "medium"
DEVICE = "cpu"
DTYPE = "float64"
FMAX_TARGET = 0.01
MAX_STEPS = 200


def relax_composition(cif_path, label):
    atoms = read(cif_path)
    n = len(atoms)

    calc = mace_mp(model=MODEL_VARIANT, device=DEVICE, default_dtype=DTYPE)
    atoms.calc = calc

    e_initial = float(atoms.get_potential_energy())
    a0, b0, c0, _, beta0, _ = atoms.cell.cellpar().tolist()

    ucf = UnitCellFilter(atoms)
    opt = BFGS(ucf, logfile="-")
    opt.run(fmax=FMAX_TARGET, steps=MAX_STEPS)

    e_final = float(atoms.get_potential_energy())
    a, b, c, _, beta, _ = atoms.cell.cellpar().tolist()
    fmax_atom = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    smax = float(np.abs(atoms.get_stress()).max())

    deviation = {
        "a":    (a    - a0)    / a0    * 100,
        "b":    (b    - b0)    / b0    * 100,
        "c":    (c    - c0)    / c0    * 100,
        "beta": (beta - beta0) / beta0 * 100,
    }

    return {
        "label": label,
        "cif_source": str(cif_path),
        "composition": atoms.get_chemical_formula(),
        "n_atoms": n,
        "model": {"name": MODEL_NAME, "variant": MODEL_VARIANT, "dtype": DTYPE, "device": DEVICE},
        "reference_origin": "Materials Project DFT-PBE (CIF input)",
        "lattice": {
            "dft_pbe": {"a": a0, "b": b0, "c": c0, "beta": beta0},
            "mace_relaxed": {"a": a, "b": b, "c": c, "beta": beta},
            "deviation_pct": deviation,
        },
        "energy_eV": {
            "initial": e_initial,
            "relaxed": e_final,
            "delta": e_final - e_initial,
        },
        "energy_per_atom_eV": {
            "initial": e_initial / n,
            "relaxed": e_final / n,
        },
        "convergence": {
            "fmax_target": FMAX_TARGET,
            "n_steps": opt.nsteps,
            "fmax_atom_final": fmax_atom,
            "smax_final_eV_per_A3": smax,
        },
    }


def print_comparison_table(results):
    header = f"{'Composition':<16} {'Param':<6} {'MP DFT-PBE':>12} {'MACE relax':>12} {'dev':>10}"
    print()
    print(header)
    print("-" * len(header))
    for r in results:
        for key in ("a", "b", "c", "beta"):
            print(
                f"{r['label']:<16} {key:<6} "
                f"{r['lattice']['dft_pbe'][key]:>12.4f} "
                f"{r['lattice']['mace_relaxed'][key]:>12.4f} "
                f"{r['lattice']['deviation_pct'][key]:>+9.2f}%"
            )
        print("-" * len(header))
    print()


COMPOSITIONS = [
    ("beta-Y2Si2O7",  "Y2Si2O7.cif"),
    ("beta-Yb2Si2O7", "Yb2Si2O7.cif"),
    ("beta-Lu2Si2O7", "Lu2Si2O7.cif"),
    ("beta-Er2Si2O7", "Er2Si2O7.cif"),
    ("beta-Sc2Si2O7", "Sc2Si2O7.cif"),
]


def main():
    results = []
    for label, cif in COMPOSITIONS:
        print(f"\n=== Relaxing {label} ({cif}) ===")
        results.append(relax_composition(cif, label))

    print_comparison_table(results)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"relaxation_results_{MODEL_VARIANT}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
