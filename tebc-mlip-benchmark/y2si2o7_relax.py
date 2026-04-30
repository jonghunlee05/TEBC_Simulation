import numpy as np
from ase.io import read
from ase.optimize import BFGS
from ase.filters import UnitCellFilter
from mace.calculators import mace_mp

REFERENCE = {"a": 6.88, "b": 8.98, "c": 4.72, "beta": 101.7}

atoms = read("Y2Si2O7.cif")
n = len(atoms)

calc = mace_mp(model="small", device="cpu", default_dtype="float64")
atoms.calc = calc

e_initial = atoms.get_potential_energy()
a0, b0, c0, _, beta0, _ = atoms.cell.cellpar()

ucf = UnitCellFilter(atoms)
opt = BFGS(ucf, logfile="-")
opt.run(fmax=0.01, steps=200)

e_final = atoms.get_potential_energy()
a, b, c, _, beta, _ = atoms.cell.cellpar()
fmax_atom = np.linalg.norm(atoms.get_forces(), axis=1).max()
stress = atoms.get_stress()
smax = np.abs(stress).max()

print()
print(f"Composition:        {atoms.get_chemical_formula()}  ({n} atoms)")
print()
print("Lattice parameters (Initial / Relaxed / Toher 2023 / dev vs ref):")
print(f"  {'Param':<8}{'Initial':>12}{'Relaxed':>12}{'Toher':>12}{'dev':>14}")
rows = [
    ("a (A)",  a0,    a,    REFERENCE["a"]),
    ("b (A)",  b0,    b,    REFERENCE["b"]),
    ("c (A)",  c0,    c,    REFERENCE["c"]),
    ("beta",   beta0, beta, REFERENCE["beta"]),
]
for label, init, final, ref in rows:
    dev = (final - ref) / ref * 100
    print(f"  {label:<8}{init:>12.4f}{final:>12.4f}{ref:>12.4f}{dev:>+13.2f}%")

print()
print(f"Energy initial:     {e_initial:.6f} eV    ({e_initial/n:.6f} eV/atom)")
print(f"Energy relaxed:     {e_final:.6f} eV    ({e_final/n:.6f} eV/atom)")
print(f"dE relaxation:      {e_final - e_initial:+.6f} eV  ({(e_final - e_initial)/n*1000:+.3f} meV/atom)")

print()
print(f"Optimisation steps: {opt.nsteps}")
print(f"Final max atomic force:  {fmax_atom:.6f} eV/A")
print(f"Final max |stress|:      {smax:.6f} eV/A^3  ({smax * 160.21766208:.4f} GPa)")
