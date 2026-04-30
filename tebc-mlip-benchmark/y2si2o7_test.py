from ase.io import read
from mace.calculators import mace_mp

atoms = read("Y2Si2O7.cif")

calc = mace_mp(model="small", device="cpu", default_dtype="float32")
atoms.calc = calc

energy = atoms.get_potential_energy()
n = len(atoms)
a, b, c, alpha, beta, gamma = atoms.cell.cellpar()

print(f"Composition:  {atoms.get_chemical_formula()}")
print(f"Atoms:        {n}")
print(f"Total energy: {energy:.6f} eV")
print(f"Per atom:     {energy / n:.6f} eV/atom")
print()
print("Lattice parameters (loaded vs Toher 2023):")
print(f"  a = {a:.4f} Å    (ref ~ 6.88)")
print(f"  b = {b:.4f} Å    (ref ~ 8.98)")
print(f"  c = {c:.4f} Å    (ref ~ 4.72)")
print(f"  beta = {beta:.2f} deg  (ref ~ 101.7)")
