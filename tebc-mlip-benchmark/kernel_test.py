from ase.build import bulk
from mace.calculators import mace_mp

si = bulk("Si", "diamond", a=5.43)

calc = mace_mp(model="small", device="cpu", default_dtype="float32")
si.calc = calc

energy = si.get_potential_energy()
print(f"Total energy: {energy:.6f} eV  ({len(si)} atoms)")
print(f"Per atom:     {energy / len(si):.6f} eV/atom")
