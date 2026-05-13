"""
cmas_reactivity.py

Rank rare-earth silicate EBC candidates by their thermodynamic reactivity
with CMAS proxies (CaO and anorthite CaAl2Si2O8), using Materials Project
DFT energies and pymatgen.analysis.interface_reactions.InterfacialReactivity.

Phase diagrams are built per rare-earth element in the
{RE, Si, O, Ca, Mg, Al} chemical system, using the legacy GGA / GGA+U
thermo type (the joint r2SCAN hull excludes many lanthanide entries).

Input : rare_earth_silicates.csv  (95 RE-Si-O compounds with mp_ids)
Output: cmas_reactivity_ranking.csv  (sorted least -> most reactive)
"""
from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path

import pandas as pd
from mp_api.client import MPRester
from pymatgen.analysis.interface_reactions import InterfacialReactivity
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition

warnings.filterwarnings("ignore", category=UserWarning)

HERE = Path(__file__).resolve().parent
INPUT_CSV = HERE / "rare_earth_silicates.csv"
OUTPUT_CSV = HERE / "cmas_reactivity_ranking.csv"
CACHE_DIR = HERE / ".pd_cache"
CACHE_DIR.mkdir(exist_ok=True)

API_KEY = os.environ["MP_API_KEY"]
EXTENDED_CHEMSYS = True            # add Mg, Al -> required for CaAl2Si2O8 proxy
CMAS_PROXIES = ["CaO", "CaAl2Si2O8"]
RANK_KEY = "rxn_energy_CaAl2Si2O8_eV_per_atom"   # used to sort the final CSV
PRIMARY_THERMO = ["GGA_GGA+U"]               # forced for stability
FALLBACK_THERMO = ["GGA_GGA+U_R2SCAN"]       # used only when RE is absent under primary


def chemsys_for(re_elem: str) -> list[str]:
    chemsys = [re_elem, "Si", "O", "Ca"]
    if EXTENDED_CHEMSYS:
        chemsys += ["Mg", "Al"]
    return chemsys


def fetch_phase_diagram(mpr: MPRester, re_elem: str) -> tuple[PhaseDiagram, str]:
    """Build a PD in the RE chemsys. Forces GGA_GGA+U; if that loses the RE
    entirely (notably Yb, whose only entries live on the r2SCAN-joint hull),
    falls back to GGA_GGA+U_R2SCAN. Returns (pd, thermo_used)."""
    cache = CACHE_DIR / f"pd_{re_elem}_ext{int(EXTENDED_CHEMSYS)}.pkl"
    if cache.exists():
        with cache.open("rb") as fh:
            pd_, thermo_used = pickle.load(fh)
        return pd_, thermo_used

    chemsys = chemsys_for(re_elem)
    entries = mpr.get_entries_in_chemsys(
        chemsys, additional_criteria={"thermo_types": PRIMARY_THERMO}
    )
    pd_ = PhaseDiagram(entries)
    thermo_used = PRIMARY_THERMO[0]
    if re_elem not in {str(el) for el in pd_.elements}:
        print(f"  [fallback] {re_elem} absent under {PRIMARY_THERMO[0]}; "
              f"retrying with {FALLBACK_THERMO[0]}")
        entries = mpr.get_entries_in_chemsys(
            chemsys, additional_criteria={"thermo_types": FALLBACK_THERMO}
        )
        pd_ = PhaseDiagram(entries)
        thermo_used = FALLBACK_THERMO[0]

    with cache.open("wb") as fh:
        pickle.dump((pd_, thermo_used), fh)
    return pd_, thermo_used


def min_reaction(c1: Composition, c2: Composition, pd_: PhaseDiagram):
    """Return (most-favorable reaction energy in eV/atom, reaction string)
    along the c1-c2 mixing line. use_hull_energy=True so we evaluate the
    stable polymorph of each composition (consistent screening basis)."""
    ir = InterfacialReactivity(
        c1=c1, c2=c2, pd=pd_,
        norm=True, use_hull_energy=True,
    )
    # tuple layout: (index, mixing_ratio, e_eV_per_atom, Reaction, kJ_per_mol)
    kinks = list(ir.get_kinks())
    if not kinks:
        return None, None
    best = min(kinks, key=lambda k: k[2])
    return float(best[2]), str(best[3])


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} compounds from {INPUT_CSV.name}")

    # Reactivity depends only on (RE, formula); polymorphs share the value.
    pair_results: dict[tuple[str, str], dict] = {}

    with MPRester(API_KEY) as mpr:
        for re_elem in sorted(df["rare_earth"].unique()):
            sub = df[df["rare_earth"] == re_elem]
            print(
                f"\n=== {re_elem}: {len(sub)} entries, "
                f"{sub['formula'].nunique()} unique formulas ==="
            )
            try:
                pd_, thermo_used = fetch_phase_diagram(mpr, re_elem)
            except Exception as exc:
                print(f"  [WARN] failed to build PD for {re_elem}: {exc}")
                continue
            print(
                f"  PD ({thermo_used}): {len(pd_.all_entries)} entries, "
                f"{len(pd_.stable_entries)} stable"
            )

            for formula in sub["formula"].unique():
                c1 = Composition(formula)
                row: dict = {"rare_earth": re_elem, "formula": formula,
                             "thermo_type": thermo_used}
                for proxy in CMAS_PROXIES:
                    c2 = Composition(proxy)
                    try:
                        e, rxn = min_reaction(c1, c2, pd_)
                    except Exception as exc:
                        e, rxn = None, f"ERROR: {type(exc).__name__}: {exc}"
                    row[f"rxn_energy_{proxy}_eV_per_atom"] = e
                    row[f"rxn_{proxy}"] = rxn
                pair_results[(re_elem, formula)] = row
                e_cao = row.get("rxn_energy_CaO_eV_per_atom")
                e_an = row.get("rxn_energy_CaAl2Si2O8_eV_per_atom")
                fmt = lambda v: "  none " if v is None else f"{v:+.4f}"
                print(f"  {formula:14s}  CaO: {fmt(e_cao)}   CaAl2Si2O8: {fmt(e_an)}")

    # Broadcast reactivity onto every mp_id row.
    enriched = []
    for _, row in df.iterrows():
        rxn = pair_results.get((row["rare_earth"], row["formula"]), {})
        out_row = row.to_dict()
        for k, v in rxn.items():
            if k not in ("rare_earth", "formula"):
                out_row[k] = v
        enriched.append(out_row)

    out = pd.DataFrame(enriched)
    # Least to most reactive: largest (least negative / most positive) energy first.
    out = out.sort_values(RANK_KEY, ascending=False, na_position="last")
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {len(out)} ranked compounds to {OUTPUT_CSV.name}")
    print(f"Sorted by {RANK_KEY} (least to most reactive).")


if __name__ == "__main__":
    main()
