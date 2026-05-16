"""
cmas_reactivity.py

Rank rare-earth silicate EBC candidates by their thermodynamic reactivity
with CMAS proxies (CaO, anorthite CaAl2Si2O8, MgO, and a model CMAS glass),
using Materials Project DFT energies and
pymatgen.analysis.interface_reactions.InterfacialReactivity.

Phase diagrams are built per rare-earth element in the
{RE, Si, O, Ca, Mg, Al} chemical system, using the legacy GGA / GGA+U
thermo type (the joint r2SCAN hull excludes many lanthanide entries).

Input  : rare_earth_silicates.csv    (95 RE-Si-O compounds with mp_ids)
Outputs: cmas_reactivity_ranking.csv  (every mp_id, sorted least -> most reactive)
         cmas_product_map.csv         (products per compound x proxy, classified
                                       into structural families)
         cmas_top_candidates.csv      (least-reactive shortlist, worst case across
                                       proxies, vs the Yb2Si2O7 baseline)
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
PRODUCT_MAP_CSV = HERE / "cmas_product_map.csv"
TOP_CANDIDATES_CSV = HERE / "cmas_top_candidates.csv"
CACHE_DIR = HERE / ".pd_cache"
CACHE_DIR.mkdir(exist_ok=True)

API_KEY = os.environ["MP_API_KEY"]
EXTENDED_CHEMSYS = True            # add Mg, Al -> required for CaAl2Si2O8 / CMAS proxies
# CMAS proxies, keyed by a short label used for the output column names.
# Model CMAS glass: 33 CaO : 9 MgO : 13 AlO1.5 : 45 SiO2 (mol%), x2 to clear fractions.
CMAS_PROXIES = {
    "CaO": "CaO",                          # basic oxide, aggressive single-oxide proxy
    "CaAl2Si2O8": "CaAl2Si2O8",            # anorthite, crystalline aluminosilicate
    "MgO": "MgO",                          # the M in CMAS
    "modelCMAS": "Ca66Mg18Al26Si90O303",   # representative CMAS glass
}
RANK_KEY = "rxn_energy_CaAl2Si2O8_eV_per_atom"   # used to sort the final CSV
PRIMARY_THERMO = ["GGA_GGA+U"]               # forced for stability
FALLBACK_THERMO = ["GGA_GGA+U_R2SCAN"]       # used only when RE is absent under primary

BASELINE_FORMULA = "Yb2Si2O7"   # standard EBC; reference row in the top-N table
TOP_N = 15                      # rows reported as "top candidates"
RE_ELEMENTS = {"La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy",
               "Ho", "Er", "Tm", "Yb", "Lu", "Y", "Sc"}


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


# --------------------------------------------------------------------------
# Output helpers: product classification and candidate ranking
# --------------------------------------------------------------------------
def parse_products(reaction: str | None) -> list[str]:
    """Extract product formulas from the RHS of a pymatgen reaction string,
    e.g. '0.2 Y2Si2O7 + 0.8 CaO -> 0.4 Ca2SiO4 + 0.2 Y2O3' -> ['Ca2SiO4','Y2O3'].
    Returns [] for missing or errored reactions."""
    if not reaction or reaction.startswith("ERROR") or "->" not in reaction:
        return []
    rhs = reaction.split("->", 1)[1]
    return [term.split()[-1] for term in rhs.split(" + ") if term.split()]


def classify_product(formula: str) -> str:
    """Heuristic structural-family label for a product phase, inferred from its
    element signature (not from the actual crystal structure). Tuned for the
    RE-Ca-Mg-Al-Si-O chemistry of CMAS attack on rare-earth silicate EBCs."""
    try:
        comp = Composition(formula)
    except Exception:
        return "unknown"
    els = {str(e) for e in comp.elements}
    only = lambda *xs: els == set(xs)
    has = lambda *xs: all(x in els for x in xs)
    re = next(iter(els & RE_ELEMENTS), None)

    if re and has("Ca", "Si", "O") and not (els & {"Al", "Mg"}):
        return "Ca-RE apatite"          # signature CMAS degradation product
    if re and "Al" in els and "O" in els and not (els & {"Si", "Ca", "Mg"}):
        r = comp.get_atomic_fraction("Al") / comp.get_atomic_fraction(re)
        if abs(r - 5 / 3) < 0.3:
            return "RE-Al garnet"        # RE3Al5O12 (YAG-type)
        if abs(r - 1.0) < 0.3:
            return "RE aluminate perovskite"   # REAlO3
        return "RE aluminate (other)"
    if re and only(re, "Si", "O"):
        r = comp.get_atomic_fraction(re) / comp.get_atomic_fraction("Si")
        if abs(r - 1.0) < 0.2:
            return "RE disilicate"       # RE2Si2O7
        if abs(r - 2.0) < 0.3:
            return "RE monosilicate"     # RE2SiO5
        if 1.4 < r < 1.8:
            return "RE silicate apatite"
        return "RE silicate (other)"
    if re and only(re, "O"):
        return "RE oxide"
    if only("Ca", "Si", "O"):
        return "Ca silicate"             # Ca2SiO4, CaSiO3, ...
    if only("Mg", "Si", "O"):
        return "Mg silicate"             # forsterite / enstatite
    if only("Mg", "Al", "O"):
        return "spinel"                  # MgAl2O4
    if has("Ca", "Al", "Si", "O") and not re:
        return "anorthite/feldspar"
    for ox, name in (("Ca", "CaO"), ("Mg", "MgO"), ("Al", "Al2O3"), ("Si", "SiO2")):
        if only(ox, "O"):
            return name
    return "other"


def write_product_map(pair_results: dict, path: Path) -> pd.DataFrame:
    """One row per (compound, proxy): the reaction, its products, and the
    structural families those products fall into. A zero reaction energy means
    no favorable reaction, recorded as '(no reaction)'."""
    rows = []
    for (re_elem, formula), row in pair_results.items():
        for label in CMAS_PROXIES:
            reaction = row.get(f"rxn_{label}")
            energy = row.get(f"rxn_energy_{label}_eV_per_atom")
            if energy is not None and energy > -1e-6:
                products, families = "", "(no reaction)"
            else:
                prods = parse_products(reaction)
                products = ", ".join(prods)
                families = (", ".join(sorted({classify_product(p) for p in prods}))
                            or "(unclassified)")
            rows.append({
                "rare_earth": re_elem, "formula": formula, "proxy": label,
                "rxn_energy_eV_per_atom": energy, "reaction": reaction,
                "products": products, "product_families": families,
            })
    pm = pd.DataFrame(rows)
    pm.to_csv(path, index=False)
    return pm


def write_top_candidates(pair_results: dict, path: Path) -> pd.DataFrame:
    """Rank unique (RE, formula) candidates by their worst-case reaction energy
    across all proxies -- a coating must resist every CMAS component, so the
    most reactive proxy sets the score. Least reactive (highest energy) first."""
    rows = []
    for (re_elem, formula), row in pair_results.items():
        energies = {lbl: row.get(f"rxn_energy_{lbl}_eV_per_atom")
                    for lbl in CMAS_PROXIES}
        scored = {k: v for k, v in energies.items() if v is not None}
        if not scored:
            continue
        worst_proxy = min(scored, key=scored.get)
        rec = {"rare_earth": re_elem, "formula": formula,
               "worst_case_proxy": worst_proxy,
               "worst_case_energy_eV_per_atom": scored[worst_proxy],
               "is_baseline": formula == BASELINE_FORMULA}
        for lbl in CMAS_PROXIES:
            rec[f"rxn_energy_{lbl}_eV_per_atom"] = energies[lbl]
        rows.append(rec)
    tc = (pd.DataFrame(rows)
          .sort_values("worst_case_energy_eV_per_atom", ascending=False)
          .reset_index(drop=True))
    tc.insert(0, "rank", tc.index + 1)
    base = tc.loc[tc["is_baseline"], "worst_case_energy_eV_per_atom"]
    if not base.empty:
        tc["delta_vs_baseline_eV_per_atom"] = (
            tc["worst_case_energy_eV_per_atom"] - base.iloc[0])
    tc.to_csv(path, index=False)
    return tc


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
                for label, proxy in CMAS_PROXIES.items():
                    c2 = Composition(proxy)
                    try:
                        e, rxn = min_reaction(c1, c2, pd_)
                    except Exception as exc:
                        e, rxn = None, f"ERROR: {type(exc).__name__}: {exc}"
                    row[f"rxn_energy_{label}_eV_per_atom"] = e
                    row[f"rxn_{label}"] = rxn
                pair_results[(re_elem, formula)] = row
                fmt = lambda v: " none  " if v is None else f"{v:+.4f}"
                energies = "  ".join(
                    f"{label}: {fmt(row[f'rxn_energy_{label}_eV_per_atom'])}"
                    for label in CMAS_PROXIES
                )
                print(f"  {formula:14s}  {energies}")

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

    # --- Output 2: product map -------------------------------------------
    pm = write_product_map(pair_results, PRODUCT_MAP_CSV)
    print(f"\nWrote product map ({len(pm)} reactions) to {PRODUCT_MAP_CSV.name}")
    counts = (pm.assign(family=pm["product_families"].str.split(", "))
                .explode("family")
                .groupby(["family", "proxy"]).size()
                .unstack(fill_value=0)
                .reindex(columns=list(CMAS_PROXIES), fill_value=0))
    print("Product families x proxy (reaction count):")
    print(counts.to_string())

    # --- Output 3: top candidates ----------------------------------------
    tc = write_top_candidates(pair_results, TOP_CANDIDATES_CSV)
    print(f"\nWrote {len(tc)} ranked candidates to {TOP_CANDIDATES_CSV.name}")
    has_delta = "delta_vs_baseline_eV_per_atom" in tc.columns
    shown = tc.head(TOP_N)
    if not tc.loc[tc["is_baseline"]].empty:
        shown = pd.concat([shown, tc.loc[tc["is_baseline"]]]).drop_duplicates("rank")
    print(f"\nTop {TOP_N} least-reactive candidates "
          f"(worst case across {len(CMAS_PROXIES)} proxies):")
    for _, r in shown.iterrows():
        delta = (f"  d_vs_base {r['delta_vs_baseline_eV_per_atom']:+.4f}"
                 if has_delta else "")
        tag = "   [Yb2Si2O7 baseline]" if r["is_baseline"] else ""
        print(f"  #{int(r['rank']):<3} {r['rare_earth']:<3} {r['formula']:<18} "
              f"worst {r['worst_case_energy_eV_per_atom']:+.4f} eV/atom "
              f"({r['worst_case_proxy']}){delta}{tag}")


if __name__ == "__main__":
    main()
