"""Unit tests for Scale 1 helpers that don't need a live VASP run."""

import pytest

# pymatgen is required to import the module top-level; skip the whole file
# if it's missing in the test environment (e.g. lightweight CI).
pytest.importorskip("pymatgen")

from tebc.scale1_atomistic.dft_interface import (  # noqa: E402
    compute_cohesive_energy,
    compute_defect_formation_energy,
    compute_surface_energy,
)


def test_cohesive_energy_sign():
    """A bound crystal (E_crystal < Σ E_atom) yields E_coh > 0."""
    E_coh = compute_cohesive_energy(
        E_crystal_eV=-100.0, N=10,
        E_atoms_eV={"X": -5.0},
        composition={"X": 10},
    )
    assert E_coh > 0
    assert E_coh == pytest.approx((-(-100.0 - (-50.0))) / 10)


def test_surface_energy_nonnegative_for_bulk_split():
    """Cleaving a bulk slab into two surfaces requires γ > 0."""
    g = compute_surface_energy(
        E_slab=-99.0, E_bulk_per_atom=-10.0,
        N_slab=10, A_surface_m2=1e-19,
    )
    assert g > 0


def test_defect_formation_energy_neutral():
    """Neutral defect (q=0) should drop the chem-potential and Fermi terms cleanly."""
    Ef = compute_defect_formation_energy(
        E_defect=-95.0, E_host=-100.0,
        mu={"X": -10.0},
        delta_n={"X": -1},   # one X removed → vacancy
        q=0, E_VBM=0.0, E_Fermi=0.0,
    )
    # E_def - E_host - δn·μ = -95 - (-100) - (-1)·(-10) = -5
    assert Ef == pytest.approx(-5.0)
