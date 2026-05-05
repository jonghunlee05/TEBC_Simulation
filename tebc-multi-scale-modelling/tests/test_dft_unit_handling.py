"""Regression tests for `dft_interface` unit handling.

`parse_elastic_tensor` returns Pa (not GPa). `extract_born_effective_charges`
enforces ΣᵢZ*ᵢ = 0 by default. Both are mocked because pymatgen needs real
VASP outputs and we want CI-fast deterministic checks.
"""

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest


def _install_pymatgen_mock(monkeypatch, elastic_kBar, born=None):
    """Inject a fake pymatgen so dft_interface._require_pymatgen works."""
    fake_outcar = MagicMock()
    fake_outcar.elastic_tensor = elastic_kBar
    fake_outcar.born = born if born is not None else np.zeros((4, 3, 3))

    OutcarCls = MagicMock(return_value=fake_outcar)

    class FakeElasticTensor:
        def __init__(self, voigt):
            self._voigt = np.asarray(voigt)
        @property
        def voigt(self):
            return self._voigt
        @classmethod
        def from_voigt(cls, arr):
            return cls(arr)

    fake_pymatgen = types.ModuleType("pymatgen")
    fake_analysis = types.ModuleType("pymatgen.analysis")
    fake_elasticity = types.ModuleType("pymatgen.analysis.elasticity")
    fake_elasticity.ElasticTensor = FakeElasticTensor
    fake_io = types.ModuleType("pymatgen.io")
    fake_vasp = types.ModuleType("pymatgen.io.vasp")
    fake_outputs = types.ModuleType("pymatgen.io.vasp.outputs")
    fake_outputs.Outcar = OutcarCls
    fake_outputs.Vasprun = MagicMock()

    monkeypatch.setitem(sys.modules, "pymatgen", fake_pymatgen)
    monkeypatch.setitem(sys.modules, "pymatgen.analysis", fake_analysis)
    monkeypatch.setitem(sys.modules, "pymatgen.analysis.elasticity", fake_elasticity)
    monkeypatch.setitem(sys.modules, "pymatgen.io", fake_io)
    monkeypatch.setitem(sys.modules, "pymatgen.io.vasp", fake_vasp)
    monkeypatch.setitem(sys.modules, "pymatgen.io.vasp.outputs", fake_outputs)
    return OutcarCls


def test_parse_elastic_tensor_returns_pascals(monkeypatch):
    """A 1850 kBar input must come out as 1.85e11 Pa (not 1.85e10 GPa)."""
    elastic_kBar = np.diag([1850.0] * 6)              # 185 GPa diagonal
    _install_pymatgen_mock(monkeypatch, elastic_kBar)

    from tebc.scale1_atomistic.dft_interface import parse_elastic_tensor
    C = parse_elastic_tensor("/fake/OUTCAR")
    # 1850 kBar = 1.85e11 Pa
    assert C[0, 0] == pytest.approx(1.85e11, rel=1e-9)


def test_extract_born_enforces_sum_rule(monkeypatch):
    """After enforcement, Σᵢ Z*ᵢ over atoms must be ≈ 0 per (α,β)."""
    # Three atoms with non-zero ΣZ* before correction.
    raw = np.array([
        [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
    ])
    _install_pymatgen_mock(monkeypatch, np.eye(6), born=raw)

    from tebc.scale1_atomistic.dft_interface import extract_born_effective_charges
    Z = extract_born_effective_charges("/fake/OUTCAR", enforce_sum_rule=True)
    np.testing.assert_allclose(Z.sum(axis=0), np.zeros((3, 3)), atol=1e-12)


def test_extract_born_can_skip_sum_rule(monkeypatch):
    raw = np.array([
        [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ])
    _install_pymatgen_mock(monkeypatch, np.eye(6), born=raw)

    from tebc.scale1_atomistic.dft_interface import extract_born_effective_charges
    Z = extract_born_effective_charges("/fake/OUTCAR", enforce_sum_rule=False)
    # Without enforcement, sum is the original (non-zero).
    assert Z.sum(axis=0)[0, 0] == pytest.approx(4.0)
