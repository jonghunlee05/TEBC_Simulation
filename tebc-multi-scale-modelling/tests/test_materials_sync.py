"""Materials database sync test.

`tebc.constants.MATERIALS` is the runtime authority; `data/materials_db.json`
is the human-readable source. They use different field names but cover
overlapping numerics. This test fails CI if the values drift apart on
the keys we know correspond, providing a cheap proxy for a "single
source of truth" until a proper loader replaces both.
"""

import json
from pathlib import Path

import pytest

from tebc.constants import MATERIALS

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "materials_db.json"


@pytest.fixture(scope="module")
def db_json():
    with DB_PATH.open() as f:
        return json.load(f)


# Field-pair correspondences. Each row: (material, json_key, py_key,
# transform from JSON value to expected Python value).
PAIRS = [
    ("beta_Yb2Si2O7", "rho_kg_m3",      "rho",   lambda v: v),
    ("beta_Yb2Si2O7", "E_Pa",           "E",     lambda v: v),
    ("beta_Yb2Si2O7", "nu",             "nu",    lambda v: v),
    ("beta_Yb2Si2O7", "alpha_K",        "alpha", lambda v: v),
    ("beta_Yb2Si2O7", "kappa_Wm1K1",    "kappa", lambda v: v),
    ("beta_Yb2Si2O7", "Gamma_int_Jm2",  "Gamma_interface", lambda v: v),
    ("beta_Yb2Si2O7", "T_melt_K",       "T_melt", lambda v: v),
    ("beta_Yb2Si2O7", "k_p_m2s_1316C",  "k_p_TGO", lambda v: v),
    ("beta_Yb2Si2O7", "Ea_kp_Jmol",     "Ea_kp",   lambda v: v),

    ("beta_Y2Si2O7",  "rho_kg_m3",      "rho",   lambda v: v),
    ("beta_Y2Si2O7",  "E_Pa",           "E",     lambda v: v),
    ("beta_Y2Si2O7",  "k_p_m2s_1316C",  "k_p_TGO", lambda v: v),

    ("7YSZ",          "rho_kg_m3",      "rho",         lambda v: v),
    ("7YSZ",          "E_dense_Pa",     "E",           lambda v: v),
    ("7YSZ",          "kappa_dense_Wm1K1","kappa",     lambda v: v),
    ("7YSZ",          "D0_O_m2s",       "D0_O",        lambda v: v),

    ("Si_bondcoat",   "rho_kg_m3",      "rho",   lambda v: v),
    ("Si_bondcoat",   "E_Pa",           "E",     lambda v: v),
    ("Si_bondcoat",   "k_p_dry_m2s_1473K", "k_p_dry", lambda v: v),
    ("Si_bondcoat",   "Ea_kp_dry_Jmol",    "Ea_kp_dry", lambda v: v),
    ("Si_bondcoat",   "k_p_wet_m2s_1589K", "k_p_wet", lambda v: v),
    ("Si_bondcoat",   "Ea_kp_wet_Jmol",    "Ea_kp_wet", lambda v: v),

    ("SiC_SiC_CMC",   "rho_kg_m3",      "rho",   lambda v: v),
    ("SiC_SiC_CMC",   "E_Pa",           "E",     lambda v: v),

    ("SiO2_TGO",      "rho_kg_m3",      "rho",   lambda v: v),
    ("SiO2_TGO",      "E_Pa",           "E",     lambda v: v),
    ("SiO2_TGO",      "PBR",            "PBR",   lambda v: v),
]


@pytest.mark.parametrize("material,json_key,py_key,transform", PAIRS)
def test_materials_drift(db_json, material, json_key, py_key, transform):
    """JSON and Python entries must agree on every shared field."""
    json_val = db_json[material][json_key]
    py_val   = MATERIALS[material][py_key]
    assert py_val == pytest.approx(transform(json_val), rel=1e-6), (
        f"{material}: JSON {json_key}={json_val} ≠ Python {py_key}={py_val}"
    )
