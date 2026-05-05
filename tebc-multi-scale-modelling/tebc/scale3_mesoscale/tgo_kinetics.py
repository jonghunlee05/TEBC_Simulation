"""
Thermally Grown Oxide (SiO₂) kinetics on Si bond coat.

Deal-Grove:   x² + Ax = B(t + τ)
Paralinear:   dx/dt = k_p/(2x) - k_l
PBR stress:   ε_TGO^ox = ⅓(PBR - 1) δ_{ij}
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from tebc.constants import R_gas
from tebc.utils import arrhenius_eval


def deal_grove_thickness(t: np.ndarray, k_p: float, k_l: float,
                          x0: float = 0.0) -> np.ndarray:
    """
    Thickness x(t) from Deal-Grove: x² + Ax = B(t + τ).

    Convention used here:
      B   = 2·k_p          (so pure-parabolic limit gives x² = 2 k_p t)
      A   = 2·k_p / k_l    (so B/A = k_l, the linear-rate constant)
      τ   = x0·(x0 + A)/B  (virtual time offset for x(0) = x0)

    Closed-form solution:
      x(t) = (A/2)·[ √(1 + 4·B·(t+τ)/A²) − 1 ]

    Limits:
      thin scale  (B(t+τ) ≪ A²/4):  x → k_l (t+τ)
      thick scale (B(t+τ) ≫ A²/4):  x → √(2 k_p (t+τ))
    """
    A = 2.0 * k_p / k_l if k_l > 0 else 1e30
    B = 2.0 * k_p
    tau = x0 * (x0 + A) / B if B > 0 else 0.0
    discriminant = 1.0 + 4.0 * B * (t + tau) / (A**2 + 1e-30)
    x = (A / 2.0) * (np.sqrt(np.maximum(discriminant, 0.0)) - 1.0)
    return x


def parabolic_rate_constant(T_K: float, k_ref: float, Ea_J: float,
                              T_ref_K: float | None = None) -> float:
    """Arrhenius extrapolation of a parabolic rate constant [m²/s].

    Two calling conventions:

    * ``T_ref_K`` given (preferred): `k_ref` is a *measured rate at T_ref*,
      and the function returns
          k(T) = k_ref · exp(-Ea/R · (1/T - 1/T_ref))
      i.e. the standard reference-shifted Arrhenius form. This is what the
      `materials_db.json` and `tebc.constants.MATERIALS` entries provide
      (e.g. `k_p_m2s_1316C` measured at 1316 °C).

    * ``T_ref_K`` omitted (legacy / explicit prefactor): `k_ref` is treated
      as the infinite-temperature prefactor A in k(T) = A·exp(-Ea/RT). This
      path is kept for callers that already have a true Arrhenius A from
      a fit, but mis-using it with a *measured* k_ref will under-predict
      k(T) at high temperature by exp(Ea/(R·T_ref)) — typically 10²–10⁴×.
    """
    if T_ref_K is None:
        return arrhenius_eval(T_K, k_ref, Ea_J)
    # Reference-shifted form, numerically stable for any |T - T_ref|.
    return k_ref * np.exp(-Ea_J / R_gas * (1.0 / T_K - 1.0 / T_ref_K))


def paralinear_ode(t: float, x: np.ndarray,
                    k_p: float, k_l: float):
    """
    dx/dt = k_p / (2x) - k_l. Steady state x_ss = k_p / (2 k_l).

    Once x ≤ 0 (volatilization outpaces oxidation faster than ODE can
    relax), the protective scale has been consumed; physically dx/dt = 0
    and recession proceeds at k_l on the bare substrate.
    """
    if x[0] <= 0.0:
        return [0.0]
    return [k_p / (2.0 * x[0]) - k_l]


def solve_paralinear(t_span: tuple, k_p: float, k_l: float,
                      x0: float = None,
                      n_points: int = 500,
                      PBR: float = 2.15) -> dict:
    """
    Solve paralinear ODE for TGO thickness x(t) and substrate recession.

    Numerically robust against the volatilization-dominated regime
    (k_l ≫ √(k_p/Δt)): integrate in y = x² instead of x, where
        dy/dt = k_p - 2 k_l √y,
    which keeps y ≥ 0 by clipping under the square root and asymptotes
    smoothly to y_ss = (k_p/(2 k_l))².

    Default x0 is set adaptively to ½·x_ss so the trajectory starts
    inside the basin of attraction of the steady state.

    UNITS NOTE: The Wagner-style substrate recession formula
    `recession = (x − x₀ + k_l·t) / PBR` returned in the result dict
    assumes ``k_l`` and ``k_p`` are expressed in **oxide-thickness**
    units (so that x_ss = k_p/(2·k_l) is the oxide steady-state
    thickness). The `tebc.constants.RS_K_L_REF` calibration to the
    Opila/Robinson-Smialek anchor follows the *substrate-recession*
    convention used in those papers; mixing the two would introduce
    an implicit factor of PBR. This is flagged in KNOWN_LIMITATIONS
    until a deliberate unit audit is done; the immediate `recession`
    number should therefore be treated as illustrative rather than
    quantitative.

    `PBR` defaults to 2.15 (SiO₂ on Si). Override for other oxide
    systems (Al₂O₃ on Al ≈ 1.28, Cr₂O₃ on Cr ≈ 2.07, etc.).
    """
    t_eval = np.linspace(*t_span, n_points)
    x_ss   = k_p / (2.0 * k_l) if k_l > 0 else np.inf

    if x0 is None:
        x0 = max(min(0.5 * x_ss, 1e-9), 1e-15)

    def rhs_y(t, y, k_p, k_l):
        y_safe = max(y[0], 0.0)
        return [k_p - 2.0 * k_l * np.sqrt(y_safe)]

    y0 = x0 ** 2
    sol = solve_ivp(rhs_y, t_span, [y0], args=(k_p, k_l),
                    t_eval=t_eval, method="LSODA",
                    rtol=1e-8, atol=1e-24)
    if not sol.success:
        import warnings
        warnings.warn(
            f"solve_paralinear: LSODA reported failure ({sol.message!r}); "
            f"results may be unreliable.",
            RuntimeWarning, stacklevel=2,
        )
    x_TGO = np.sqrt(np.maximum(sol.y[0], 0.0))

    # Wagner / paralinear substrate recession (see UNITS NOTE in the
    # docstring). Each unit of oxide formation consumes 1/PBR units of
    # substrate; integrating over the paralinear ODE with x0 = x(0):
    #     Si_loss(t) = (1/PBR) · [x(t) − x0 + k_l·t]
    # The previous code returned `recession = k_l · t`, which is the
    # oxide mass loss to volatilization rather than substrate loss.
    recession = (x_TGO - x0 + k_l * sol.t) / PBR
    return {"t": sol.t, "x_TGO": x_TGO, "recession": recession,
            "x_ss": x_ss, "success": bool(sol.success)}


def tgo_growth_stress(x_TGO: float, E_TGO: float, nu_TGO: float,
                       alpha_TGO: float, alpha_sub: float,
                       dT: float, PBR: float = 2.15) -> float:
    """
    Total TGO biaxial stress = thermal + growth (PBR).

    The biaxial film stress is *thickness-independent* in this elastic
    treatment (Stoney-like), so `x_TGO` does not enter the formula —
    it is kept in the signature only because callers commonly have
    x_TGO at hand and the parameter documents that the result is the
    stress *in* the oxide layer of thickness x_TGO. (A real model with
    thickness-dependent plastic relaxation would use it.)

    Linear (1-D) growth strain from a volumetric Pilling–Bedworth ratio:

        ε_growth = PBR^(1/3) − 1

    The previous implementation used the small-strain approximation
    (PBR − 1)/3, which only holds for PBR ≈ 1; for SiO₂ on Si
    (PBR = 2.15) it overestimates the strain by ≈ 30 % (0.383 vs 0.291).

    Note: this is the *elastic* growth-stress estimate. Real TGOs
    accommodate most of this strain by viscoplastic flow in the bond
    coat, so measured stresses are typically 1–3 GPa rather than the
    ~30 GPa this formula predicts. See KNOWN_LIMITATIONS.md item #3.
    """
    del x_TGO  # documented above; intentionally unused.
    eps_growth = PBR ** (1.0 / 3.0) - 1.0
    biaxial_mod = E_TGO / (1 - nu_TGO)
    sigma_thermal = biaxial_mod * (alpha_TGO - alpha_sub) * dT
    sigma_growth  = -biaxial_mod * eps_growth
    return sigma_thermal + sigma_growth


def integrate_tgo_temperature_schedule(
    t_total: float, schedule: list,
    k_p_at: callable, k_l_at: callable,
    x0: float | None = None, n_points: int = 500,
    cycle_period: float | None = None,
    PBR: float = 2.15,
) -> dict:
    """Integrate the paralinear ODE under a periodic temperature schedule.

    Real T/EBC service is *cyclic*, not isothermal; the bulk of each
    cycle is spent ramping or dwelling at intermediate T where the
    Arrhenius rates k_p(T), k_l(T) are orders of magnitude smaller than
    at T_hot. Integrating at constant T_hot for the whole `t_total`
    therefore overestimates total TGO growth by a large factor.

    Parameters
    ----------
    t_total : float
        Total cumulative service time [s] (sum of all dwells across all
        cycles).
    schedule : list of (T_K, fraction)
        Duty cycle: each entry is a temperature [K] and the fraction of
        the cycle spent at that T. Fractions must sum to 1. For a simple
        hot-dwell / cold-soak engine cycle, e.g.
        [(1600, 0.7), (1000, 0.2), (400, 0.1)].
    k_p_at, k_l_at : callable
        Functions T_K → rate constant (m²/s and m/s respectively).
    x0, n_points : numerics, see `solve_paralinear`.

    Returns
    -------
    dict with keys ``t``, ``x_TGO``, ``recession``, ``x_ss_at_each_T``,
    ``effective_k_p``, ``effective_k_l``. The trajectory uses the
    duty-cycle-weighted *effective* rates rather than a piecewise
    integration; this is exact for a true paralinear (where the rate is
    additive) and a defensible average for the parabolic regime.
    """
    fracs = np.array([f for _, f in schedule], dtype=float)
    if not np.isclose(fracs.sum(), 1.0):
        raise ValueError(
            f"Schedule fractions must sum to 1, got {fracs.sum():.6f}.",
        )
    if (fracs < 0).any():
        raise ValueError("Schedule fractions must be non-negative.")
    Ts = np.array([T for T, _ in schedule], dtype=float)
    if (Ts <= 0).any():
        raise ValueError("Schedule temperatures must be positive.")

    # Duty-cycle-weighted effective rates.
    k_p_eff = float(sum(f * k_p_at(T) for T, f in schedule))
    k_l_eff = float(sum(f * k_l_at(T) for T, f in schedule))

    # Validity check: duty-cycle averaging is exact only when one cycle
    # is short compared with the growth timescale τ_g ≈ x_ss / k_l_eff.
    # If cycle_period is provided and is comparable to τ_g, warn that
    # the "averaged-rate" approximation breaks down.
    if cycle_period is not None and k_l_eff > 0 and k_p_eff > 0:
        x_ss_eff = k_p_eff / (2.0 * k_l_eff)
        tau_g = x_ss_eff / k_l_eff
        if cycle_period > 0.1 * tau_g:
            import warnings
            warnings.warn(
                f"integrate_tgo_temperature_schedule: cycle_period "
                f"({cycle_period:.3g} s) is not ≪ growth timescale "
                f"({tau_g:.3g} s). Duty-cycle averaging assumes a clear "
                f"separation; consider piecewise integration instead.",
                UserWarning, stacklevel=2,
            )

    sol = solve_paralinear((0.0, t_total), k_p_eff, k_l_eff,
                            x0=x0, n_points=n_points, PBR=PBR)
    sol["effective_k_p"] = k_p_eff
    sol["effective_k_l"] = k_l_eff
    sol["x_ss_at_each_T"] = np.array([
        k_p_at(T) / (2.0 * k_l_at(T)) if k_l_at(T) > 0 else np.inf
        for T in Ts
    ])
    return sol


def robinson_smialek_recession(T_K: float, P_H2O: float,
                                P_tot: float, v_gas: float,
                                Ea_J: float = 108e3,
                                k0: float | None = None,
                                warn_out_of_domain: bool = True) -> float:
    """k_l ∝ v^{0.5} * P_H2O^2 * P_tot^{-0.5} * exp(-ΔQ/RT).

    The default `k0` is back-solved so the correlation reproduces the
    Robinson–Smialek calibration anchor (`tebc.constants.RS_*`).

    `warn_out_of_domain` (default True) emits a `UserWarning` when any
    of (v_gas, P_H2O, T_K) exceeds 5× / drops below 0.2× the calibration
    anchors. Beyond that range the v^0.5 dependence is empirically
    questionable (boundary-layer regime change, droplet impact, etc.).
    """
    from tebc.constants import (
        RS_K_L_REF,
        RS_P_H2O_REF_PA,
        RS_T_REF_K,
        RS_V_GAS_REF,
        atm_Pa,
    )
    if warn_out_of_domain:
        import warnings
        msgs = []
        if v_gas > 5.0 * RS_V_GAS_REF or v_gas < 0.2 * RS_V_GAS_REF:
            msgs.append(
                f"v_gas={v_gas:.3g} m/s is outside the Opila/Robinson–Smialek "
                f"calibration window of ~{RS_V_GAS_REF} m/s "
                f"(v^0.5 dependence is empirical; out-of-domain behaviour is "
                f"not characterised).",
            )
        if P_H2O > 5.0 * RS_P_H2O_REF_PA or P_H2O < 0.2 * RS_P_H2O_REF_PA:
            msgs.append(
                f"P_H2O={P_H2O:.3g} Pa is outside the calibration window "
                f"of ~{RS_P_H2O_REF_PA:.3g} Pa.",
            )
        if T_K > RS_T_REF_K + 200 or T_K < RS_T_REF_K - 300:
            msgs.append(
                f"T={T_K:.0f} K is outside the {RS_T_REF_K-300:.0f}–"
                f"{RS_T_REF_K+200:.0f} K window of the calibration.",
            )
        if msgs:
            warnings.warn(
                "Robinson–Smialek extrapolated outside its calibration "
                "domain: " + "; ".join(msgs),
                UserWarning, stacklevel=2,
            )

    if k0 is None:
        k0 = RS_K_L_REF / (
            RS_V_GAS_REF**0.5
            * RS_P_H2O_REF_PA**2
            * atm_Pa**(-0.5)
            * np.exp(-Ea_J / (R_gas * RS_T_REF_K))
        )
    return k0 * v_gas**0.5 * P_H2O**2 * P_tot**(-0.5) * np.exp(-Ea_J/(R_gas*T_K))
