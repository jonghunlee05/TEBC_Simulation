"""
Deep Neural Network Potential (DeePMD) training wrapper.

Architecture: DeepPot-SE smooth descriptor + fitting net
Loss:  L = (p_e/N)|ΔE|² + (p_f/3N)Σ|ΔF_i|² + (p_v/9N)|ΔΞ|²
"""

from __future__ import annotations

import json
import subprocess

import numpy as np


def build_deepmd_input(
    type_map: list,
    r_cut: float      = 7.0,
    r_cut_smth: float = 2.0,
    sel: list    = [60, 40, 20],
    neuron_embed: list = [25, 50, 100],
    axis_neuron: int        = 16,
    neuron_fit: list   = [240, 240, 240],
    n_training: int         = 1_000_000,
    batch_size: int         = 32,
    start_lr: float         = 1e-3,
    stop_lr:  float         = 3.51e-8,
    decay_steps: int        = 5000,
    pref_e: float = 1.0,
    pref_f: float = 1.0,
    pref_v: float = 0.02,
    output_path: str = "input_deepmd.json",
) -> dict:
    """Generate DeePMD-kit v3 JSON input."""
    cfg = {
        "model": {
            "type_map": type_map,
            "descriptor": {
                "type": "se_e2_a",
                "rcut": r_cut,
                "rcut_smth": r_cut_smth,
                "sel": sel,
                "neuron": neuron_embed,
                "axis_neuron": axis_neuron,
                "resnet_dt": False,
                "seed": 1,
            },
            "fitting_net": {
                "neuron": neuron_fit,
                "resnet_dt": False,
                "seed": 1,
            },
        },
        "learning_rate": {
            "type": "exp",
            "decay_steps": decay_steps,
            "start_lr": start_lr,
            "stop_lr": stop_lr,
        },
        "loss": {
            "type": "ener",
            "start_pref_e": 0.02, "limit_pref_e": pref_e,
            "start_pref_f": 1000, "limit_pref_f": pref_f,
            "start_pref_v": 0.0,  "limit_pref_v": pref_v,
        },
        "training": {
            "training_data": {"systems": ["./data/train"], "batch_size": batch_size},
            "validation_data": {"systems": ["./data/valid"], "batch_size": batch_size},
            "numb_steps": n_training,
            "seed": 1,
            "disp_file": "lcurve.out",
            "disp_freq": 1000,
            "save_freq": 10000,
            "save_ckpt": "model.ckpt",
        },
    }
    with open(output_path, "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg


def run_deepmd_training(input_json: str = "input_deepmd.json",
                        n_gpu: int = 1) -> None:
    """dp train / freeze / compress."""
    cmds = [
        f"dp train {input_json}",
        "dp freeze -o graph.pb",
        "dp compress -i graph.pb -o graph_compress.pb",
    ]
    for cmd in cmds:
        subprocess.run(cmd.split(), check=True)


def evaluate_model_deviation(frames: list, model_paths: list) -> dict:
    """
    Committee model deviation: σ_F = max_i √⟨|F_i^(k) - ⟨F_i⟩|²⟩_k
    Threshold: σ_lo = 0.10 eV/Å, σ_hi = 0.25 eV/Å
    """
    sigma_lo, sigma_hi = 0.10, 0.25
    try:
        from deepmd.calculator import DP
    except ImportError:
        raise ImportError("pip install deepmd-kit")

    calcs = [DP(model=p) for p in model_paths]
    sigma_F_all = []
    for frame in frames:
        forces = []
        for calc in calcs:
            atoms = frame.copy()
            atoms.calc = calc
            forces.append(atoms.get_forces())
        forces = np.stack(forces)
        mean_F = forces.mean(axis=0)
        dev_F  = np.sqrt(((forces - mean_F)**2).mean())
        sigma_F_all.append(dev_F)
    sigma_F = np.array(sigma_F_all)
    return {
        "sigma_F": sigma_F,
        "uncertain_mask": (sigma_F > sigma_lo) & (sigma_F < sigma_hi),
        "too_uncertain":  sigma_F >= sigma_hi,
    }
