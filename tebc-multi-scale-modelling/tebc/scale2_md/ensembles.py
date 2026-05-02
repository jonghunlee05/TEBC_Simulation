"""
MD ensemble implementations:
  NVE  : velocity-Verlet integrator (microcanonical)
  NVT  : Nosé-Hoover chain thermostat (canonical)
  NPT  : Parrinello-Rahman barostat + NHC thermostat
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class NoseHooverChain:
    """Nosé-Hoover chain thermostat (Martyna-Klein-Tuckerman 1992)."""
    T_target: float
    n_atoms:  int
    n_dof:    int = None
    tau:      float = 0.1e-12
    M:        int   = 3

    def __post_init__(self):
        from tebc.constants import k_B
        self.k_B = k_B
        if self.n_dof is None:
            self.n_dof = 3 * self.n_atoms
        g = self.n_dof
        Q1 = g * k_B * self.T_target * self.tau**2
        Qk = k_B * self.T_target * self.tau**2
        self.Q = np.array([Q1] + [Qk]*(self.M-1))
        self.xi   = np.zeros(self.M)
        self.p_xi = np.zeros(self.M)

    def conserved_quantity(self, KE: float) -> float:
        g = self.n_dof
        return (KE
                + np.sum(self.p_xi**2 / (2*self.Q))
                + g * self.k_B * self.T_target * self.xi[0]
                + self.k_B * self.T_target * np.sum(self.xi[1:]))

    def step(self, KE_atoms: float, dt: float) -> float:
        """Yoshida-Suzuki integration of NHC. Returns scaling factor s."""
        g = self.n_dof
        kBT = self.k_B * self.T_target
        G = np.zeros(self.M)
        G[0] = 2*KE_atoms - g*kBT
        for k in range(1, self.M):
            G[k] = self.p_xi[k-1]**2 / self.Q[k-1] - kBT
        s = 1.0
        for k in range(self.M-1, -1, -1):
            self.p_xi[k] += 0.5*dt * G[k]
        s = np.exp(-0.5*dt * self.p_xi[0] / self.Q[0])
        for k in range(self.M):
            self.xi[k] += dt * self.p_xi[k] / self.Q[k]
        G[0] = 2*KE_atoms*s**2 - g*kBT
        self.p_xi[0] += 0.5*dt * G[0]
        for k in range(1, self.M):
            G[k] = self.p_xi[k-1]**2 / self.Q[k-1] - kBT
            self.p_xi[k] += 0.5*dt * G[k]
        return s


def velocity_verlet_step(pos, vel, forces, masses, dt):
    """Velocity-Verlet integrator (NVE or any constant-force ensemble)."""
    acc   = forces / masses[:, None]
    pos_new = pos + vel*dt + 0.5*acc*dt**2
    vel_half = vel + 0.5*acc*dt
    return pos_new, vel_half


def parrinello_rahman_step(h, h_dot, stress_int, stress_ext, W, dt):
    """
    Parrinello-Rahman barostat equation of motion:
      W ḧ = V (σ_int - p_ext I)(h^T)⁻¹
    """
    V   = np.abs(np.linalg.det(h))
    hT_inv = np.linalg.inv(h.T)
    P_diff = stress_int - stress_ext * np.eye(3)
    h_ddot = V * P_diff @ hT_inv / W
    h_new     = h     + h_dot*dt     + 0.5*h_ddot*dt**2
    h_dot_new = h_dot + 0.5*h_ddot*dt
    return h_new, h_dot_new
