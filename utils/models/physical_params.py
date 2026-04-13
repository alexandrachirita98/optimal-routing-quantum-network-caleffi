"""
PhysicalParams — hardware parameter dataclass for the quantum network (Caleffi 2017).

The paper's reference values live in utils.constants.physical_constants as PARAMS.
"""

from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass
class PhysicalParams:
    # ── Local entanglement (Eq. 1) ──
    p_ht: float   # photon pair generation probability
    nu_h: float   # heralding detector efficiency
    nu_t: float   # telecom detector efficiency

    # ── Optical BSM ──
    nu_o: float   # optical BSM efficiency

    # ── Fiber ──
    l0: float     # attenuation length  [m]
    cf: float     # speed of light in fiber  [m/s]

    # ── Timing ──
    tau_p: float  # atom excitation pulse  [s]
    tau_h: float  # heralding-cavity coupling time  [s]
    tau_t: float  # telecom-cavity coupling time  [s]
    tau_d: float  # atom cooling time  [s]
    tau_o: float  # optical processing time  [s]

    # ── Entanglement swapping ──
    tau_a: float  # swapping BSM duration  [s]
    nu_a: float   # swapping BSM success probability

    # ── Quantum memory ──
    t_ch: float   # coherence time  [s]

    def p(self) -> float:
        """Local entanglement generation probability per attempt (Eq. 1):  p = sqrt(p_ht · ν_h · ν_t)."""
        return math.sqrt(self.p_ht * self.nu_h * self.nu_t)

    def attempt_frequency(self) -> float:
        """Maximum memory excitation rate [Hz]:  1 / (τ_p + τ_h + τ_t + τ_d)."""
        return 1.0 / (self.tau_p + self.tau_h + self.tau_t + self.tau_d)

    def fiber_attenuation(self) -> float:
        """Photon loss coefficient per metre:  1 / l₀."""
        return 1.0 / self.l0
