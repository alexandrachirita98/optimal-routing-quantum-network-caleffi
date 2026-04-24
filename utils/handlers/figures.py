"""
Figures — replication of Caleffi (2017) paper figures.

Each method produces one figure from the numerical-results section
(Section V) of "Optimal Routing for Quantum Networks", IEEE Access 2017.
"""

from __future__ import annotations

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace

from utils.constants.physical_constants import PARAMS
from utils.models.physical_params import PhysicalParams
from utils.models.optimal_routing import OptimalRouting
from utils.models.topology import Topology

class MockTopologies:
    """Helper class to generate simple topologies for testing and plotting."""
    # Helper to create a simple two-node topology with a single link of given length.
    @staticmethod
    def _line_topology(d_m: float) -> Topology:
        return Topology(nodes=["vi", "vj"], edges=[("vi", "vj", d_m)])
    
    @staticmethod
    def _two_hop_topology(d_ik: float, d_kj: float) -> Topology:
        return Topology(nodes=["vi", "vk", "vj"],
                        edges=[("vi", "vk", d_ik), ("vk", "vj", d_kj)])

class Figures:

    def __init__(self, params: PhysicalParams = PARAMS) -> None:
        self.params  = params
        self.routing = OptimalRouting(params)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _conv_rate(self, d_m: float) -> float:
        """
        Conventional link entanglement rate [pairs/s] (paper baseline):

            Rate_conv = p_{i,j} / (d_{i,j}/c_f + tau)

        with tau = 100 μs and nu_o = 1 (ideal optical BSM).
        """
        p   = self.params
        tau = 100e-6
        P_ij = p.p() ** 2 * math.exp(-d_m / p.l0)   # nu_o = 1
        return P_ij / (d_m / p.cf + tau)

    # ── Figures ───────────────────────────────────────────────────────────────

    def figure4(
        self,
        show: bool = True,
        save_path: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Figure 4: Expected Link Entanglement Rate xi_{i,j}(T^ch) vs link length
        d_{i,j} for different atom cooling times tau_d. Decoherence time
        T^ch = 10 ms (fixed). Logarithmic y-axis.
        """
        distances_km = np.linspace(0.1, 200, 2000)
        distances_m  = distances_km * 1_000
        
        tau_d_values = [10e-6, 100e-6, 1e-3, 100e-3]
        labels = [
            r"Eq. (8), $\tau^d = 10\,\mu s$",
            r"Eq. (8), $\tau^d = 100\,\mu s$",
            r"Eq. (8), $\tau^d = 1\,ms$",
            r"Eq. (8), $\tau^d = 100\,ms$",
        ]
        colors = ["#1565C0", "#E65100", "#F9A825", "#6A1B9A"]

        fig, ax = plt.subplots(figsize=(8, 6))

        for tau_d, label, color in zip(tau_d_values, labels, colors):
            r_v  = OptimalRouting(replace(self.params, tau_d=tau_d))
            rates = np.array([
                r_v.xi(["vi", "vj"], MockTopologies._line_topology(d)) for d in distances_m
            ])
            mask = rates > 0
            ax.semilogy(distances_km[mask], rates[mask], "--",
                        color=color, linewidth=1.8, label=label)

        conv = np.array([self._conv_rate(d) for d in distances_m])
        mask_c = conv > 0
        ax.semilogy(distances_km[mask_c], conv[mask_c], ":",
                    color="red", linewidth=1.8, label="Conventional Rate")

        ax.set_xlabel(r"link length $d_{i,j}$ [Km]", fontsize=12)
        ax.set_ylabel(r"Link Entanglement Rate $\xi_{i,j}(T^{CH})$", fontsize=12)
        ax.set_title("Figure 4 – Link Entanglement Rate vs Link Length", fontsize=12)
        ax.set_xlim(0, 200)
        ax.set_ylim(1e-6, 1e4)
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, which="both", linestyle=":", alpha=0.4)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        if show:
            plt.show()
        return fig, ax

    def figure5(
        self,
        show: bool = True,
        save_path: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Figure 5: Expected End-to-End Entanglement Rate xi_{r_{i,j}}(T^ch)
        vs total route length d_{i,k} + d_{k,j} for a 2-hop route
        r_{i,j} = {e_{i,k}, e_{k,j}}, varying the repeater position.

        Parameters fixed to paper values: tau_d = 100 us, T^ch = 1 ms.
        """
        params5  = replace(self.params, tau_d=100e-6, t_ch=1e-3)
        routing5 = OptimalRouting(params5)

        total_km = np.linspace(0.1, 300, 1500)
        total_m  = total_km * 1_000

        # Each entry: (fraction of D assigned to d_{i,k}, label, color, linestyle)
        # fraction=None → single direct link; "lim" → d_{i,k} → 0
        configs = [
            (None,  r"single link $d_{i,j} = d_{i,k} + d_{k,j}$",              "#1565C0", "-"),
            (0.5,   r"$d_{i,k} = d_{k,j}$",                                     "#E65100", "--"),
            (1/3,   r"$d_{i,k} = d_{k,j}/2$",                                   "#B71C1C", "--"),
            (0.2,   r"$d_{i,k} = d_{k,j}/4$",                                   "#6A1B9A", "--"),
            ("lim", r"$d_{i,k} = \lim_{n\to\infty} d_{k,j}/n$",                "#2E7D32", "--"),
        ]

        fig, ax = plt.subplots(figsize=(9, 7))

        for frac, label, color, ls in configs:
            rates = []
            for D in total_m:
                if frac is None:
                    topo = MockTopologies._line_topology(D)
                    r = routing5.xi(["vi", "vj"], topo)
                elif frac == "lim":
                    eps   = max(D * 0.001, 1.0)          # d_{i,k} → 0
                    topo  = MockTopologies._two_hop_topology(eps, D - eps)
                    r = routing5.xi(["vi", "vk", "vj"], topo)
                else:
                    d_ik  = frac * D
                    d_kj  = (1 - frac) * D
                    topo  = MockTopologies._two_hop_topology(d_ik, d_kj)
                    r = routing5.xi(["vi", "vk", "vj"], topo)
                rates.append(r)

            rates = np.array(rates)
            mask  = rates > 0
            if mask.any():
                ax.semilogy(total_km[mask], rates[mask], ls,
                            color=color, linewidth=1.8, label=label)

        # Conventional rate — midpoint repeater (d_{i,k} = d_{k,j} = D/2)
        # P_e2e = p^4 * exp(-D/l0),  T = D/cf + 2*tau  (nu_o = 1)
        tau_conv = 100e-6
        p = params5
        conv = np.array([
            p.p() ** 4 * math.exp(-D / p.l0) / (D / p.cf + 2 * tau_conv)
            for D in total_m
        ])
        ax.semilogy(total_km, conv, ":", color="red", linewidth=1.8,
                    label=r"Conventional Rate, $d_{i,k} = d_{k,j}$")

        ax.set_xlabel(r"route length $d_{i,k} + d_{k,j}$ [Km]", fontsize=12)
        ax.set_ylabel(r"End-to-End Entanglement Rate $\xi_{r_{i,j}}(T^{CH})$", fontsize=12)
        ax.set_title("Figure 5 – E2E Rate vs Route Length (2-hop, varying repeater position)",
                     fontsize=12)
        ax.set_xlim(0, 300)
        ax.set_ylim(1e-3, 1e4)
        ax.legend(fontsize=10)
        ax.grid(True, which="both", linestyle=":", alpha=0.4)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        if show:
            plt.show()
        return fig, ax

    def figure6(
        self,
        show: bool = True,
        save_path: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Figure 6: End-to-end entanglement rate vs total distance for linear chains
        with different numbers of hops (equal-length links). Shows when adding
        repeater nodes improves performance over a direct link.
        """
        pass

    def figure8(
        self,
        show: bool = True,
        save_path: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Figure 8: Optimal routing on the asymmetric topology — direct long link
        vs 3-hop chain with fixed 20-km segments.  Plots both xi values as a
        function of the direct-link distance and marks the crossover where the
        optimal path switches.
        """
        pass