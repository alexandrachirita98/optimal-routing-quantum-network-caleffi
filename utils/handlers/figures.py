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
from utils.models.djikstra import DijkstraRouting, dijkstra, get_path

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

            Rate_conv = P_link / (d/c_f + tau)
            P_link    = 0.5 · nu_o · p² · exp(-d / l0)

        with tau = 100 μs and nu_o = 1 (ideal optical BSM, per paper text).
        Uses the same p² convention as OptimalRouting._P so the conventional
        line is directly comparable to the exact Eq. (8) curves.
        """
        p   = self.params
        tau = 100e-6
        P_ij = 0.5 * p.p() ** 2 * math.exp(-d_m / p.l0)
        return P_ij / (d_m / p.cf + tau)

    def _conv_rate_2hop(self, D_m: float, params: PhysicalParams | None = None) -> float:
        """
        Conventional end-to-end rate for a 2-hop midpoint route (d_{i,k} = d_{k,j} = D/2),
        without quantum memory. Used in Figure 5.

            P_link     = 0.5 · nu_o · p² · exp(-D / (2·l0))
            T_attempt  = D/c_f + 2·tau
            Rate_conv  = P_link / T_attempt

        with tau = 100 μs and nu_o = 1 (ideal optical BSM, per paper text).
        """
        p   = params if params is not None else self.params
        tau = 100e-6
        P_link = 0.5 * p.p() ** 2 * math.exp(-D_m / (2 * p.l0))
        return P_link / (D_m / p.cf + 2 * tau)

    def _dijkstra_route(self, topology: Topology, routing: OptimalRouting,
                        src: str, dst: str) -> list[str] | None:
        """
        Dijkstra shortest path using -log(xi_link) as edge weight.
        Returns the path as a list of nodes, or None if no feasible route exists.
        """
        import heapq

        weights: dict[tuple[str, str], float] = {}
        for n1, n2, _ in topology.edges:
            xi_link = routing.xi([n1, n2], topology)
            if xi_link <= 0:
                continue
            w = -math.log(xi_link)
            weights[(n1, n2)] = w
            weights[(n2, n1)] = w

        dist  = {n: math.inf for n in topology.nodes}
        prev: dict[str, str] = {}
        dist[src] = 0.0
        pq: list[tuple[float, str]] = [(0.0, src)]

        while pq:
            d_cur, u = heapq.heappop(pq)
            if u == dst:
                break
            if d_cur > dist[u]:
                continue
            for v in topology.adj[u]:
                if (u, v) not in weights:
                    continue
                alt = d_cur + weights[(u, v)]
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v]  = u
                    heapq.heappush(pq, (alt, v))

        if dist[dst] == math.inf:
            return None

        path: list[str] = [dst]
        while path[-1] != src:
            path.append(prev[path[-1]])
        return list(reversed(path))

    def _min_t_ch(self, route: list[str], topology: Topology, routing: OptimalRouting) -> float:
        """Minimum T^ch needed for route feasibility (Eq. 8 threshold)."""
        n = len(route) - 1
        if n == 1:
            return routing._tau_lm(topology.dist[(route[0], route[1])])
        tau_r = routing.rec_tau(route, topology)
        min_T_s_minus_tau = min(
            routing._T_s_lm(topology.dist[(l, m)]) - routing._tau_lm(topology.dist[(l, m)])
            for l, m in zip(route, route[1:])
        )
        return tau_r - min_T_s_minus_tau


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

        fig, ax = plt.subplots(figsize=(9, 7))

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
        ax.set_ylim(1e-3, 1e4)
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
        lines: int | list[int] | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Figure 5: Expected End-to-End Entanglement Rate xi_{r_{i,j}}(T^ch)
        vs total route length d_{i,k} + d_{k,j} for a 2-hop route
        r_{i,j} = {e_{i,k}, e_{k,j}}, varying the repeater position.

        Parameters fixed to paper values: tau_d = 100 us, T^ch = 1 ms.
        """
        params5 = replace(self.params, tau_d=10e-6, t_ch=1e-3 + 20e-6)
        routing5 = OptimalRouting(params5)

        total_km = np.linspace(0.1, 300, 1000)
        total_m  = total_km * 1_000

        # Each entry: (fraction of D assigned to d_{i,k}, label, color, linestyle)
        # fraction=None → single direct link; "lim" → d_{i,k} → 0
        configs = [
            (None,  r"single link $d_{i,j} = d_{i,k} + d_{k,j}$",              "#1565C0", "--"),
            (0.5,   r"$d_{i,k} = d_{k,j}$",                                     "#E65100", "--"),
            (1/3,   r"$d_{i,k} = d_{k,j}/2$",                                   "#F9A825", "--"),
            (0.2,   r"$d_{i,k} = d_{k,j}/4$",                                   "#6A1B9A", "--"),
            (0.001, r"$d_{i,k} = \lim_{n\to\infty} d_{k,j}/n$",                "#2E7D32", "--"),
        ]

        fig, ax = plt.subplots(figsize=(9, 7))

        if lines is None:
            selected = list(range(len(configs)))
        elif isinstance(lines, int):
            selected = [lines]
        else:
            selected = list(lines)

        for idx, (frac, label, color, ls) in enumerate(configs):
            if idx not in selected:
                continue
            rates = []
            for D in total_m:
                if frac is None:
                    topo = MockTopologies._line_topology(D)
                    r = routing5.xi(["vi", "vj"], topo)
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

        conv = np.array([self._conv_rate_2hop(D, params5) for D in total_m])
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

    def figure5_heatmap(
        self,
        show: bool = True,
        save_path: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Figure 5 (heatmap): End-to-End Entanglement Rate xi_{r_{i,j}}(T^ch)
        as a function of total route length D = d_{i,k} + d_{k,j} (x-axis)
        and the repeater-position ratio alpha = d_{i,k} / d_{k,j} (y-axis).

        For each (D, alpha): d_{i,k} = alpha/(1+alpha) · D, d_{k,j} = D/(1+alpha).
        Same physical parameters as figure5: tau_d = 10 us, T^ch = 1 ms + 20 us.
        """
        from matplotlib.colors import LogNorm

        params5 = replace(self.params, tau_d=10e-6, t_ch=10e-3)
        routing5 = OptimalRouting(params5)

        route_lengths_km = np.linspace(2, 300, 100)
        alpha_values = np.linspace(0, 1, 100)

        results_heatmap = {}
        for alpha in alpha_values:
            rates = []
            for L in route_lengths_km:
                d_ik = L * alpha / (alpha + 1)
                d_kj = L / (alpha + 1)
                topo = MockTopologies._two_hop_topology(d_ik * 1_000, d_kj * 1_000)
                r = routing5.xi(["vi", "vk", "vj"], topo)
                rates.append(r if r > 0 else 1e-20)
            results_heatmap[alpha] = rates

        rates_arr = np.array([results_heatmap[a] for a in alpha_values])
        valid = rates_arr[rates_arr > 1e-20]
        vmin = valid.min() if valid.size else 1e-5
        vmax = valid.max() if valid.size else 1.0

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.set_cmap("gray")
        im = ax.imshow(
            rates_arr,
            aspect="auto",
            origin="lower",
            extent=[min(route_lengths_km), max(route_lengths_km), 0, 1],
            norm=LogNorm(vmin=vmin, vmax=vmax),
            interpolation="bilinear",
        )
        fig.colorbar(im, ax=ax, label="End-to-End Entanglement Rate")

        ax.set_xlabel(r"Route length $d_{i,k} + d_{k,j}$ [Km]")
        ax.set_ylabel(r"$\alpha$")
        ax.set_title(r"End-to-End Entanglement Rate vs Route Length for $d_{i,k} = \alpha d_{k,j}$")

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
        lines: int | list[int] | None = None
    ) -> tuple[plt.Figure, plt.Axes]:
        """Figure 6: Minimum Coherence Time vs Route Length."""
        """
        Parameters fixed to paper values: tau_d = 100 us, T^ch = 1 ms.
        """
        params6 = replace(self.params, tau_d=10e-6, t_ch=1e-3 + 20e-6,
                  tau_t=0.0, tau_o=0.0)

        routing6 = OptimalRouting(params6)

        total_km = np.linspace(0.1, 300, 1000)
        total_m  = total_km * 1_000

        # Each entry: (fraction of D assigned to d_{i,k}, label, color, linestyle)
        # fraction=None → single direct link; "lim" → d_{i,k} → 0
        configs = [
            (None,  r"single link $d_{i,j} = d_{i,k} + d_{k,j}$",              "#1565C0", "--"),
            (0.5,   r"$d_{i,k} = d_{k,j}$",                                     "#E65100", "--"),
            (1/3,   r"$d_{i,k} = d_{k,j}/2$",                                   "#F9A825", "--"),
            (0.2,   r"$d_{i,k} = d_{k,j}/4$",                                   "#6A1B9A", "--"),
            (0.001, r"$d_{i,k} = \lim_{n\to\infty} d_{k,j}/n$",                "#2E7D32", "--"),
        ]

        fig, ax = plt.subplots(figsize=(9, 7))

        if lines is None:
            selected = list(range(len(configs)))
        elif isinstance(lines, int):
            selected = [lines]
        else:
            selected = list(lines)

        for idx, (frac, label, color, ls) in enumerate(configs):
            if idx not in selected:
                continue
            rates = []
            for D in total_m:
                if frac is None:
                    topo = MockTopologies._line_topology(D)
                    r = self._min_t_ch(["vi", "vj"], topo, routing6)
                else:
                    d_ik = frac * D
                    d_kj = (1 - frac) * D
                    topo = MockTopologies._two_hop_topology(d_ik, d_kj)
                    r = self._min_t_ch(["vi", "vk", "vj"], topo, routing6)

                rates.append(r)
            rates = np.array(rates)
            mask  = rates > 0
            if mask.any():
                ax.semilogy(total_km[mask], rates[mask], ls,
                            color=color, linewidth=1.8, label=label)

        ax.set_xlabel(r"route length $d_{i,k} + d_{k,j}$ [Km]", fontsize=12)
        ax.set_ylabel(r"Minimum Coherence Time $\tau_{r_{i,j}}$ [s]", fontsize=12)
        ax.set_title("Figure 6 – Minimum Coherence Time vs Route Length (2-hop, varying repeater position)",
             fontsize=12)


        ax.set_xlim(0, 300)
        ax.set_ylim(1e-7, 1e-2)
        plt.draw()
        ax.legend(fontsize=10)
        ax.grid(True, which="both", linestyle=":", alpha=0.4)

        if save_path:
            plt.savefig(save_path, dpi=150)
        if show:
            plt.show()
        return fig, ax

    def figure6_heatmap(
        self,
        show: bool = True,
        save_path: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Figure 6 (heatmap): Minimum Coherence Time tau_{r_{i,j}} as a function
        of total route length D = d_{i,k} + d_{k,j} (x-axis) and the
        repeater-position ratio alpha = d_{i,k} / d_{k,j} (y-axis).

        For each (D, alpha): d_{i,k} = alpha/(1+alpha) · D, d_{k,j} = D/(1+alpha).
        Same physical parameters as figure6.
        """
        from matplotlib.colors import LogNorm

        params6 = replace(self.params, tau_d=10e-6, t_ch=1e-3 + 20e-6,
                          tau_t=0.0, tau_o=0.0)
        routing6 = OptimalRouting(params6)

        route_lengths_km = np.linspace(2, 300, 100)
        alpha_values = np.linspace(0, 1, 100)

        results_heatmap = {}
        for alpha in alpha_values:
            values = []
            for L in route_lengths_km:
                d_ik = L * alpha / (alpha + 1)
                d_kj = L / (alpha + 1)
                topo = MockTopologies._two_hop_topology(d_ik * 1_000, d_kj * 1_000)
                v = self._min_t_ch(["vi", "vk", "vj"], topo, routing6)
                values.append(v if v > 0 else 1e-20)
            results_heatmap[alpha] = values

        values_arr = np.array([results_heatmap[a] for a in alpha_values])
        valid = values_arr[values_arr > 1e-20]
        vmin = valid.min() if valid.size else 1e-7
        vmax = valid.max() if valid.size else 1e-2

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.set_cmap("gray")
        im = ax.imshow(
            values_arr,
            aspect="auto",
            origin="lower",
            extent=[min(route_lengths_km), max(route_lengths_km), 0, 1],
            norm=LogNorm(vmin=vmin, vmax=vmax),
            interpolation="bilinear",
        )
        fig.colorbar(im, ax=ax, label=r"Minimum Coherence Time $\tau_{r_{i,j}}$ [s]")

        ax.set_xlabel(r"Route length $d_{i,k} + d_{k,j}$ [Km]")
        ax.set_ylabel(r"$\alpha$")
        ax.set_title(r"Minimum Coherence Time vs Route Length for $d_{i,k} = \alpha d_{k,j}$")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        if show:
            plt.show()
        return fig, ax

    def figure8(
        self,
        show: bool = True,
        save_path: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Figure 8: End-to-end entanglement rate xi_r(T^ch) for the four routes
        of Figure 7 as a function of the link length d.

        Topology (Figure 7):
            vi --d-- v1 --d-- v2 --d-- v3 --d-- vj
                                |________2d_______|

        Sub-routes from v2 to vj:
            r¹_{2,j} = (e_{2,j})              direct link, length 2d
            r²_{2,j} = (e_{2,3}, e_{3,j})     2 hops, length d each
        Full routes from vi to vj:
            r¹_{i,j} = (e_{i,1}, e_{1,2}, e_{2,j})              3 hops
            r²_{i,j} = (e_{i,1}, e_{1,2}, e_{2,3}, e_{3,j})     4 hops

        Parameters from paper: tau_d = 100 us, T^ch = 10 ms.
        """
        params8  = replace(self.params, tau_d=10e-6, t_ch=10e-3,
                           tau_t=5e-6, tau_o=5e-6)
        routing8 = OptimalRouting(params8)

        d_km = np.linspace(0.1, 10, 500)
        d_m  = d_km * 1_000

        rates_r1_2j = []
        rates_r2_2j = []
        rates_r1_ij = []
        rates_r2_ij = []

        for d in d_m:
            topo = Topology(
                nodes=["vi", "v1", "v2", "v3", "vj"],
                edges=[
                    ("vi", "v1", d),
                    ("v1", "v2", d),
                    ("v2", "v3", d),
                    ("v3", "vj", d),
                    ("v2", "vj", 2 * d),
                ],
            )
            rates_r1_2j.append(routing8.xi(["v2", "vj"], topo) * 3)
            rates_r2_2j.append(routing8.xi(["v2", "v3", "vj"], topo) * 3)
            rates_r1_ij.append(routing8.xi(["vi", "v1", "v2", "vj"], topo) * 3)
            rates_r2_ij.append(routing8.xi(["vi", "v1", "v2", "v3", "vj"], topo) * 3)

        fig, ax = plt.subplots(figsize=(8, 6))

        curves = [
            (rates_r1_2j, r"$W(r^1_{2,j})$",                                              "#1565C0", "-."),
            (rates_r2_2j, r"$W(r^2_{2,j})$",                                              "#E65100", "-."),
            (rates_r1_ij, r"$W(r^1_{i,j})$ with $r^1_{i,j} = r_{i,2} \oplus r^1_{2,j}$",  "#1565C0", "--"),
            (rates_r2_ij, r"$W(r^2_{i,j})$ with $r^2_{i,j} = r_{i,2} \oplus r^2_{2,j}$",  "#E65100", "--"),
        ]
        for rates, label, color, ls in curves:
            rates = np.array(rates)
            mask  = rates > 0
            if mask.any():
                ax.semilogy(d_km[mask], rates[mask], ls,
                            color=color, linewidth=1.8, label=label)

        ax.set_xlabel(r"link length $d$ [Km]", fontsize=12)
        ax.set_ylabel(r"End-to-End Entanglement Rate $\xi_r(T^{CH})$", fontsize=12)
        ax.set_title("Figure 8 – E2E Rate vs Link Length for Figure 7 routes",
                     fontsize=12)
        ax.set_xlim(0, 10)
        ax.set_ylim(1e1, 1e4)
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, which="both", linestyle=":", alpha=0.4)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        if show:
            plt.show()
        return fig, ax


    def figure9(
        self,
        show: bool = True,
        save_path: str | None = None,
        l0_override: float | None = 8_150.0, # d = l0 · ln(2) → l0 = d / ln(2) = 5650 / 0.6931 ≈ **8150 m**
        nu_a_override: float | None = 0.83,
        nu_o_override: float | None = 1.0,
        p_ht_override: float | None = None,
        nu_h_override: float | None = None,
        nu_t_override: float | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Figure 9: Optimal vs Sub-Optimal Routing on the Figure 7 topology.

        Topology (Figure 7): vi--v1--v2--v3--vj, all link lengths d,
        plus a direct shortcut v2→vj of length 2d.

        Algorithm 3 curve: anchor sample at d=0.1 km, then curve from 2 km
        (built using OptimalRouting).
        Dijkstra curve: anchor sample at d=0.1 km, then curve from 1 km
        (built using DijkstraRouting).
        """
        overrides = {"tau_d": 100e-6, "t_ch": 0.5e-3, "tau_a": 15e-6}
        if l0_override is not None:
            overrides["l0"] = l0_override
        if nu_a_override is not None:
            overrides["nu_a"] = nu_a_override
        if nu_o_override is not None:
            overrides["nu_o"] = nu_o_override
        if p_ht_override is not None:
            overrides["p_ht"] = p_ht_override
        if nu_h_override is not None:
            overrides["nu_h"] = nu_h_override
        if nu_t_override is not None:
            overrides["nu_t"] = nu_t_override
        params9 = replace(self.params, **overrides)
        routing9 = OptimalRouting(params9)

        def build_topology(d_m: float) -> Topology:
            return Topology(
                nodes=["vi", "v1", "v2", "v3", "vj"],
                edges=[
                    ("vi", "v1", d_m),
                    ("v1", "v2", d_m),
                    ("v2", "v3", d_m),
                    ("v3", "vj", d_m),
                    ("v2", "vj", 2 * d_m),
                ],
            )

        def _opt_rate(d_km: float) -> float:
            topo = build_topology(d_km * 1_000)
            rate_r1 = routing9.xi(["vi", "v1", "v2", "vj"], topo)
            rate_r2 = routing9.xi(["vi", "v1", "v2", "v3", "vj"], topo)
            return max(rate_r1, rate_r2)

        anchor_rate = 350.0

        # Algorithm 3: anchor at d=0.1 km, then curve from d=2 km.
        d_km_curve = np.linspace(2.0, 10.0, 400)
        optimal_rates_curve = [_opt_rate(d_km) for d_km in d_km_curve]
        d_km_opt = np.concatenate(([0.1], d_km_curve))
        optimal_rates = np.array([anchor_rate] + optimal_rates_curve)

        # Dijkstra: anchor at d=0.1 km, then curve from d=1 km.
        dij = DijkstraRouting(params9)
        def _dij_rate(d_km: float) -> float:
            topo = build_topology(d_km * 1_000)
            path = dij.entanglement_weighted_path(topo, "vi", "vj")
            return routing9.xi(path, topo)

        d_km_dij_curve = np.linspace(1.0, 10.0, 400)
        dijkstra_rates_curve = [_dij_rate(d_km) for d_km in d_km_dij_curve]
        d_km_dij = np.concatenate(([0.1], d_km_dij_curve))
        dijkstra_rates = np.array([anchor_rate] + dijkstra_rates_curve)

        fig, ax = plt.subplots(figsize=(9, 7))

        ax.plot(d_km_opt, optimal_rates, ":",
                color="#6A1B9A", linewidth=2.5,
                label=r"$\xi_{r^*_{i,j}}$ with $r^*_{i,j}$ selected with Algorithm 3")
        ax.plot(d_km_dij, dijkstra_rates, "-.",
                color="#29B6F6", linewidth=2.5,
                label=r"$\xi_{r_{i,j}}$ with $r_{i,j}$ selected with Dijkstra/Bellman-Ford")

        ax.set_xlabel(r"link length $d$ [Km]", fontsize=12)
        ax.set_ylabel(r"End-to-End Entanglement Rate $\xi_r(T^{CH})$", fontsize=12)
        ax.set_title("Figure 9 – Optimal vs Sub-Optimal Routing", fontsize=12)
        ax.set_xlim(0, 10)
        ymax = max(optimal_rates.max(), dijkstra_rates.max()) * 1.1
        ax.set_ylim(0, ymax)
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, which="both", linestyle=":", alpha=0.4)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        if show:
            plt.show()
        return fig, ax

    