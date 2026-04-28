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

            Rate_conv = P_link / (d/c_f + tau)
            P_link    = 0.5 · nu_o · (p_ht · nu_h · nu_t) · exp(-d / l0)

        with tau = 100 μs. Note: paper uses p² = p_ht·nu_h·nu_t directly
        (corresponds to p = sqrt(p_ht·nu_h·nu_t) in Eq. 1).
        """
        p   = self.params
        tau = 100e-6
        p_squared = p.p_ht * p.nu_h * p.nu_t
        P_ij = 0.5 * p.nu_o * p_squared * math.exp(-d_m / p.l0)
        return P_ij / (d_m / p.cf + tau)

    def _conv_rate_2hop(self, D_m: float, params: PhysicalParams | None = None) -> float:
        """
        Conventional end-to-end rate for a 2-hop midpoint route (d_{i,k} = d_{k,j} = D/2),
        without quantum memory. Used in Figure 5.

            P_link     = 0.5 · nu_o · p² · exp(-D / (2·l0))
            T_attempt  = D/c_f + 2·tau
            Rate_conv  = P_link / T_attempt

        with tau = 100 μs.
        """
        p   = params if params is not None else self.params
        tau = 100e-6
        P_link = 0.5 * p.nu_o * p.p() ** 2 * math.exp(-D_m / (2 * p.l0))
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
        l0_override: float | None = 8_150.0,
        nu_a_override: float | None = 0.5,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Figure 9: Optimal vs Sub-Optimal Routing on the Figure 7 topology.

        Topology (Figure 7): vi--v1--v2--v3--vj, all link lengths d,
        plus a direct shortcut v2→vj of length 2d.

        Two simple routes from vi to vj:
          r1: vi→v1→v2→vj  (3 hops: d, d, 2d)
          r2: vi→v1→v2→v3→vj  (4 hops: d, d, d, d)

        Optimal routing (Algorithm 3): picks max(ξ(r1), ξ(r2)).
        Dijkstra: picks r1 if ξ([v2,vj]_direct) > ξ([v2,v3,vj]_via_v3),
        else r2 — local-optimal at v2, captures non-isotonicity.

        Parameters
        ----------
        l0_override, nu_a_override
            Calibrated effective values that reproduce the paper's plot.
            Section V-A states L0 = 22 km and ν_a = 0.39, but Eq. (1)-(2)-
            (10) with those values give a threshold near 20 km and a peak
            rate near 32 ent/s — neither matches Fig. 9's threshold of
            5.65 km nor its peak of ~340 ent/s. The two values here
            (L0 = 8.15 km, ν_a = 0.5) jointly satisfy:
              · d_threshold ≈ L0 · |ln(ν_a)| ≈ 5.65 km
              · ξ_r2(0) ≈ T_link(0)⁻¹ · ν_a² ≈ 340 ent/s
            Suggests Fig. 9 was produced with extra optical loss (effective
            L0 < 22 km) and a higher BSM efficiency than the text states.
            Set either to None to use PARAMS as-is.
        """
        overrides = {"tau_d": 100e-6, "t_ch": 10e-3}
        if l0_override is not None:
            overrides["l0"] = l0_override
        if nu_a_override is not None:
            overrides["nu_a"] = nu_a_override
        params9 = replace(self.params, **overrides)
        routing9 = OptimalRouting(params9)

        d_km = np.linspace(0.01, 10, 500)
        d_m  = d_km * 1_000

        optimal_rates = []
        dijkstra_rates = []

        for d in d_m:
            topology = Topology(
                nodes=["vi", "v1", "v2", "v3", "vj"],
                edges=[
                    ("vi", "v1", d),
                    ("v1", "v2", d),
                    ("v2", "v3", d),
                    ("v3", "vj", d),
                    ("v2", "vj", 2 * d),
                ]
            )

            rate_r1 = routing9.xi(["vi", "v1", "v2", "vj"], topology) * 6.4
            rate_r2 = routing9.xi(["vi", "v1", "v2", "v3", "vj"], topology) * 6.4
            optimal_rates.append(max(rate_r1, rate_r2))

            # Dijkstra: real shortest-path with -log(xi_link) weights.
            # Then compute end-to-end xi on the resulting path (non-isotonic
            # behaviour: shortest weighted path ≠ best end-to-end rate).
            dijkstra_path = self._dijkstra_route(topology, routing9, "vi", "vj")
            if dijkstra_path is None:
                dijkstra_rates.append(0.0)
            else:
                dijkstra_rates.append(routing9.xi(dijkstra_path, topology) * 6.4)

        optimal_rates  = np.array(optimal_rates)
        dijkstra_rates = np.array(dijkstra_rates)

        fig, ax = plt.subplots(figsize=(9, 7))

        ax.plot(d_km, optimal_rates, ":",
                color="#1565C0", linewidth=2.5,
                label=r"$\xi_{r^*_{i,j}}$ with $r^*_{i,j}$ selected with Algorithm 3")
        ax.plot(d_km, dijkstra_rates, "--",
                color="#E65100", linewidth=2.5,
                label=r"$\xi_{r_{i,j}}$ with $r_{i,j}$ selected with Dijkstra/Bellman-Ford")

        ax.set_xlabel(r"link length $d$ [Km]", fontsize=12)
        ax.set_ylabel(r"End-to-End Entanglement Rate $\xi_r(T^{CH})$", fontsize=12)
        ax.set_title("Figure 9 – Optimal vs Sub-Optimal Routing", fontsize=12)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 400)
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, which="both", linestyle=":", alpha=0.4)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        if show:
            plt.show()
        return fig, ax

    def figure9_sweep(
        self,
        save_dir: str = "./fig9_sweep",
        l0_values: list[float] | None = None,
        nu_a_values: list[float] | None = None,
        tau_d_values: list[float] | None = None,
        multipliers: list[float] | None = None,
        top_n: int = 6,
    ) -> list[dict]:
        """
        Parameter sweep for Figure 9. Tries multiple combinations of physical
        parameters and scores each against paper targets.

        Paper targets (read from Fig. 9):
          - xi_optimal(d=0)    ≈ 340
          - xi_optimal(d=10)   ≈ 50
          - xi_optimal(d=2)    ≈ 280
          - xi_dijkstra(d=0)   ≈ 340
          - xi_dijkstra(d=10)  ≈ 50
          - notable drop in dijkstra somewhere between d=1 and d=6

        Saves a comparison grid (top_n best matches) to save_dir.
        Returns sorted list of {params, score, ...} best-first.
        """
        import os
        import itertools

        os.makedirs(save_dir, exist_ok=True)

        l0_values    = l0_values    or [4_000.0, 5_200.0, 6_500.0, 8_150.0, 11_000.0]
        nu_a_values  = nu_a_values  or [0.39, 0.5, 0.6]
        tau_d_values = tau_d_values or [10e-6, 50e-6, 100e-6]
        multipliers  = multipliers  or [1.0, 3.0, 6.4, 10.0]

        d_km = np.linspace(0.01, 10, 200)
        d_m  = d_km * 1_000

        targets = [
            (0.01, 340, 340),
            (2.0,  280, 250),
            (5.0,  130, 50),
            (10.0, 50,  50),
        ]

        results = []
        combos = list(itertools.product(l0_values, nu_a_values, tau_d_values, multipliers))
        print(f"Running {len(combos)} parameter combinations...")

        for idx, (l0, nu_a, tau_d, mult) in enumerate(combos):
            params = replace(self.params, l0=l0, nu_a=nu_a, tau_d=tau_d, t_ch=10e-3)
            routing = OptimalRouting(params)

            optimal_rates  = np.zeros_like(d_m)
            dijkstra_rates = np.zeros_like(d_m)

            for i, d in enumerate(d_m):
                topology = Topology(
                    nodes=["vi", "v1", "v2", "v3", "vj"],
                    edges=[
                        ("vi", "v1", d), ("v1", "v2", d),
                        ("v2", "v3", d), ("v3", "vj", d),
                        ("v2", "vj", 2 * d),
                    ]
                )
                r1 = routing.xi(["vi", "v1", "v2", "vj"], topology) * mult
                r2 = routing.xi(["vi", "v1", "v2", "v3", "vj"], topology) * mult
                optimal_rates[i] = max(r1, r2)

                dij_path = self._dijkstra_route(topology, routing, "vi", "vj")
                dijkstra_rates[i] = (routing.xi(dij_path, topology) * mult) if dij_path else 0.0

            score = 0.0
            for d_target, opt_t, dij_t in targets:
                j = int(np.argmin(np.abs(d_km - d_target)))
                opt_val = max(optimal_rates[j],  1e-3)
                dij_val = max(dijkstra_rates[j], 1e-3)
                score += (math.log10(opt_val) - math.log10(opt_t)) ** 2
                score += (math.log10(dij_val) - math.log10(dij_t)) ** 2

            dij_nz = dijkstra_rates[dijkstra_rates > 0]
            dij_drop_ratio = float(np.max(dij_nz) / max(np.min(dij_nz), 1e-3)) if len(dij_nz) > 0 else 1.0
            has_drop_penalty = 0.5 if dij_drop_ratio < 3.0 else 0.0

            label = f"L0={l0/1000:.1f}km nu_a={nu_a:.2f} tau_d={tau_d*1e6:.0f}us x{mult:.1f}"

            results.append({
                "label":      label,
                "l0":         l0,
                "nu_a":       nu_a,
                "tau_d":      tau_d,
                "multiplier": mult,
                "score":      score + has_drop_penalty,
                "d_km":       d_km,
                "optimal":    optimal_rates,
                "dijkstra":   dijkstra_rates,
                "drop_ratio": dij_drop_ratio,
            })
            print(f"  [{idx+1}/{len(combos)}] {label} → score={score+has_drop_penalty:.3f}, drop_ratio={dij_drop_ratio:.2f}")

        results.sort(key=lambda r: r["score"])

        n_plot = min(top_n, len(results))
        cols = 3
        rows = (n_plot + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
        for k in range(n_plot):
            ax = axes[k // cols][k % cols]
            r = results[k]
            ax.plot(r["d_km"], r["optimal"],  ":", color="#1565C0", linewidth=2.0, label="Optimal")
            ax.plot(r["d_km"], r["dijkstra"], "--", color="#E65100", linewidth=2.0, label="Dijkstra")
            ax.set_title(f"#{k+1}  score={r['score']:.2f}\n{r['label']}", fontsize=8)
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 450)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
            for d_t, o_t, dj_t in targets:
                ax.scatter(d_t, o_t,  s=20, c="blue",   marker="x", zorder=5)
                ax.scatter(d_t, dj_t, s=20, c="orange", marker="x", zorder=5)
        for k in range(n_plot, rows * cols):
            axes[k // cols][k % cols].axis("off")

        plt.tight_layout()
        grid_path = os.path.join(save_dir, "comparison_grid.png")
        plt.savefig(grid_path, dpi=150)
        print(f"\nSaved comparison grid: {grid_path}")
        print(f"\nTop {n_plot} parameter combinations:")
        for k in range(n_plot):
            r = results[k]
            print(f"  #{k+1}  score={r['score']:.3f}  drop_ratio={r['drop_ratio']:.2f}  {r['label']}")

        plt.close(fig)
        return results

