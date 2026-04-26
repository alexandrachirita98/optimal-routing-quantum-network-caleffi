"""
OptimalRouting — Algorithms 1-3 from Caleffi (2017).

Reference
---------
M. Caleffi, "Optimal Routing for Quantum Networks," IEEE Access, 2017.
"""

from __future__ import annotations
import math

from utils.models.physical_params import PhysicalParams
from utils.models.topology import Topology


class OptimalRouting:

    def __init__(self, params: PhysicalParams) -> None:
        self.params = params

    # ── Private helpers ───────────────────────────────────────────────────────

    def _P(self, distance_m: float) -> float:
        """
        Link entanglement success probability per attempt.

            P_{i,j} = p^2 * nu_o * exp(-d / l0)
        """
        p = self.params
        # p = p.p_ht * p.nu_h * p.nu_t
        return 0.5 * p.p() ** 2 * p.nu_o * math.exp(-distance_m / p.l0)

    def _T_c_lm(self, distance_m: float) -> float:
        """
        Classical communication delay for a single link [s]  (paper definition).

            T^c_{l,m} = d_{l,m} / (2 * cf)
        """
        return distance_m / (2 * self.params.cf)

    def _tau_lm(self, distance_m: float) -> float:
        """
        Time from atom-photon entanglement generation to ACK reception [s].

            tau_{l,m} = tau_t + d/(2*cf) + tau_o + T^c_{l,m}
        """
        p = self.params
        return p.tau_t + distance_m / (2 * p.cf) + p.tau_o + self._T_c_lm(distance_m)

    def _T_s_lm(self, distance_m: float) -> float:
        """
        Time for one successful entanglement attempt on a link [s].

            T^s_{i,j} = tau_p + max{tau_h, tau_{i,j}}
            tau_{i,j} = tau_t + d/(2*cf) + tau_o + T^c_{l,m}
        """
        p = self.params
        return p.tau_p + max(p.tau_h, self._tau_lm(distance_m))

    def _T_f_lm(self, distance_m: float) -> float:
        """
        Time for one failed entanglement attempt on a link [s].

            T^f_{i,j} = tau_p + max{tau_h, tau_{i,j}, tau_d}
        """
        p = self.params
        return p.tau_p + max(p.tau_h, self._tau_lm(distance_m), p.tau_d)

    def _T_lm(self, distance_m: float) -> float:
        """
        Average time to achieve link entanglement, including failed attempts [s].

            T_{i,j} = (p_bar * T^f + p * T^s) / p
        """
        p = self._P(distance_m)
        T_s = self._T_s_lm(distance_m)
        T_f = self._T_f_lm(distance_m)
        return ((1 - p) * T_f + p * T_s) / p

    def _T_c(self, route: list[str], topology: Topology) -> float:
        """
        Classical communication delay for a sub-route [s]  (Algorithm 2).

            T^c_{r_{a,k}} = sum_{e_{l,m} in r_{a,k}} T^c_{l,m}
        """
        return sum(
            self._T_c_lm(topology.dist[(l, m)])
            for l, m in zip(route, route[1:])
        )

    def _T_c_range(self, route: list[str], topology: Topology, a: int, b: int) -> float:
        """
        Classical communication delay over the sub-route route[a..b] [s].

            T^c_{r_{a,b}} = sum_{i=a}^{b-1} T^c_{route[i], route[i+1]}

        Equivalent to `_T_c(route[a:b+1], topology)` but avoids slicing.
        """
        return sum(
            self._T_c_lm(topology.dist[(route[i], route[i+1])])
            for i in range(a, b)
        )
    
    # ── Public algorithms ─────────────────────────────────────────────────────

    def xi(self, route: list[str], topology: Topology) -> float:
        """
        End-to-end entanglement rate for a route (Theorem 1, Eq. 8).

            xi(T^ch) = 0           if T^ch < rec_tau(route)
                       1 / rec_T   otherwise

        Parameters
        ----------
        route    : ordered node list, e.g. ["r1", "r2", "r3"]
        topology : network graph

        Returns
        -------
        Entanglement rate [pairs / s].  0 if route is infeasible.
        """
        n = len(route) - 1
        if n == 1:
            l, m = route[0], route[1]
            d = topology.dist[(l, m)]
            if self._tau_lm(d) <= self.params.t_ch:
                return 1.0 / self._T_lm(d)
            return 0.0
        else:
            k = math.ceil((n + 1) / 2)
            left = route[:k]
            right = route[k-1:]
            T_r_i_k = self.rec_T(left, topology)
            T_r_k_j = self.rec_T(right, topology)
            T_tilde = max(T_r_i_k, T_r_k_j)
            T_c_tilde = max(self._T_c(left, topology), self._T_c(right, topology))
            tau_r_i_k = self.rec_tau(left, topology)
            tau_r_k_j = self.rec_tau(right, topology)
            tau_tilde = max(tau_r_i_k, tau_r_k_j)
            tau_r_i_j = tau_tilde + self.params.tau_a + T_c_tilde
            min_T_s_minus_tau = min(
                self._T_s_lm(topology.dist[(l, m)]) - self._tau_lm(topology.dist[(l, m)])
                for l, m in zip(route, route[1:])
            )
            if tau_r_i_j - min_T_s_minus_tau <= self.params.t_ch:
                return 1.0 / ((T_tilde + self.params.tau_a + T_c_tilde) / self.params.nu_a)
            return 0.0

    def rec_T(self, route: list[str], topology: Topology,
          a: int | None = None, b: int | None = None) -> float:
        """
        Recursive end-to-end entanglement generation time (Algorithm 2,
        lines 1-13).

        Base case  (n = 2):  T_r = T_ij
        Recursive  (n > 2):  T_r = (max(rec_T(left), rec_T(right)) + tau_a + T_c_tilde) / nu_a

        Parameters
        ----------
        route    : ordered node list
        topology : network graph

        Returns
        -------
        Average time to generate end-to-end entanglement [s].
        """
        if a is None: a = 0
        if b is None: b = len(route) - 1

        if b - a == 1:                                       # caz de bază: o singură muchie
            return self._T_lm(topology.dist[(route[a], route[b])])

        k = math.ceil((a + b) / 2)                           # ⌈(a+b)/2⌉

        T_r_a_k   = self.rec_T(route, topology, a, k)        # recT(r_{a,k}, …)
        T_r_k_b   = self.rec_T(route, topology, k, b)        # recT(r_{k,b}, …)
        t_tilde   = max(T_r_a_k, T_r_k_b)
        t_c_tilde = max(self._T_c_range(route, topology, a, k),
                        self._T_c_range(route, topology, k, b))
        return (t_tilde + self.params.tau_a + t_c_tilde) / self.params.nu_a

    def rec_tau(self, route: list[str], topology: Topology, 
            a: int | None = None, b: int | None = None) -> float:       
        """
        Recursive minimum coherence time required by the route (Algorithm 2,
        lines 14-26).

        Base case  (n = 2):  tau_r = T^s_ij
        Recursive  (n > 2):  tau_r = max(rec_tau(left), rec_tau(right)) + tau_a + T_c_tilde

        Parameters
        ----------
        route    : ordered node list
        topology : network graph

        Returns
        -------
        Minimum coherence time t_ch needed to make the route feasible [s].
        """
        if a is None: a = 0
        if b is None: b = len(route) - 1

        if b - a == 1:                                       # caz de bază: o singură muchie
            return self._T_s_lm(topology.dist[(route[a], route[b])])

        k = math.ceil((a + b) / 2)                           # ⌈(a+b)/2⌉

        tau_r_a_k = self.rec_tau(route, topology, a, k)      # recTau(r_{a,k}, …)
        tau_r_k_b = self.rec_tau(route, topology, k, b)      # recTau(r_{k,b}, …)
        tau_tilde = max(tau_r_a_k, tau_r_k_b)
        T_c_tilde = max(self._T_c_range(route, topology, a, k),
                        self._T_c_range(route, topology, k, b))
        return tau_tilde + self.params.tau_a + T_c_tilde

    def optimal_path(self, topology: Topology) -> dict[tuple[str, str], list[str]]:
        """
        Find the route with the highest xi between every node pair (Algorithm 3).

        Enumerates all simple paths (xi is monotone but not isotone, so
        Dijkstra is not applicable).

        Parameters
        ----------
        topology : network graph

        Returns
        -------
        Dict mapping (src, dst) -> optimal ordered node list.
        """
        # w[i,j] = 0 for all v_i, v_j in V

        w = { (vi, vj): 0.0
            for vi in topology.nodes
            for vj in topology.nodes
        }
        R_i_j = {}
        r_star_i_j = {}

        for n1, n2, _ in topology.edges:
            route = [n1, n2]
            R_i_j[(n1, n2)] = [route]        # R(i,j).append(e_ij)
            r_star_i_j[(n1, n2)] = route                # r*_ij = e_ij
            w[(n1, n2)] = self.xi(route, topology) # w_ij = Xi(e_ij, D)
            # And add incerse route
            route_inv = [n2, n1]
            R_i_j[(n2, n1)] = [route_inv]
            r_star_i_j[(n2, n1)] = route_inv
            w[(n2, n1)] = self.xi(route_inv, topology)

        for k in topology.nodes:
            for i in topology.nodes:
                for j in topology.nodes:
                    for p1 in R_i_j.get((i, k), []):
                        for p2 in R_i_j.get((k, j), []):
                            shared = set(p1) & set(p2) - {k}
                            if not shared:  # !(V(p1) & V(p2) & V\{k})
                                r = p1 + p2[1:]  # drop the leading k from p2 to avoid duplicate
                                if (i, j) not in R_i_j:
                                    R_i_j[(i, j)] = []
                                R_i_j[(i, j)].append(r)
                                w_r = self.xi(r, topology)
                                # If new weight is better, update the optimal route and weight
                                if w_r > w[(i, j)]:
                                    r_star_i_j[(i, j)] = r
                                    w[(i, j)] = w_r
        return {(vi, vj): (r_star_i_j[(vi, vj)], w[(vi, vj)]) for vi in topology.nodes for vj in topology.nodes if (vi, vj) in r_star_i_j}