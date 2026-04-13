"""
Topology — physical graph of the quantum network.

JSON format
-----------
{
    "nodes": ["r1", "r2", "r3"],
    "edges": [
        ["r1", "r2", 20000],
        ["r2", "r3", 20000]
    ]
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property


@dataclass
class Topology:
    nodes: list[str]
    edges: list[tuple[str, str, float]]   # (node1, node2, distance_m)

    @classmethod
    def from_json(cls, path: str) -> Topology:
        with open(path) as f:
            data = json.load(f)
        return cls(
            nodes=data["nodes"],
            edges=[(e[0], e[1], float(e[2])) for e in data["edges"]],
        )

    @cached_property
    def dist(self) -> dict[tuple[str, str], float]:
        """Symmetric distance lookup: (node_i, node_j) → distance [m]."""
        d = {}
        for n1, n2, distance in self.edges:
            d[(n1, n2)] = distance
            d[(n2, n1)] = distance
        return d

    @cached_property
    def adj(self) -> dict[str, list[str]]:
        """Adjacency list: node → list of neighbours."""
        a: dict[str, list[str]] = {n: [] for n in self.nodes}
        for n1, n2, _ in self.edges:
            a[n1].append(n2)
            a[n2].append(n1)
        return a
