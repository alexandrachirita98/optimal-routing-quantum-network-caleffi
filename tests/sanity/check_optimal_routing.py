"""
Sanity check for OptimalRouting on a simple 3-node linear topology:

    r1 ——20km—— r2 ——20km—— r3

Expected behaviour
------------------
- Direct links r1-r2 and r2-r3 should have a positive xi.
- The 2-hop route r1->r2->r3 should be found as optimal for (r1, r3).
- xi(r1->r2->r3) should be <= xi(r1->r2), since longer routes are slower.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.constants.physical_constants import PARAMS
from utils.models.topology import Topology
from utils.models.optimal_routing import OptimalRouting

# ── Build topology ────────────────────────────────────────────────────────────

topology = Topology(
    nodes=["r1", "r2", "r3"],
    edges=[
        ("r1", "r2", 20_000.0),
        ("r2", "r3", 20_000.0),
    ],
)

routing = OptimalRouting(PARAMS)

# ── Check single-link xi ──────────────────────────────────────────────────────

xi_r1_r2 = routing.xi(["r1", "r2"], topology)
xi_r2_r3 = routing.xi(["r2", "r3"], topology)
xi_r1_r3 = routing.xi(["r1", "r2", "r3"], topology)

print(f"xi(r1->r2)       = {xi_r1_r2:.4f} pairs/s")
print(f"xi(r2->r3)       = {xi_r2_r3:.4f} pairs/s")
print(f"xi(r1->r2->r3)   = {xi_r1_r3:.4f} pairs/s")

assert xi_r1_r2 > 0,  "xi(r1->r2) should be positive"
assert xi_r2_r3 > 0,  "xi(r2->r3) should be positive"
assert xi_r1_r3 > 0,  "xi(r1->r2->r3) should be positive"
assert xi_r1_r3 <= xi_r1_r2, "2-hop route should be slower than single link"

print()

# ── Check optimal_path ────────────────────────────────────────────────────────

result = routing.optimal_path(topology)

print("optimal_path results:")
for (src, dst), (route, score) in result.items():
    print(f"  ({src}, {dst}): route={route}, xi={score:.4f}")

# (r1, r3) should be routed through r2
assert ("r1", "r3") in result, "(r1, r3) should be in result"
route_r1_r3, _ = result[("r1", "r3")]
assert route_r1_r3 == ["r1", "r2", "r3"], f"Expected ['r1','r2','r3'], got {route_r1_r3}"

print()
print("All checks passed.")
