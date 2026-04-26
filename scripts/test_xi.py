import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from utils.constants.physical_constants import PARAMS
from utils.models.topology import Topology
from utils.models.physical_params import PhysicalParams
from utils.models.optimal_routing import OptimalRouting

def test_xi(routing):
    topology = Topology(
        nodes=["r1", "r2", "r3", "r4"],
        edges=[
            ("r1", "r2", 20_000.0),
            ("r2", "r3", 20_000.0),
            ("r3", "r4", 20_000.0),
            ("r1", "r3", 50_000.0),
        ],
    )

    xi_single = routing.xi(["r1", "r2"], topology)
    xi_two    = routing.xi(["r1", "r2", "r3"], topology)
    xi_three  = routing.xi(["r1", "r2", "r3", "r4"], topology)
    xi_direct = routing.xi(["r1", "r3"], topology)

    print(f"xi(r1->r2)        = {xi_single:.3e} pairs/s")
    print(f"xi(r1->r2->r3)    = {xi_two:.3e} pairs/s")
    print(f"xi(r1->r2->r3->r4)= {xi_three:.3e} pairs/s")
    print(f"xi(r1->r3 direct) = {xi_direct:.3e} pairs/s")

    assert xi_single >= 0.0
    assert xi_two    >= 0.0
    assert xi_three  >= 0.0
    assert xi_direct >= 0.0

test_xi(OptimalRouting(PARAMS))