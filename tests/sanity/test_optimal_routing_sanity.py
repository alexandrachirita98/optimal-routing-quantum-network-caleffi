import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from utils.constants.physical_constants import PARAMS
from utils.models.topology import Topology
from utils.models.optimal_routing import OptimalRouting


@pytest.fixture
def setup():
    topology = Topology(
        nodes=["r1", "r2", "r3"],
        edges=[
            ("r1", "r2", 20_000.0),
            ("r2", "r3", 20_000.0),
        ],
    )
    routing = OptimalRouting(PARAMS)
    return routing, topology


def test_xi_single_link_positive(setup):
    routing, topology = setup
    assert routing.xi(["r1", "r2"], topology) > 0
    assert routing.xi(["r2", "r3"], topology) > 0


def test_optimal_path_finds_two_hop_route(setup):
    routing, topology = setup
    result = routing.optimal_path(topology)
    assert ("r1", "r3") in result
    route, _ = result[("r1", "r3")]
    assert route == ["r1", "r2", "r3"]
