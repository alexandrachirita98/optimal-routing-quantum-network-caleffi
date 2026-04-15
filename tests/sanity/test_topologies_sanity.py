import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from utils.constants.physical_constants import PARAMS
from utils.models.topology import Topology
from utils.models.optimal_routing import OptimalRouting

TOPOLOGIES_DIR = os.path.join(os.path.dirname(__file__), "..", "topologies", "core", "input")

topology_files = [
    f for f in os.listdir(TOPOLOGIES_DIR) if f.endswith(".json")
]


@pytest.fixture
def routing():
    return OptimalRouting(PARAMS)


@pytest.mark.parametrize("filename", topology_files)
def test_optimal_path_runs(routing, filename):
    topology = Topology.from_json(os.path.join(TOPOLOGIES_DIR, filename))
    result = routing.optimal_path(topology)
    assert isinstance(result, dict)
    assert len(result) > 0


@pytest.mark.parametrize("filename", topology_files)
def test_all_xi_scores_non_negative(routing, filename):
    topology = Topology.from_json(os.path.join(TOPOLOGIES_DIR, filename))
    result = routing.optimal_path(topology)
    for (src, dst), (route, score) in result.items():
        assert score >= 0, f"{filename}: negative xi for ({src}, {dst})"


