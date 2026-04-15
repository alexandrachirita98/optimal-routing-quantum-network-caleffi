import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from utils.constants.physical_constants import PARAMS
from utils.models.topology import Topology
from utils.models.optimal_routing import OptimalRouting

TOPOLOGIES_DIR = os.path.join(os.path.dirname(__file__), "..", "topologies", "sanity_input")
EXPECTED_DIR   = os.path.join(os.path.dirname(__file__), "..", "topologies", "expected_results")

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


expected_files = [
    f for f in os.listdir(EXPECTED_DIR) if f.endswith(".json")
]


@pytest.mark.parametrize("filename", expected_files)
def test_routes_match_expected(routing, filename):
    topology = Topology.from_json(os.path.join(TOPOLOGIES_DIR, filename))
    expected = json.load(open(os.path.join(EXPECTED_DIR, filename)))
    result = routing.optimal_path(topology)

    for key, entry in expected.items():
        src, dst = key.split(",")
        assert (src, dst) in result, f"({src},{dst}) missing from optimal_path output"
        route, _ = result[(src, dst)]
        assert route == entry["route"], (
            f"({src},{dst}): expected {entry['route']}, got {route}"
        )
