from __future__ import annotations

import heapq
import math

from utils.models.physical_params import PhysicalParams
from utils.models.topology import Topology
from utils.models.optimal_routing import OptimalRouting

def dijkstra(topology: Topology, start_node: str, target_node: str = None):
    """
    Implements Dijkstra's algorithm using the Topology class.
    
    :param topology: An instance of the Topology class.
    :param start_node: The string ID of the starting node.
    :param target_node: (Optional) The string ID of the destination node.
    :return: (distances, previous_nodes)
    """
    # 1. Initialize distances and predecessors
    # distances: node -> shortest distance from start_node
    distances = {node: float('inf') for node in topology.nodes}
    distances[start_node] = 0
    
    # previous_nodes: node -> parent node in the shortest path tree
    previous_nodes = {node: None for node in topology.nodes}
    
    # 2. Initialize Priority Queue (min-heap)
    # Stores tuples of (distance, node_id)
    pq = [(0, start_node)]
    
    # Pre-fetch adjacency and distance lookups for performance
    adj = topology.adj
    edge_weights = topology.dist

    while pq:
        current_distance, u = heapq.heappop(pq)

        # Optimization: If we already found a better path to u, skip it
        if current_distance > distances[u]:
            continue

        # If we reached the target, we can stop early
        if target_node and u == target_node:
            break

        # 3. Relaxation Step
        for v in adj[u]:
            weight = edge_weights[(u, v)]
            distance = current_distance + weight

            # If this new path to v is shorter than any previously known path
            if distance < distances[v]:
                distances[v] = distance
                previous_nodes[v] = u
                heapq.heappush(pq, (distance, v))

    return distances, previous_nodes

def get_path(previous_nodes, target_node):
    """Utility to turn the previous_nodes dict into a list (the path)."""
    path = []
    curr = target_node
    while curr is not None:
        path.append(curr)
        curr = previous_nodes[curr]
    return path[::-1] # Reverse it


class DijkstraRouting:
    """
    Dijkstra-based routing for quantum networks.
    
    Uses Dijkstra's algorithm to find the shortest physical path (by distance),
    then computes the entanglement rate for that path using the quantum
    channel model.
    """
    
    def __init__(self, params: PhysicalParams) -> None:
        self.params = params
        self.optimal_routing = OptimalRouting(params)
    
    def shortest_path(self, topology: Topology, start_node: str, target_node: str) -> list[str]:
        """
        Find the shortest path from start_node to target_node using Dijkstra's algorithm.
        
        Parameters
        ----------
        topology : Topology
            The network topology.
        start_node : str
            The starting node.
        target_node : str
            The destination node.
            
        Returns
        -------
        list[str]
            The path from start_node to target_node.
        """
        distances, previous_nodes = dijkstra(topology, start_node, target_node)
        path = get_path(previous_nodes, target_node)
        
        # Verify path is valid
        if path[0] != start_node or path[-1] != target_node:
            return []
        
        return path
    
    def entanglement_weighted_path(self, topology: Topology, start_node: str, target_node: str) -> list[str]:
        """
        Paper-style Dijkstra: edge weight = 1/xi_link.

        Demonstrates the non-isotonicity argument from Caleffi (2017): locally
        cheapest edges (highest per-link xi) need not yield the best end-to-end
        entanglement rate. Weights are always positive, so Dijkstra terminates
        even when the topology contains cycles.
        """
        weighted_edges = []
        for n1, n2, _d in topology.edges:
            xi_link = self.optimal_routing.xi([n1, n2], topology)
            w = 1.0 / xi_link if xi_link > 0 else math.inf
            weighted_edges.append((n1, n2, w))
        weighted = Topology(nodes=list(topology.nodes), edges=weighted_edges)
        _, previous_nodes = dijkstra(weighted, start_node, target_node)
        path = get_path(previous_nodes, target_node)

        if not path or path[0] != start_node or path[-1] != target_node:
            return []

        return path

    def xi_shortest_path(self, topology: Topology, start_node: str, target_node: str) -> float:
        """
        Compute the entanglement rate for the shortest physical path.
        
        Parameters
        ----------
        topology : Topology
            The network topology.
        start_node : str
            The starting node.
        target_node : str
            The destination node.
            
        Returns
        -------
        float
            The end-to-end entanglement rate for the shortest path.
        """
        path = self.shortest_path(topology, start_node, target_node)
        
        if not path or len(path) < 2:
            return 0.0
        
        # Use the optimal_routing's xi method on the found path
        return self.optimal_routing.xi(path, topology)