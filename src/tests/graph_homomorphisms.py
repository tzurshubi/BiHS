from collections import defaultdict
import os
import json
from networkx.readwrite import json_graph
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher, DiGraphMatcher
from typing import Dict, Hashable, Iterable, Tuple, Optional, List


def load_graph_from_file(file_path):
    G = nx.Graph()
    with open(file_path, "r") as file:
        data = json.load(file)
        # Extract node labels
        nodes = [node["id"] for node in data["nodes"]]
        G.add_nodes_from(nodes)
        # Extract edges
        edges = [(edge["source"], edge["target"]) for edge in data["links"]]
        G.add_edges_from(edges)
    return G


def all_automorphisms(
    G: nx.Graph,
    must_map: Optional[Dict[Hashable, Hashable]] = None,
    exclude_identity: bool = False,
) -> List[Dict[Hashable, Hashable]]:
    """
    Enumerate all automorphisms f: V(G) -> V(G) (adjacency-preserving bijections),
    optionally constrained to satisfy a given partial mapping `must_map`.

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
    must_map : dict, optional
        A partial mapping {u: v, ...} that every automorphism must respect
        (i.e., f(u) == v for each pair). Nodes must exist in G, and values
        must be injective (no two keys map to the same v).
    exclude_identity : bool
        If True, remove the identity mapping from the results.

    Returns
    -------
    list[dict]
        Each dict is a vertex permutation (automorphism) of G.

    Notes
    -----
    - Works for Graph and DiGraph. MultiGraphs are not supported.
    - For efficiency, we pre-check simple necessary conditions on `must_map`
      (degree/profile and self-loop compatibility).
    """
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError("MultiGraphs not supported.")

    # Normalize and validate must_map
    if must_map:
        # Basic checks: nodes exist and target values are injective
        for u, v in must_map.items():
            if u not in G or v not in G:
                raise ValueError(f"must_map contains unknown node: {u}->{v}")
        if len(set(must_map.values())) != len(must_map):
            raise ValueError("must_map must be injective (distinct targets).")

        # Quick feasibility checks: degree/self-loop (and in/out for DiGraph)
        if isinstance(G, nx.DiGraph):
            for u, v in must_map.items():
                if (G.out_degree(u), G.in_degree(u)) != (G.out_degree(v), G.in_degree(v)):
                    # Impossible to extend to an automorphism
                    return []
                if G.has_edge(u, u) != G.has_edge(v, v):
                    return []
        else:
            for u, v in must_map.items():
                if G.degree(u) != G.degree(v):
                    return []
                if G.has_edge(u, u) != G.has_edge(v, v):
                    return []

    Matcher = DiGraphMatcher if isinstance(G, nx.DiGraph) else GraphMatcher
    GM = Matcher(G, G)

    autos = []
    for f in GM.isomorphisms_iter():
        # Enforce the forced pairs
        if must_map and any(f[u] != v for u, v in must_map.items()):
            continue
        if exclude_identity and all(f[u] == u for u in G.nodes):
            continue
        autos.append(f)
    return autos


if __name__ == "__main__":
    graph_file_name = "7d_cube.json"
    graph_file_dir = "/home/tzur-shubi/Documents/Programming/BiHS/data/graphs/cubes/"
    print_auto = False # False # True

    G = load_graph_from_file(graph_file_dir + graph_file_name)
    
    # if G.has_edge(0, 1): G.remove_edge(0, 1)
    # G.remove_nodes_from((set(G.neighbors(1)) | set(G.neighbors(3))) - {0, 7})
    
    print(f"Loaded graph [{graph_file_name}] with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")    
    autos = all_automorphisms(G, must_map={0:7, 7:0, 1:3, 3:1})
    print(f"Found {len(autos)} automorphisms.")
    
    if print_auto:
        for i, f in enumerate(autos):
            print(f"Automorphism #{i+1}:")
            for u in G.nodes():
                print(f"  {u} -> {f[u]}")
            print()
