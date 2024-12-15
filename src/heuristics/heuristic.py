import networkx as nx
# from sage.all import *
# from sage.graphs.connectivity import TriconnectivitySPQR
# from sage.graphs.graph import Graph
from .h_mis import *

# Return the total number of vertices in the graph
# This is the most basic heuristic which is also admissible in this case
def heuristic0(state):
    return len(state.graph.nodes)


def dfs_reachable(graph, start, visited):
    stack = [start]
    reachable_count = 0

    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.append(vertex)
            if vertex != start:
                reachable_count += 1
            for neighbor in graph.neighbors(vertex):
                if neighbor not in visited and neighbor not in stack:
                    stack.append(neighbor)

    return reachable_count


def reachable_heuristic(state):
    visited = state.tail()
    head = state.head
    if head is None:
        return 0
    return dfs_reachable(state.graph, head, visited)


def bcc_heuristic(state, goal):
    graph = state.graph.copy()  # Clone the graph to avoid modifying the original
    tail_nodes = state.tail()  # Nodes to be removed
    graph.remove_nodes_from(tail_nodes)  # Remove tail nodes

    head = state.head

    # Find all bi-connected components and articulation points
    bcc = list(nx.biconnected_components(graph))
    articulation_points = list(nx.articulation_points(graph))

    # Create a mapping from nodes to BCCs
    node_to_bcc = {}
    for i, component_i in enumerate(bcc):
        for node in component_i:
            if node not in node_to_bcc:
                node_to_bcc[node] = []
            node_to_bcc[node].append(i)

    # Create the BCT
    bct = nx.Graph()
    for i, component_i in enumerate(bcc):
        bct.add_node(i)
    for i, component_i in enumerate(bcc):
        for node in component_i:
            for j, component_j in enumerate(bcc):
                if j <= i:
                    continue
                if node in component_j:
                    bct.add_edge(i, j)

            # if node in articulation_points:
            #     bct.add_edge(i, node)

    # Find the path in the BCT from the BCC containing the head to the BCC containing the goal
    head_bcc = node_to_bcc.get(head, [None])[0]
    goal_bcc = node_to_bcc.get(goal, [None])[0]

    if head_bcc is None or goal_bcc is None:
        return 0

    # In the BCT, there can be only one path from the head_bcc to the goal_bcc
    # If there were more, the definition of BBC is defied
    try:
        bcc_path = nx.shortest_path(bct, head_bcc, goal_bcc)
    # except nx.NodeNotFound:
    #     print("node not found in bcc: ", bcc)
    #     return 0
    except nx.NetworkXNoPath:
        return 0

    # Count the total number of reachable vertices in each BCC along this path
    reachable_vertices = set()
    for element in bcc_path:
        if isinstance(element, int) and 0 <= element < len(bcc):
            reachable_vertices.update(bcc[element])

    return len(reachable_vertices) - 1


def find_component_containing_vertex(tric, vertex):
    # Iterate over triconnected components to find which contains the vertex
    for component in tric.get_triconnected_components():
        for edge in component:
            # Each component is represented as a tuple of (node1, node2, ...). Check if the vertex is part of any edge.
            if vertex in edge[:2]:  # Check the first two elements, which represent vertices in the edge
                return component
    return None


def mis_heuristic(state, goal, snake):
    """
    Compute the h_MIS heuristic, combining the BCC-based heuristic and SPQR tree-based exclusion pairs.

    Parameters:
    - state: The current search state, including the graph.
    - goal: The target goal node in the graph.

    Returns:
    - int: The heuristic estimate for the longest simple path.
    """
    # Step 1: Compute h_BCC to identify and remove non-must-include vertices.
    graph = state.graph.copy()  # Clone the graph to avoid modifying the original
    tail_nodes = state.tail()  # Nodes to be removed
    graph.remove_nodes_from(tail_nodes)  # Remove tail nodes
    # return get_max_nodes_spqr_recursive(graph, state.head, goal, return_nodes=False)-1

    
    head = state.head

    # Find all bi-connected components and articulation points
    bcc = list(nx.biconnected_components(graph))
    articulation_points = list(nx.articulation_points(graph))

    # Create a mapping from nodes to BCCs
    node_to_bcc = {}
    for i, component_i in enumerate(bcc):
        for node in component_i:
            if node not in node_to_bcc:
                node_to_bcc[node] = []
            node_to_bcc[node].append(i)

    # Create the BCT (Block-Cut Tree)
    bct = nx.Graph()
    for i, component_i in enumerate(bcc):
        bct.add_node(i)
    for i, component_i in enumerate(bcc):
        for node in component_i:
            for j, component_j in enumerate(bcc):
                if j <= i:
                    continue
                if node in component_j:
                    bct.add_edge(i, j)

    # Find the path in the BCT from the BCC containing the head to the BCC containing the goal
    head_bcc = node_to_bcc.get(head, [None])[0]
    goal_bcc = node_to_bcc.get(goal, [None])[0]

    if head_bcc is None or goal_bcc is None:
        return 0

    try:
        bcc_path = nx.shortest_path(bct, head_bcc, goal_bcc)
    except nx.NetworkXNoPath:
        return 0

    # Count the total number of reachable vertices in each BCC along this path
    reachable_vertices = set()
    for element in bcc_path:
        if isinstance(element, int) and 0 <= element < len(bcc):
            reachable_vertices.update(bcc[element])

    # Create the reduced graph G' by removing vertices not in the reachable set.
    reduced_graph = graph.subgraph(reachable_vertices).copy()

    # Step 2: Compute the SPQR-based MIS heuristic on the reduced graph.
    in_node = head
    out_node = goal

    # Use the get_max_nodes_spqr_recursive function for computing relevant nodes.
    if not snake:
        spqr_nodes_count = get_max_nodes_spqr_recursive(reduced_graph, in_node, out_node, return_nodes=False)
    else:
        spqr_nodes_count = get_max_nodes_spqr_snake(reduced_graph, in_node, out_node, return_pairs=False)

    # Combine results from both steps to form the h_MIS value.
    return spqr_nodes_count - 1


import networkx as nx

def bct_is_heuristic(state, goal, snake=True):
    """
    Computes the combined BCT + IS heuristic for the Snake/Coil in the Box problem.

    Args:
    - state: A dictionary representing the current state, including:
        - 'path': List of vertices in the current path.
        - 'g': The cost (length) of the current path.
        - 'head': The current head of the path.
        - 'path_vertices_bitmap': Bitmap representation of visited vertices.
        - 'path_vertices_and_neighbors_bitmap': Bitmap of visited vertices and their neighbors.
        - 'graph': The NetworkX graph representation of the problem.
    - goal: The goal vertex (ignored for heuristic computation in MAX problems).
    - snake: Boolean, True for Snake in the Box, False for Coil in the Box.

    Returns:
    - A heuristic estimate of the maximum length of the path from the current state.
    """

    graph = state.graph
    head = state.head
    path = state.path

    # Compute the block-cut-point tree of the reachable subgraph
    reachable_graph = nx.Graph(graph.subgraph(nx.node_connected_component(graph, head)))
    biconnected_components = list(nx.biconnected_components(reachable_graph))
    bct = nx.Graph()
    component_mapping = {}

    # Build the BCT
    for i, component in enumerate(biconnected_components):
        bct.add_node(i, vertices=set(component))
        for vertex in component:
            component_mapping[vertex] = i

    # Add edges between biconnected components sharing a cut point
    for vertex in reachable_graph.nodes:
        connected_components = {component_mapping[neighbor] for neighbor in graph.neighbors(vertex) if neighbor in component_mapping}
        for c1, c2 in zip(connected_components, list(connected_components)[1:]):
            bct.add_edge(c1, c2)

    # Heuristic values for each biconnected component
    def compute_h_component(component):
        # Calculate IS-based heuristic
        vertices = bct.nodes[component]['vertices']
        independent_set = nx.maximal_independent_set(reachable_graph.subgraph(vertices))
        return len(independent_set) - 1

    def compute_h_bct(component):
        # Recursive computation of heuristic across BCT
        h_component = compute_h_component(component)
        child_components = [neighbor for neighbor in bct.neighbors(component) if neighbor != component_mapping[state['head']]]
        return max(
            h_component,
            max((compute_h_bct(child) + h_component for child in child_components), default=0)
        )

    # Apply the heuristic calculation
    root_component = component_mapping[head]
    h_bct_is = compute_h_bct(root_component)
    
    # Adjust heuristic for Coil in the Box
    if not snake:
        tail = path[0]
        h_bct_is += 1 if tail in bct.nodes[root_component]['vertices'] else 0

    return h_bct_is




def heuristic(state, goal, heuristic_name, snake):
    # print(f"Running heuristic with parameters: state: {state}, goal: {goal}, heuristic_name: {heuristic_name}")
    if not isinstance(goal,int):
        goal = max(goal)

    if heuristic_name == "heuristic0":
        return heuristic0(state)
    elif heuristic_name == "reachable_heuristic":
        return reachable_heuristic(state)
    elif heuristic_name == "bcc_heuristic":
        return bcc_heuristic(state, goal)
    elif heuristic_name == "mis_heuristic":
        return mis_heuristic(state, goal, snake)
    elif heuristic_name == "bct_is_heuristic":
        return bct_is_heuristic(state, goal, snake)
    else:
        print(f"Invalid heuristic name: {heuristic_name}")
        return 1 / 0
