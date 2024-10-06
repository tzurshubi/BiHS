import networkx as nx
from sage.all import *
from sage.graphs.connectivity import TriconnectivitySPQR
from sage.graphs.graph import Graph


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
    head = state.head()
    if head is None:
        return 0
    return dfs_reachable(state.graph, head, visited)


def bcc_heuristic(state, goal):
    graph = state.graph.copy()  # Clone the graph to avoid modifying the original
    tail_nodes = state.tail()  # Nodes to be removed
    graph.remove_nodes_from(tail_nodes)  # Remove tail nodes

    head = state.head()

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


def traverse_spqr(tree, head_component, goal_component):
    """
    Traverse the SPQR tree to compute the h_MIS value based on the given SPQR components.
    This implementation follows the logic presented in the paper's TRAVERSE algorithm.
    
    Arguments:
    - tree: The SPQR tree.
    - head_component: The component containing the head vertex.
    - goal_component: The component containing the goal vertex.

    Returns:
    - The computed h_MIS value.
    """
    def traverse_component(component, edge_in):
        # Base case: If the component is None, return 0
        if component is None:
            return 0

        # Get the type of the SPQR component
        component_type = component.label()

        # List of virtual edges for this component
        virtual_edges = component.virtual_edges()

        # If the component is a P component
        if component_type == 'P':
            max_size = 0
            for edge in virtual_edges:
                if edge != edge_in:
                    subtree_size = traverse_component(edge.target(), edge)
                    max_size = max(max_size, subtree_size)
            return max_size

        # If the component is an S component
        elif component_type == 'S':
            # Sum the sizes of all the virtual edge subtrees
            subtree_sum = sum(traverse_component(edge.target(), edge) for edge in virtual_edges if edge != edge_in)
            # Add the size of the vertices in the cycle minus the endpoints of the entering edge
            num_vertices = len(component.vertices()) - 2  # Do not count s' and t'
            return num_vertices + subtree_sum

        # If the component is an R component
        elif component_type == 'R':
            # Start with the number of internal vertices in the component, excluding s' and t'
            total = len(component.vertices()) - 2
            non_exclusion_edges = set(virtual_edges)

            # Handle virtual edges incident on s' or t'
            for endpoint in [edge_in.source(), edge_in.target()]:
                exclusion_edges = {e for e in virtual_edges if e.source() == endpoint or e.target() == endpoint}
                if len(exclusion_edges) >= 2:
                    total += max(traverse_component(edge.target(), edge) for edge in exclusion_edges)
                    non_exclusion_edges -= exclusion_edges

            # Handle 2-edge cuts
            cuts = find_two_edge_cuts(component, edge_in)
            for cut in cuts:
                total += max(traverse_component(edge.target(), edge) for edge in cut)
                non_exclusion_edges -= cut

            # Add all the remaining mutually independent subtrees
            total += sum(traverse_component(edge.target(), edge) for edge in non_exclusion_edges)

            return total

        # If the component type is unknown, return 0
        return 0

    # Starting traversal from the head component to the goal component
    return traverse_component(tree.root(), None)


def h_mis_heuristic(state, goal):
    
    # Use the graph from the current state
    graph = state.graph.copy()  # Clone the graph to avoid modifying the original
    tail_nodes = state.tail()  # Nodes to be removed
    graph.remove_nodes_from(tail_nodes)  # Remove tail nodes
    
    head = state.head()
    
    # Convert the NetworkX graph to a Sage Graph
    sage_graph = Graph(graph)


    # Create SPQR tree using SageMath's TriconnectivitySPQR
    tric = TriconnectivitySPQR(sage_graph)
    T = tric.get_spqr_tree()
    tric.print_triconnected_components()
    
    # Find components containing the head and goal vertices
    head_component = sage_graph.connected_component_containing_vertex(state.head(), sort=False) #find_component_containing_vertex(tric, state.head())
    goal_component = sage_graph.connected_component_containing_vertex(goal,sort=False) # find_component_containing_vertex(tric, goal)
    print(f"head_component: {head_component}")
    print(f"goal_component: {goal_component}")


    # Return 0 if head or goal is not in any component
    if head_component is None or goal_component is None:
        return 0
    
    # Traverse the SPQR tree to compute the h_MIS heuristic
    return traverse_spqr(T, head_component, goal_component) + 1


def heuristic(state, goal, heuristic_name):
    print(f"Running heuristic with parameters: state: {state}, goal: {goal}, heuristic_name: {heuristic_name}")

    if heuristic_name == "heuristic0":
        return heuristic0(state)
    elif heuristic_name == "reachable_heuristic":
        return reachable_heuristic(state)
    elif heuristic_name == "bcc_heuristic":
        return bcc_heuristic(state, goal)
    elif heuristic_name == "mis_heuristic":
        return h_mis_heuristic(state, goal)
    else:
        print("Invalid heuristic name")
        return 1 / 0
