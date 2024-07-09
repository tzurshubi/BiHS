import networkx as nx


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
    except nx.NetworkXNoPath:
        return 0

    # Count the total number of reachable vertices in each BCC along this path
    reachable_vertices = set()
    for element in bcc_path:
        if isinstance(element, int) and 0 <= element < len(bcc):
            reachable_vertices.update(bcc[element])

    return len(reachable_vertices) - 1


def heuristic(state, goal, heuristic_name):
    if heuristic_name == "heuristic0":
        return heuristic0(state)
    elif heuristic_name == "reachable_heuristic":
        return reachable_heuristic(state)
    elif heuristic_name == "bcc_heuristic":
        return bcc_heuristic(state, goal)
    else:
        print("Invalid heuristic name")
        return 1 / 0
