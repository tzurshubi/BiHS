import networkx as nx
import pulp
import matplotlib.pyplot as plt
import math
from collections import defaultdict


def solve_longest_path_lp_relaxed(G: nx.Graph, source, target):
    """
    Solves the LP relaxation of the longest simple path problem
    in an unweighted, undirected graph using PuLP.

    Args:
        G (nx.Graph): NetworkX undirected graph
        source: Source node label
        target: Target node label

    Returns:
        Tuple:
            - Objective value (float)
            - Dictionary of edge variable values {(u, v): value}
    """
    # Create LP problem
    prob = pulp.LpProblem("Longest_Simple_Path_LP", pulp.LpMaximize)

    # Define edge variables e_{ij} ∈ [0, 1]
    edge_vars = {}
    for u, v in G.edges():
        name = f"e_{str(u)}_{str(v)}"
        edge_vars[(u, v)] = pulp.LpVariable(name, lowBound=0, upBound=1, cat='Continuous')

    # Define vertex variables v_i ∈ [0, 1]
    vertex_vars = {}
    for i in G.nodes():
        vertex_vars[i] = pulp.LpVariable(f"v_{str(i)}", lowBound=0, upBound=1, cat='Continuous')

    # Objective: Maximize the sum of edge variables (total path length)
    prob += pulp.lpSum(edge_vars.values())

    # Degree constraints:
    for node in G.nodes():
        incident_edges = []
        for u, v in G.edges(node):
            key = (u, v) if (u, v) in edge_vars else (v, u)
            incident_edges.append(edge_vars[key])

        if node == source or node == target:
            prob += pulp.lpSum(incident_edges) == 1
        else:
            prob += pulp.lpSum(incident_edges) == 2 * vertex_vars[node]

    # Edge-vertex consistency: e_{ij} ≤ v_i, e_{ij} ≤ v_j
    for (u, v), e in edge_vars.items():
        prob += e <= vertex_vars[u]
        prob += e <= vertex_vars[v]

    # Solve the LP
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Prepare the output
    edge_values = {
        (u, v): pulp.value(var)
        for (u, v), var in edge_vars.items()
        if pulp.value(var) > 1e-5  # Filter near-zero values
    }

    return pulp.value(prob.objective), edge_values


def plot_graph_with_lp_solution(G: nx.Graph, edge_values: dict, sol_value=None):
    """
    Draws the graph with edge weights (values) from LP solution, arranged in a grid layout.
    Handles missing node indices by leaving blank spaces.

    Args:
        G (nx.Graph): The original graph with integer node labels
        edge_values (dict): A dictionary {(u, v): value} from the LP solution
        sol_value (float, optional): The LP objective value
    """
    title = "LP Solution"
    if sol_value is not None:
        title += f". Solution Value: {sol_value:.2f}"

    # Get all node indices (assumes integer labels)
    try:
        node_indices = sorted(int(n) for n in G.nodes())
    except ValueError:
        raise ValueError("Node labels must be integers or convertible to integers for grid layout.")

    min_idx = min(node_indices)
    max_idx = max(node_indices)
    num_slots = max_idx - min_idx + 1

    # Estimate grid dimensions
    grid_cols = math.ceil(math.sqrt(num_slots))
    grid_rows = math.ceil(num_slots / grid_cols)

    # Full grid position map
    grid_positions = {}
    all_indices = list(range(min_idx, min_idx + grid_rows * grid_cols))

    for i, idx in enumerate(all_indices):
        row = i // grid_cols
        col = i % grid_cols
        grid_positions[idx] = (col, -row)

    # Actual node positions
    pos = {node: grid_positions[node] for node in G.nodes()}

    # Prepare edge visuals
    edge_colors = []
    edge_widths = []
    edge_labels = {}

    for u, v in G.edges():
        key = (u, v) if (u, v) in edge_values else (v, u)
        value = edge_values.get(key, 0.0)
        edge_labels[(u, v)] = f"{value:.2f}" if value > 1e-5 else ""
        edge_colors.append("green" if value > 1e-5 else "gray")
        edge_widths.append(2 + 4 * value)

    # Draw nodes and base edges
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=700)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths)
    nx.draw_networkx_labels(G, pos, font_weight='bold')

    # Draw edge labels on top
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="black", label_pos=0.5)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.savefig("LP Solution")


def solve_coil_in_the_box_lp_relaxed(G: nx.Graph, init_vertices=None, init_edges=None):
    """
    Solves the LP relaxation of the Coil-in-the-Box problem (finding the longest induced cycle)
    in a d-dimensional hypercube.

    Args:
        G (nx.Graph): A d-dimensional hypercube (undirected, unweighted)
        init_vertices (list of int): List of vertices to fix to value 1 (optional)
        init_edges (list of tuple): List of edges to fix to value 1 (optional)

    Returns:
        Tuple:
            - Objective value (float)
            - Dictionary of edge variable values {(u, v): value}
            - Dictionary of vertex variable values {v: value}
    """
    prob = pulp.LpProblem("Coil_In_The_Box_LP", pulp.LpMaximize)

    # Define edge and vertex variables
    edge_vars = {}
    for u, v in G.edges():
        edge_vars[(u, v)] = pulp.LpVariable(f"e_{u}_{v}", lowBound=0, upBound=1, cat='Continuous')

    vertex_vars = {}
    for i in G.nodes():
        vertex_vars[i] = pulp.LpVariable(f"v_{i}", lowBound=0, upBound=1, cat='Continuous')

    # Objective
    prob += pulp.lpSum(edge_vars.values())

    # Degree constraint: sum of incident edges = 2 * v_i
    for node in G.nodes():
        incident_edges = []
        for neighbor in G.neighbors(node):
            key = (node, neighbor) if (node, neighbor) in edge_vars else (neighbor, node)
            incident_edges.append(edge_vars[key])
        prob += pulp.lpSum(incident_edges) == 2 * vertex_vars[node]

    # Edge-vertex consistency
    for (u, v), e in edge_vars.items():
        prob += e <= vertex_vars[u]
        prob += e <= vertex_vars[v]
        prob += e >= vertex_vars[u] + vertex_vars[v] - 1

    # Dimension
    d = next(iter(dict(G.degree()).values()))

    # Induced subgraph condition
    for v in G.nodes():
        neighbor_sum = pulp.lpSum(vertex_vars[u] for u in G.neighbors(v))
        prob += neighbor_sum <= (2 - d) * vertex_vars[v] + d

    # Set initial fixed values for vertices and edges if provided
    if init_vertices:
        for v in init_vertices:
            if v in vertex_vars:
                vertex_vars[v].setInitialValue(1.0)
                vertex_vars[v].fixValue()

    if init_edges:
        for u, v in init_edges:
            key = (u, v) if (u, v) in edge_vars else (v, u)
            if key in edge_vars:
                edge_vars[key].setInitialValue(1.0)
                edge_vars[key].fixValue()

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Output
    edge_values = {
        (u, v): pulp.value(var)
        for (u, v), var in edge_vars.items()
        if pulp.value(var) > 1e-5
    }
    vertex_values = {
        v: pulp.value(var)
        for v, var in vertex_vars.items()
        if pulp.value(var) > 1e-5
    }

    return pulp.value(prob.objective), edge_values, vertex_values


def plot_hypercube_with_lp_solution(G: nx.Graph, edge_values: dict, sol_value=None):
    """
    Draws the hypercube graph with edge weights from LP solution.
    Nodes must be integers from 0 to 2^d - 1. Nodes are positioned according to their bitstring.

    Args:
        G (nx.Graph): A hypercube graph with integer-labeled nodes
        edge_values (dict): A dictionary {(u, v): value} from the LP solution
        sol_value (float, optional): The LP objective value
    """
    # Get dimension d from log2 of number of nodes
    import math
    d = int(math.log2(len(G.nodes)))
    if 2 ** d != len(G.nodes):
        raise ValueError("Graph does not appear to be a full hypercube.")

    title = "LP Solution for Hypercube"
    if sol_value is not None:
        title += f" (Objective: {sol_value:.2f})"

    # Normalize node layout to keep cube visually balanced
    # n = len(G.nodes())
    # radius = 10  # adjust for spacing
    # pos = {}

    # for i, node in enumerate(sorted(G.nodes())):
    #     angle = 2 * math.pi * i / n
    #     x = radius * math.cos(angle)
    #     y = radius * math.sin(angle)
    #     pos[node] = (x, y)

    pos = nx.spring_layout(G)
    # Prepare edge visuals
    edge_colors = []
    edge_widths = []
    edge_labels = {}

    for u, v in G.edges():
        key = (u, v) if (u, v) in edge_values else (v, u)
        value = edge_values.get(key, 0.0)
        edge_labels[(u, v)] = f"{value:.2f}" if value > 1e-5 else ""
        edge_colors.append("green" if value > 1e-5 else "gray")
        edge_widths.append(2 + 4 * value)

    # Draw nodes and base edges # G, pos
    nx.draw_networkx_nodes(G,  pos, node_color='lightblue', node_size=700)
    nx.draw_networkx_edges(G,  pos, edge_color=edge_colors, width=edge_widths)
    nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in G.nodes()}, font_weight='bold', font_size=10)

    # Draw edge labels on top
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="black", label_pos=0.5, font_size=8)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.savefig("LP_Solution_Hypercube.png")


def solve_coil_round(G):
    """
    Iteratively solves the LP relaxation of the coil-in-the-box problem,
    each time adding the highest-scoring (unfixed) vertex to the fixed set.

    Returns:
        - Final solution value
        - Final edge values
        - Final vertex values
    """
    init_vertices = [0, 1, 3, 7]
    init_edges = [(0, 1), (1, 3), (3, 7)]
    fixed_vertices = set(init_vertices)

    while True:
        # Solve with current fixed set
        sol_value, edge_values, vertex_values = solve_coil_in_the_box_lp_relaxed(
            G,
            init_vertices=list(fixed_vertices),
            init_edges=init_edges
        )

        print(f"Current solution value: {sol_value:.2f}, fixed vertices: {sorted(fixed_vertices)}")

        # Find vertex with highest value not yet fixed
        remaining = {v: val for v, val in vertex_values.items() if v not in fixed_vertices}
        if not remaining:
            break  # all vertices are fixed

        # Get the vertex with the highest LP value
        next_vertex = max(remaining, key=remaining.get)
        max_val = remaining[next_vertex]

        # If it's "not meaningful", stop
        if max_val < 1e-3:
            break

        # Add it to fixed set
        fixed_vertices.add(next_vertex)

    # Final solve with all fixed vertices
    sol_value, edge_values, vertex_values = solve_coil_in_the_box_lp_relaxed(
        G,
        init_vertices=list(fixed_vertices),
        init_edges=init_edges
    )

    return sol_value, edge_values, vertex_values

# def solve_longest_path_lp_relaxed(G: nx.Graph, source, target):
#     """
#     Solves the LP relaxation of the longest simple path problem
#     in an unweighted, undirected graph using PuLP.

#     Args:
#         G (nx.Graph): NetworkX undirected graph
#         source: Source node
#         target: Target node

#     Returns:
#         Tuple:
#             - Objective value (float)
#             - Dictionary of edge variable values {(u, v): value}
#     """
#     # Ensure edges are undirected and consistent
#     G = G.copy()
#     G = nx.convert_node_labels_to_integers(G, label_attribute="original_label")  # For integer node IDs
#     original_labels = nx.get_node_attributes(G, "original_label")
#     reverse_labels = {v: k for k, v in original_labels.items()}

#     n = G.number_of_nodes()

#     # Create LP problem
#     prob = pulp.LpProblem("Longest_Simple_Path_LP", pulp.LpMaximize)

#     # Define edge variables e_{ij} ∈ [0, 1]
#     edge_vars = {}
#     for u, v in G.edges():
#         name = f"e_{min(u, v)}_{max(u, v)}"
#         edge_vars[(u, v)] = pulp.LpVariable(name, lowBound=0, upBound=1, cat='Continuous')

#     # Define vertex variables v_i ∈ [0, 1]
#     vertex_vars = {}
#     for i in G.nodes():
#         vertex_vars[i] = pulp.LpVariable(f"v_{i}", lowBound=0, upBound=1, cat='Continuous')

#     # Objective: Maximize the sum of edge variables (total path length)
#     prob += pulp.lpSum(edge_vars.values())

#     # Degree constraints:
#     for node in G.nodes():
#         incident_edges = [edge_vars[(u, v)] if (u, v) in edge_vars else edge_vars[(v, u)]
#                           for u, v in G.edges(node)]

#         if node == source or node == target:
#             prob += pulp.lpSum(incident_edges) == 1
#         else:
#             prob += pulp.lpSum(incident_edges) == 2 * vertex_vars[node]

#     # Edge-vertex consistency: e_{ij} ≤ v_i, e_{ij} ≤ v_j
#     for (u, v), e in edge_vars.items():
#         prob += e <= vertex_vars[u]
#         prob += e <= vertex_vars[v]

#     # Solve the LP
#     prob.solve(pulp.PULP_CBC_CMD(msg=False))

#     # Prepare the output
#     edge_values = {}
#     for (u, v), var in edge_vars.items():
#         if pulp.value(var) > 1e-5:  # ignore near-zero values
#             label_u = reverse_labels[u]
#             label_v = reverse_labels[v]
#             edge_values[(label_u, label_v)] = pulp.value(var)

#     return pulp.value(prob.objective), edge_values
