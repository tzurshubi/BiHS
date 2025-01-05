import networkx as nx
# from sage.all import *
# from sage.graphs.connectivity import TriconnectivitySPQR
# from sage.graphs.graph import Graph
from .h_mis import *
from models.state import State
from utils.utils import *


def OLD_spqr_heuristic(state, goal, snake):
    """
    Compute the h_MIS heuristic, combining the BCC-based heuristic and SPQR tree-based exclusion pairs.

    Parameters:
    - state: The current search state, including the graph.
    - goal: The target goal node in the graph.
    - snake: Boolean indicating whether the "snake" constraint is applied.

    Returns:
    - int: The heuristic estimate for the longest simple path.
    """
    # Step 1: Compute h_BCC to identify and remove non-must-include vertices.
    graph = state.graph.copy()  # Clone the graph to avoid modifying the original
    tail_nodes = state.tail()  # Nodes to be removed
    graph.remove_nodes_from(tail_nodes)  # Remove tail nodes

    head = state.head

    # Find all bi-connected components and articulation points
    bcc = list(nx.biconnected_components(graph))
    articulation_points = set(nx.articulation_points(graph))

    # Special case: the whole graph is one biconnected component
    if len(bcc) == 1:
        bcc_vertices = bcc[0]
        bcc_subgraph = graph.subgraph(bcc_vertices).copy()

        # Compute SPQR-based heuristic directly
        if not snake:
            return get_max_nodes_spqr_recursive(bcc_subgraph, head, goal, return_nodes=False) - 1
        else:
            return get_max_nodes_spqr_snake(bcc_subgraph, head, goal,y_filter=True, return_pairs=False) - 1

    # Create a mapping from nodes to BCCs
    node_to_bcc = {}
    for i, component_i in enumerate(bcc):
        for node in component_i:
            if node not in node_to_bcc:
                node_to_bcc[node] = []
            node_to_bcc[node].append(i)

    # Find the BCCs containing head and goal
    head_bccs = node_to_bcc.get(head, [])
    goal_bccs = node_to_bcc.get(goal, [])

    # Check if head and goal are in the same BCC
    common_bcc = None
    for bcc_index in head_bccs:
        if bcc_index in goal_bccs:
            common_bcc = bcc_index
            break

    if common_bcc is not None:
        # Head and goal are in the same BCC
        bcc_vertices = bcc[common_bcc]
        bcc_subgraph = graph.subgraph(bcc_vertices).copy()

        # Compute SPQR-based heuristic directly
        if not snake:
            return get_max_nodes_spqr_recursive(bcc_subgraph, head, goal, return_nodes=False) - 1
        else:
            return get_max_nodes_spqr_snake(bcc_subgraph, head, goal, return_pairs=False) - 1

    # Otherwise, proceed to find separate BCCs for head and goal
    head_bcc = head_bccs[0] if head_bccs else None
    goal_bcc = goal_bccs[0] if goal_bccs else None

    if head_bcc is None or goal_bcc is None:
        return 0

    # Create the BCT (Block-Cut Tree)
    bct = nx.Graph()
    for i, component_i in enumerate(bcc):
        bct.add_node(i, vertices=component_i)  # Store vertices in BCC
    for i, component_i in enumerate(bcc):
        for node in component_i:
            if node in articulation_points:
                for j, component_j in enumerate(bcc):
                    if j > i and node in component_j:
                        bct.add_edge(i, j, articulation_point=node)

    # Find the path in the BCT from the BCC containing the head to the BCC containing the goal
    try:
        bcc_path = nx.shortest_path(bct, head_bcc, goal_bcc)
    except nx.NetworkXNoPath:
        return 0

    # Handle the case where bcc_path contains only one node
    if len(bcc_path) == 1:
        single_bcc_index = bcc_path[0]
        bcc_vertices = bct.nodes[single_bcc_index]["vertices"]
        bcc_subgraph = graph.subgraph(bcc_vertices).copy()

        # Compute SPQR-based heuristic directly
        if not snake:
            return get_max_nodes_spqr_recursive(bcc_subgraph, head, goal, return_nodes=False) - 1
        else:
            return get_max_nodes_spqr_snake(bcc_subgraph, head, goal,y_filter=True, return_pairs=False) - 1

    # Step 2: Compute the SPQR-based MIS heuristic for each BCC in the path
    total_heuristic = 0

    for i in range(len(bcc_path)):
        bcc_index = bcc_path[i]
        bcc_vertices = bct.nodes[bcc_index]["vertices"]
        bcc_subgraph = graph.subgraph(bcc_vertices).copy()

        # Determine entry and exit points for the BCC
        if i == 0:
            # First BCC: entry is head, exit is the articulation point connecting to the next BCC
            in_node = head
            out_node = bct[bcc_index][bcc_path[i + 1]]["articulation_point"]
        elif i == len(bcc_path) - 1:
            # Last BCC: entry is the articulation point connecting to the previous BCC, exit is goal
            in_node = bct[bcc_index][bcc_path[i - 1]]["articulation_point"]
            out_node = goal
        else:
            # Intermediate BCC: entry and exit are the articulation points connecting to adjacent BCCs
            in_node = bct[bcc_index][bcc_path[i - 1]]["articulation_point"]
            out_node = bct[bcc_index][bcc_path[i + 1]]["articulation_point"]

        # Compute SPQR-based heuristic
        if not snake:
            spqr_nodes_count = get_max_nodes_spqr_recursive(
                bcc_subgraph, in_node, out_node, return_nodes=False
            )
        else:
            spqr_nodes_count = get_max_nodes_spqr_snake(
                bcc_subgraph, in_node, out_node,y_filter=True, return_pairs=False
            )

        # Add the heuristic value from the SPQR tree of this BCC
        total_heuristic += spqr_nodes_count - 1

    # Combine results from all BCCs to form the h_MIS value
    return total_heuristic


def bct_is_heuristic(state, goal, snake=True):
    """
    Compute the BCT-integrated h_IS heuristic for the coil-in-the-box problem.
    Args:
        state: The current state of the problem.
        goal: The goal vertex.
        snake: Whether the snake constraint is applied.
    Returns:
        h_is: The heuristic value.
    """
    graph = state.graph
    head = state.head
    path = state.path

    # Step 1: Compute Qn
    Qn = set(graph.nodes) - state.illegal

    # Step 2: Compute Rn
    reachable_nodes = nx.single_source_shortest_path_length(graph, head)
    Rn = set(reachable_nodes.keys()) & Qn

    # Step 3: Construct BCT from the graph induced by Rn
    induced_subgraph = graph.subgraph(Rn.add(head))
    Rn.remove(head)
    biconnected_components = list(nx.biconnected_components(induced_subgraph))
    cut_points = set(nx.articulation_points(induced_subgraph))

    # Create the block-cut-point tree
    bct = nx.Graph()
    for i, block in enumerate(biconnected_components):
        bct.add_node(f"B{i}", vertices=block)  # Add each block as a node
        for vertex in block:
            if vertex in cut_points:
                bct.add_edge(f"B{i}", vertex)  # Connect blocks to cut-points

    # Step 4: Find the branch in the BCT from the block containing head to the block containing goal
    block_of_head = next(b for b in biconnected_components if head in b)
    block_of_goal = next(b for b in biconnected_components if goal in b)

    head_block_node = [node for node, data in bct.nodes(data=True) if data.get("vertices") == block_of_head][0]
    goal_block_node = [node for node, data in bct.nodes(data=True) if data.get("vertices") == block_of_goal][0]

    path_in_bct = nx.shortest_path(bct, source=head_block_node, target=goal_block_node)

    # Step 5: Compute heuristic based on whether head and goal are in the same block
    if head_block_node == goal_block_node:
        # Case 1: head and goal are in the same block
        block_vertices = bct.nodes[head_block_node]["vertices"]
        neighbors_of_head = set(graph.neighbors(head))
        non_neighbors = block_vertices - neighbors_of_head
        h_is_total = min(len(block_vertices), len(non_neighbors) + 1)
    else:
        # Case 2: head and goal are in different blocks
        h_is_total = 0
        for i in range(len(path_in_bct) - 1):
            current_node = path_in_bct[i]
            next_node = path_in_bct[i + 1]

            if current_node.startswith("B"):  # Only process block nodes
                block_vertices = bct.nodes[current_node]["vertices"]

                # Determine entry and exit vertices
                if i == 0:  # First block
                    entry_vertex = head
                else:
                    entry_vertex = next(v for v in block_vertices if v in bct.nodes[path_in_bct[i - 1]]["vertices"])

                exit_vertex = next(v for v in block_vertices if v in bct.nodes[next_node]["vertices"])

                # Compute the number of vertices not neighbors of entry and exit
                neighbors_of_entry = set(graph.neighbors(entry_vertex))
                neighbors_of_exit = set(graph.neighbors(exit_vertex))
                non_neighbors = block_vertices - (neighbors_of_entry | neighbors_of_exit)

                # Add to the heuristic
                h_is_total += min(len(block_vertices), len(non_neighbors) + 2)

    return h_is_total



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


def Y_heuristic(graph):
    counter = 0

    while graph.number_of_nodes() > 0:
        # Find the vertex with the largest degree
        largest_degree_node = max(graph.nodes, key=lambda node: graph.degree(node), default=None)

        if largest_degree_node is None:
            break  # Graph is empty, stop

        largest_degree = graph.degree(largest_degree_node)

        # If the degree of the vertex is less than 3, stop the process
        if largest_degree < 3:
            return counter + len(graph)

        # Get the neighbors of the vertex
        neighbors = list(graph.neighbors(largest_degree_node))

        # Sort the neighbors by degree and select the three with the lowest degree
        neighbors_sorted_by_degree = sorted(neighbors, key=lambda node: graph.degree(node))
        selected_neighbors = neighbors_sorted_by_degree[:3]

        # Remove the vertex and the selected neighbors from the graph
        graph.remove_nodes_from([largest_degree_node] + selected_neighbors)

        # Increment the counter by 3
        counter += 3

    return counter


def bcc_heuristic(state, goal):
    graph = state.graph.copy()  # Clone the graph to avoid modifying the original
    tail_nodes = state.tail()  # Nodes to be removed
    graph.remove_nodes_from(tail_nodes)  # Remove tail nodes

    head = state.head

    # Find all bi-connected components and articulation points
    bcc = list(nx.biconnected_components(graph))

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

    # Find the path in the BCT from the BCC containing the head to the BCC containing the goal
    head_bcc = node_to_bcc.get(head, [None])[0]
    goal_bcc = node_to_bcc.get(goal, [None])[0]

    if head_bcc is None or goal_bcc is None:
        return 0

    # In the BCT, there can be only one path from the head_bcc to the goal_bcc
    # If there were more, the definition of BBC is defied
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


def bcc_snake_heuristic(state, goal):
    graph = state.graph
    head = state.head

    if goal == head: 
        return 0
    if goal in state.illegal: 
        return -1

    # Step 1: Compute Rn - the vertices in Qn that are reachable from state.head
    Qn = set(graph.nodes) - state.illegal
    if isinstance(goal,State):
        Qn = (Qn - goal.illegal)|{goal.head}
        goal = goal.head
    Qn_subgraph = graph.subgraph(Qn|{head}).copy()
    reachable_nodes = nx.single_source_shortest_path_length(Qn_subgraph, head)
    Rn = set(reachable_nodes.keys()) & Qn

    if goal not in Rn: 
        return -1

    # Step 2: Construct BCT from the graph induced by Rn U {state.head}
    Rn_graph = nx.induced_subgraph(graph,Rn | {head})
    BCCs = list(nx.biconnected_components(Rn_graph))
    cut_points = set(nx.articulation_points(Rn_graph))

    # Create the block-cut-point tree
    bct = nx.Graph()
    for i, block in enumerate(BCCs):
        bct.add_node(f"B{i}", vertices=block)  # Add each block as a node
        for vertex in block:
            if vertex in cut_points:
                bct.add_edge(f"B{i}", vertex)  # Connect blocks to cut-points

    block_of_head = next(b for b in BCCs if head in b)
    block_of_goal = next(b for b in BCCs if goal in b)

    head_block_node = [node for node, data in bct.nodes(data=True) if data.get("vertices") == block_of_head][0]
    goal_block_node = [node for node, data in bct.nodes(data=True) if data.get("vertices") == block_of_goal][0]

    try:
        bct_path = nx.shortest_path(bct, source=head_block_node, target=goal_block_node)
    except nx.NetworkXNoPath:
        return 0

    Bn = set()
    for element in bct_path:
        if not isinstance(element, int): # element might be a cut-point. in this case don't add its vertices.
            Bn.update(bct.nodes[element]["vertices"])

    # Step 3: Remove head, neighbors of head, and neighbors of goal from Bn (we will add 2 to the heuristic later)
    HGP = Bn - {head} - set(graph.neighbors(head)) - set(graph.neighbors(goal))
    HGP_graph = graph.subgraph(HGP).copy()

    # Step 4: count number of vertices in HGP, Y patterns count as 3.
    Y = Y_heuristic(HGP_graph)

    return Y + 2 # add 1 for neighbor of head, 1 for neighbor of goal


def find_component_containing_vertex(tric, vertex):
    # Iterate over triconnected components to find which contains the vertex
    for component in tric.get_triconnected_components():
        for edge in component:
            # Each component is represented as a tuple of (node1, node2, ...). Check if the vertex is part of any edge.
            if vertex in edge[:2]:  # Check the first two elements, which represent vertices in the edge
                return component
    return None


def mis_heuristic(state,goal):
    graph = state.graph.copy()  # Clone the graph to avoid modifying the original
    tail_nodes = state.tail()  # Nodes to be removed
    graph.remove_nodes_from(tail_nodes)  # Remove tail nodes

    head = state.head

    # Find all bi-connected components and articulation points
    BCCs = list(nx.biconnected_components(graph))
    articulation_points = set(nx.articulation_points(graph))

    # Special case: the whole graph is one biconnected component
    if len(BCCs) == 1:
        bcc_vertices = BCCs[0]
        bcc_subgraph = graph.subgraph(bcc_vertices).copy()

        # Compute SPQR-based heuristic directly
        return get_max_nodes_spqr_recursive(bcc_subgraph, head, goal, return_nodes=False) - 1

    # Create a mapping from nodes to BCCs
    node_to_bcc = {}
    for i, component_i in enumerate(BCCs):
        for node in component_i:
            if node not in node_to_bcc:
                node_to_bcc[node] = []
            node_to_bcc[node].append(i)

    # Find the BCCs containing head and goal
    head_bccs = node_to_bcc.get(head, [])
    goal_bccs = node_to_bcc.get(goal, [])

    # Check if head and goal are in the same BCC
    common_bcc = None
    for bcc_index in head_bccs:
        if bcc_index in goal_bccs:
            common_bcc = bcc_index
            break

    if common_bcc is not None:
        # Head and goal are in the same BCC
        bcc_vertices = BCCs[common_bcc]
        bcc_subgraph = graph.subgraph(bcc_vertices).copy()

        # Compute SPQR-based heuristic directly
        return get_max_nodes_spqr_recursive(bcc_subgraph, head, goal, return_nodes=False) - 1
        
    # Otherwise, proceed to find separate BCCs for head and goal
    head_bcc = head_bccs[0] if head_bccs else None
    goal_bcc = goal_bccs[0] if goal_bccs else None

    if head_bcc is None or goal_bcc is None:
        return 0

    # Create the BCT (Block-Cut Tree)
    bct = nx.Graph()
    for i, component_i in enumerate(BCCs):
        bct.add_node(i, vertices=component_i)  # Store vertices in BCC
    for i, component_i in enumerate(BCCs):
        for node in component_i:
            if node in articulation_points:
                for j, component_j in enumerate(BCCs):
                    if j > i and node in component_j:
                        bct.add_edge(i, j, articulation_point=node)

    # Find the path in the BCT from the BCC containing the head to the BCC containing the goal
    try:
        BCT_brach = nx.shortest_path(bct, head_bcc, goal_bcc)
    except nx.NetworkXNoPath:
        return 0

    # Special case: head is a cut-point, so the first BCC in the BCT_brach might be redundant
    if head == bct[BCT_brach[0]][BCT_brach[1]]["articulation_point"]:
        BCT_brach = BCT_brach[1:]

    # Handle the case where bcc_path contains only one node
    if len(BCT_brach) == 1:
        single_bcc_index = BCT_brach[0]
        bcc_vertices = bct.nodes[single_bcc_index]["vertices"]
        bcc_subgraph = graph.subgraph(bcc_vertices).copy()

        # Compute SPQR-based heuristic directly
        return get_max_nodes_spqr_recursive(bcc_subgraph, head, goal, return_nodes=False) - 1

    # Compute the SPQR-based MIS heuristic for each BCC in the path
    total_heuristic = 0

    for i in range(len(BCT_brach)):
        bcc_index = BCT_brach[i]
        bcc_vertices = bct.nodes[bcc_index]["vertices"]
        bcc_subgraph = graph.subgraph(bcc_vertices).copy()

        # Determine entry and exit points for the BCC
        if i == 0:
            # First BCC: entry is head, exit is the articulation point connecting to the next BCC
            in_node = head
            out_node = bct[bcc_index][BCT_brach[i + 1]]["articulation_point"]
        elif i == len(BCT_brach) - 1:
            # Last BCC: entry is the articulation point connecting to the previous BCC, exit is goal
            in_node = bct[bcc_index][BCT_brach[i - 1]]["articulation_point"]
            out_node = goal
        else:
            # Intermediate BCC: entry and exit are the articulation points connecting to adjacent BCCs
            in_node = bct[bcc_index][BCT_brach[i - 1]]["articulation_point"]
            out_node = bct[bcc_index][BCT_brach[i + 1]]["articulation_point"]

        # Compute SPQR-based heuristic
        spqr_nodes_count = get_max_nodes_spqr_recursive(bcc_subgraph, in_node, out_node, return_nodes=False)

        # Add the heuristic value from the SPQR tree of this BCC
        total_heuristic += spqr_nodes_count - 1

    # Combine results from all BCCs to form the h_MIS value
    return total_heuristic




def mis_snake_heuristic(state, goal, snake):
    graph = state.graph
    head = state.head
    path = state.path

    # Step 1: Compute Qn - the set of all legal vertices for the state
    Qn = set(graph.nodes) - state.illegal

    # Step 2: Compute Rn - the vertices in Qn that are reachable from state.head
    reachable_nodes = nx.single_source_shortest_path_length(graph, head)
    Rn = set(reachable_nodes.keys()) & Qn

    # Step 3: Construct BCT from the graph induced by Rn U {state.head}
    induced_subgraph = nx.induced_subgraph(graph,Rn | {head})
    # induced_subgraph = nx.induced_subgraph(graph,Rn.add(head))
    biconnected_components = list(nx.biconnected_components(induced_subgraph))
    cut_points = set(nx.articulation_points(induced_subgraph))

    # Create the block-cut-point tree
    bct = nx.Graph()
    for i, block in enumerate(biconnected_components):
        bct.add_node(f"B{i}", vertices=block)  # Add each block as a node
        for vertex in block:
            if vertex in cut_points:
                bct.add_edge(f"B{i}", vertex)  # Connect blocks to cut-points

    # Step 4: Find the branch in the BCT from the block containing head to the block containing goal
    block_of_head = next(b for b in biconnected_components if head in b)
    block_of_goal = next(b for b in biconnected_components if goal in b)

    head_block_node = [node for node, data in bct.nodes(data=True) if data.get("vertices") == block_of_head][0]
    goal_block_node = [node for node, data in bct.nodes(data=True) if data.get("vertices") == block_of_goal][0]

    if head_block_node == goal_block_node:
        print("head and goal are in the same BCC")
        bcc_subgraph = graph.subgraph(block_of_head).copy()
        bcc_subgraph_size = len(bcc_subgraph)
        # Compute SPQR-based heuristic directly
        if snake:
            return get_max_nodes_spqr_snake(bcc_subgraph, head, goal,y_filter=True, return_pairs=False) - 1
        else:
            return get_max_nodes_spqr_recursive(bcc_subgraph, head, goal, return_nodes=False) - 1
    else:
        print("TO DO")
    path_in_bct = nx.shortest_path(bct, source=head_block_node, target=goal_block_node)
    c=1


def heuristic(state, goal, heuristic_name, snake):
    if not isinstance(goal,int):
        if not snake: goal = max(goal)
        else: goal = State(state.graph,goal, snake)

    if heuristic_name == "heuristic0":
        return heuristic0(state)
    
    elif heuristic_name == "reachable_heuristic":
        return reachable_heuristic(state)
    
    elif heuristic_name == "bcc_heuristic":
        if snake: 
            return bcc_snake_heuristic(state, goal)
        else: 
            return bcc_heuristic(state,goal)
    elif heuristic_name == "mis_heuristic":
        if snake: 
            return mis_snake_heuristic(state, goal, snake)
        else: 
            return mis_heuristic(state,goal)
        
    # elif heuristic_name == "bct_is_heuristic":
    #     return bct_is_heuristic(state, goal, snake)
    else:
        print(f"Invalid heuristic name: {heuristic_name}")
        return 1 / 0
