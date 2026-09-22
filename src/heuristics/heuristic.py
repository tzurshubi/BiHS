import networkx as nx
from sage.all import *
from sage.graphs.connectivity import TriconnectivitySPQR
from sage.graphs.graph import Graph
from .h_mis import *
from models.state import State
from utils.utils import *
from collections import defaultdict


# def OLD_spqr_heuristic(state, goal, snake):
#     """
#     Compute the h_MIS heuristic, combining the BCC-based heuristic and SPQR tree-based exclusion pairs.

#     Parameters:
#     - state: The current search state, including the graph.
#     - goal: The target goal node in the graph.
#     - snake: Boolean indicating whether the "snake" constraint is applied.

#     Returns:
#     - int: The heuristic estimate for the longest simple path.
#     """
#     # Step 1: Compute h_BCC to identify and remove non-must-include vertices.
#     graph = state.graph.copy()  # Clone the graph to avoid modifying the original
#     tail_nodes = state.tail()  # Nodes to be removed
#     graph.remove_nodes_from(tail_nodes)  # Remove tail nodes

#     head = state.head

#     # Find all bi-connected components and articulation points
#     bcc = list(nx.biconnected_components(graph))
#     articulation_points = set(nx.articulation_points(graph))

#     # Special case: the whole graph is one biconnected component
#     if len(bcc) == 1:
#         bcc_vertices = bcc[0]
#         bcc_subgraph = graph.subgraph(bcc_vertices).copy()

#         # Compute SPQR-based heuristic directly
#         if not snake:
#             return get_max_nodes_spqr_recursive(bcc_subgraph, head, goal, return_nodes=False) - 1
#         else:
#             return get_max_nodes_spqr_snake(bcc_subgraph, head, goal,y_filter=True, return_pairs=False) - 1

#     # Create a mapping from nodes to BCCs
#     node_to_bcc = {}
#     for i, component_i in enumerate(bcc):
#         for node in component_i:
#             if node not in node_to_bcc:
#                 node_to_bcc[node] = []
#             node_to_bcc[node].append(i)

#     # Find the BCCs containing head and goal
#     head_bccs = node_to_bcc.get(head, [])
#     goal_bccs = node_to_bcc.get(goal, [])

#     # Check if head and goal are in the same BCC
#     common_bcc = None
#     for bcc_index in head_bccs:
#         if bcc_index in goal_bccs:
#             common_bcc = bcc_index
#             break

#     if common_bcc is not None:
#         # Head and goal are in the same BCC
#         bcc_vertices = bcc[common_bcc]
#         bcc_subgraph = graph.subgraph(bcc_vertices).copy()

#         # Compute SPQR-based heuristic directly
#         if not snake:
#             return get_max_nodes_spqr_recursive(bcc_subgraph, head, goal, return_nodes=False) - 1
#         else:
#             return get_max_nodes_spqr_snake(bcc_subgraph, head, goal, return_pairs=False) - 1

#     # Otherwise, proceed to find separate BCCs for head and goal
#     head_bcc = head_bccs[0] if head_bccs else None
#     goal_bcc = goal_bccs[0] if goal_bccs else None

#     if head_bcc is None or goal_bcc is None:
#         return 0

#     # Create the BCT (Block-Cut Tree)
#     bct = nx.Graph()
#     for i, component_i in enumerate(bcc):
#         bct.add_node(i, vertices=component_i)  # Store vertices in BCC
#     for i, component_i in enumerate(bcc):
#         for node in component_i:
#             if node in articulation_points:
#                 for j, component_j in enumerate(bcc):
#                     if j > i and node in component_j:
#                         bct.add_edge(i, j, articulation_point=node)

#     # Find the path in the BCT from the BCC containing the head to the BCC containing the goal
#     try:
#         bcc_path = nx.shortest_path(bct, head_bcc, goal_bcc)
#     except nx.NetworkXNoPath:
#         return 0

#     # Handle the case where bcc_path contains only one node
#     if len(bcc_path) == 1:
#         single_bcc_index = bcc_path[0]
#         bcc_vertices = bct.nodes[single_bcc_index]["vertices"]
#         bcc_subgraph = graph.subgraph(bcc_vertices).copy()

#         # Compute SPQR-based heuristic directly
#         if not snake:
#             return get_max_nodes_spqr_recursive(bcc_subgraph, head, goal, return_nodes=False) - 1
#         else:
#             return get_max_nodes_spqr_snake(bcc_subgraph, head, goal,y_filter=True, return_pairs=False) - 1

#     # Step 2: Compute the SPQR-based MIS heuristic for each BCC in the path
#     total_heuristic = 0

#     for i in range(len(bcc_path)):
#         bcc_index = bcc_path[i]
#         bcc_vertices = bct.nodes[bcc_index]["vertices"]
#         bcc_subgraph = graph.subgraph(bcc_vertices).copy()

#         # Determine entry and exit points for the BCC
#         if i == 0:
#             # First BCC: entry is head, exit is the articulation point connecting to the next BCC
#             in_node = head
#             out_node = bct[bcc_index][bcc_path[i + 1]]["articulation_point"]
#         elif i == len(bcc_path) - 1:
#             # Last BCC: entry is the articulation point connecting to the previous BCC, exit is goal
#             in_node = bct[bcc_index][bcc_path[i - 1]]["articulation_point"]
#             out_node = goal
#         else:
#             # Intermediate BCC: entry and exit are the articulation points connecting to adjacent BCCs
#             in_node = bct[bcc_index][bcc_path[i - 1]]["articulation_point"]
#             out_node = bct[bcc_index][bcc_path[i + 1]]["articulation_point"]

#         # Compute SPQR-based heuristic
#         if not snake:
#             spqr_nodes_count = get_max_nodes_spqr_recursive(
#                 bcc_subgraph, in_node, out_node, return_nodes=False
#             )
#         else:
#             spqr_nodes_count = get_max_nodes_spqr_snake(
#                 bcc_subgraph, in_node, out_node,y_filter=True, return_pairs=False
#             )

#         # Add the heuristic value from the SPQR tree of this BCC
#         total_heuristic += spqr_nodes_count - 1

#     # Combine results from all BCCs to form the h_MIS value
#     return total_heuristic


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
def f2f_reachable_heuristic(state_F, state_B, args=None):
    g = args.graph.copy()
    to_remove = [v for v in g.nodes if ((state_F.illegal >> v) & 1) or ((state_B.illegal >> v) & 1)]
    g.remove_nodes_from(to_remove)
    if state_F.head not in g or state_B.head not in g:
        print(f"Error in f2f_reachable_heuristic: head of F or B is not in the union graph after removing illegal vertices. state_F.head: {state_F.head}, state_B.head: {state_B.head}, g.nodes: {g.nodes}")
        return 0
    connected_component = nx.node_connected_component(g, state_F.head)
    return len(connected_component) if state_B.head in connected_component else 0

def heuristic0(state):
    return state.num_graph_vertices


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


def find_square(graph):
    # D is a dictionary to store pairs of neighbors and the vertex connecting them
    D = defaultdict(lambda: None)

    # Iterate over each vertex in the graph
    for v in graph.nodes():
        neighbors = list(graph.neighbors(v))
        
        # Check all pairs of neighbors (s, t)
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                s, t = neighbors[i], neighbors[j]
                
                # Ensure s < t for consistency (undirected graph)
                if s > t:
                    s, t = t, s

                if D[(s, t)] is None:
                    # Store the current vertex for this pair
                    D[(s, t)] = v
                else:
                    # Found a square: return the vertices
                    return v, s, t, D[(s, t)]

    # If no square is found, return None
    return None, None, None, None


def Y_heuristic(graph):
    counter = 0

    # Y patters
    while graph.number_of_nodes() > 0:
        # Find the vertex with the smallest degree that has a degree of at least 3
        smallest_degree_over_3_node = min(
            (node for node in graph.nodes if graph.degree(node) >= 3),
            key=lambda node: graph.degree(node),
            default=None
        )

        if smallest_degree_over_3_node is None:
            # No vertex with degree >= 3, stop the process
            break

        # Get the neighbors of the vertex
        neighbors = list(graph.neighbors(smallest_degree_over_3_node))

        # Sort the neighbors by degree and select the three with the lowest degree
        neighbors_sorted_by_degree = sorted(neighbors, key=lambda node: graph.degree(node))
        selected_neighbors = neighbors_sorted_by_degree[:3]

        # Remove the vertex and the selected neighbors from the graph
        graph.remove_nodes_from([smallest_degree_over_3_node] + selected_neighbors)

        # Increment the counter by 3
        counter += 3
    
    # Square pattern
    # while graph.number_of_nodes() > 0:
    #     a,b,c,d = find_square(graph)
    #     if a:
    #         graph.remove_nodes_from([a,b,c,d])
    #         counter += 3
    #     else:
    #         break

    return counter + len(graph)


def bcc_heuristic_paper(state, goal, graph=None):
    """
    Heuristic: add an edge between head (=state.head) and goal to form Q.
    Let B be the biconnected component in Q that contains both head and goal.
    Return |V(B)| - 1. If head/goal are absent or no such B exists, return 0.
    """

    head = getattr(state, "head", None)
    if head is None or goal is None or head == goal:
        return 0

    # Copy graph and remove tail nodes
    G = state.graph if graph is None else graph
    tail_nodes = set(state.tail())
    if head in tail_nodes or goal in tail_nodes:
        return 0
    if graph is None: G.remove_nodes_from(tail_nodes)

    if head not in G or goal not in G:
        return 0

    # Add the probe edge
    s_t_edge_existed = G.has_edge(head, goal)
    if not s_t_edge_existed:
        G.add_edge(head, goal)

    # Find all biconnected components
    for comp in nx.biconnected_components(G):
        if head in comp and goal in comp:
            if not s_t_edge_existed: G.remove_edge(head, goal)
            return max(0, len(comp) - 1)
        
    if not s_t_edge_existed: G.remove_edge(head, goal)
    return 0


def F2F_bcc_heuristic(state_F, state_B, graph):
    """
    Heuristic: add an edge between head (=state.head) and goal to form Q.
    Let B be the biconnected component in Q that contains both head and goal.
    Return |V(B)| - 1. If head/goal are absent or no such B exists, return 0.
    """
    s = getattr(state_F, "head", None)
    t = getattr(state_B, "head", None)
    if s not in graph or t not in graph:
        return 0

    # Add the probe edge
    s_t_edge_existed = graph.has_edge(s, t)
    if not s_t_edge_existed:
        graph.add_edge(s, t)
    
    # Find all biconnected components
    # bccs = list(nx.biconnected_components(G)) # for debug
    # print(f"graph has {len(bccs)} biconnected components")
    for bcc in nx.biconnected_components(graph):
        if s in bcc and t in bcc:
            # h = max(0, len(comp) - 1) # for debug
            if not s_t_edge_existed: graph.remove_edge(s, t)
            return max(0, len(bcc) - 1)
        
    if not s_t_edge_existed: graph.remove_edge(s, t)
    return 0

def F2F_bcc_snake_heuristic(state_F, state_B, graph):
    """
    Heuristic: add an edge between head (=state.head) and goal to form Q.
    Let B be the biconnected component in Q that contains both head and goal.
    Return |V(B)| - 1. If head/goal are absent or no such B exists, return 0.\
    v should be in the path from s to t.
    """
    calc_Y_heuristic = True    # True # False
    s = getattr(state_F, "head", None)
    t = getattr(state_B, "head", None)
    if s not in graph or t not in graph:
        return 0

    # Add the probe edge (necessarily there isn't one before because of the snake constraint)
    graph.add_edge(s, t)
    
    # Find all biconnected components
    # bccs = list(nx.biconnected_components(graph)) # for debug
    # print(f"graph has {len(bccs)} biconnected components")
    for bcc in nx.biconnected_components(graph):
        if s in bcc and t in bcc:
            # snake head heuristic inside this BCC
            graph.remove_edge(s, t)
            nodes_to_remove = [n for n in graph if n not in bcc or n in graph[s] or n in graph[t]]
            if s in nodes_to_remove or t in nodes_to_remove:
                raise ValueError(f"Error: trying to remove head or goal from the graph. s: {s}, t: {t}, nodes_to_remove: {nodes_to_remove}")
            
            if calc_Y_heuristic: 
                graph.remove_nodes_from(nodes_to_remove)
                return Y_heuristic(graph) + 1 # add 1 for neighbor of head or neighbor of goal
            else: return len(graph.nodes) - len(nodes_to_remove) + 1 # max(0, len(graph.nodes) + 1)
            # return max(0, len(comp) - 1)
    return 0

def F2F_bcc_snake_heuristic_paper(state_F, state_B):
    calc_Y_heuristic = True    # True # False
    info = {}
    graph = state_F.graph
    head_F = state_F.head
    head_B = state_B.head

    # 1. Base case: The frontiers have exactly met
    if head_F == head_B: 
        return 0, info
        
    # 2. Intersection Check: If one head is in the illegal footprint of the other, 
    # it means connecting them would cause a chord or intersection.
    if isinstance(state_F.illegal, int):
        if ((state_F.illegal >> head_B) & 1) or ((state_B.illegal >> head_F) & 1):
            return -1, info
    else:
        if head_B in state_F.illegal or head_F in state_B.illegal:
            return -1, info

    # 3. Adjacency Check: If they are adjacent and legal, the remaining distance is exactly 1 edge.
    if graph.has_edge(head_F, head_B):
        return 1, info

    # 4. Compute Qn (Nodes available to BOTH frontiers)
    if isinstance(state_F.illegal, int):
        # Combine both illegal bitmasks
        illegal_mask = state_F.illegal | state_B.illegal
        Qn = {v for v in graph.nodes if not (illegal_mask & (1 << v))}
    else:
        # Set subtraction
        Qn = set(graph.nodes) - state_F.illegal - state_B.illegal

    # 5. Create Subgraph and add dummy edge between heads
    Qn_subgraph = graph.subgraph(Qn | {head_F, head_B}).copy()
    Qn_subgraph.add_edge(head_F, head_B)

    # 6. Find Biconnected Component bridging the two heads
    Bn = None
    for comp in nx.biconnected_components(Qn_subgraph):
        if head_F in comp and head_B in comp:
            Bn = set(comp)
            break
            
    # If no component contains both, they are disconnected. 
    # (Replaced the original 0/0 crash with a graceful -1 dead-end return)
    if not Bn:
        return -1, info
    
    info['Bn'] = len(Bn)
    
    # 7. Shrink to HGP (Heuristic Graph Pattern)
    # Remove everything not in the bridging component
    Qn_subgraph.remove_nodes_from(set(Qn_subgraph.nodes) - Bn)
    
    # Remove both heads and their respective neighbors
    neighbors_F = set(graph.neighbors(head_F))
    neighbors_B = set(graph.neighbors(head_B))
    Qn_subgraph.remove_nodes_from({head_F, head_B} | neighbors_F | neighbors_B)
    
    info['HGP'] = len(Qn_subgraph.nodes)
    Y = len(Qn_subgraph.nodes)

    # 8. Compute Y patterns
    if calc_Y_heuristic: 
        Y = Y_heuristic(Qn_subgraph) 

    # Add 3 (1 edge from F to its neighbor n_f , 1 edge from n_f into the Y nodes , Y-1 edges connecting the Y nodes together, 1 edge from the Y nodes to B's neighbor n_b, 1 edge from n_b to B. Total maximum edges = 1 + 1 + (Y - 1) + 1 + 1 = Y + 3
    return Y + 3, info

def F2F_bcc_heuristic_solVert(state_F, state_B, v, graph):
    """
    Heuristic: add an edge between head (=state.head) and goal to form Q.
    Let B be the biconnected component in Q that contains both head and goal.
    Return |V(B)| - 1. If head/goal are absent or no such B exists, return 0.
    """
    s = getattr(state_F, "head", None)
    t = getattr(state_B, "head", None)
    if s not in graph or t not in graph or v not in graph:
        return 0

    # Add the probe edge
    G = graph.copy()
    G.add_edge(s, t)
    
    # Find all biconnected components
    # bccs = list(nx.biconnected_components(G)) # for debug
    # print(f"graph has {len(bccs)} biconnected components")
    for comp in nx.biconnected_components(G):
        if s in comp and t in comp and v in comp:
            # h = max(0, len(comp) - 1) # for debug
            return max(0, len(comp) - 1)

    return 0

def F2F_bcc_snake_heuristic_solVert(state_F, state_B, v, graph):
    """
    Heuristic: add an edge between head (=state.head) and goal to form Q.
    Let B be the biconnected component in Q that contains both head and goal.
    Return |V(B)| - 1. If head/goal are absent or no such B exists, return 0.\
    v should be in the path from s to t.
    """
    calc_Y_heuristic = True    # True # False
    s = getattr(state_F, "head", None)
    t = getattr(state_B, "head", None)
    if s not in graph or t not in graph or v not in graph:
        return 0

    # Add the probe edge (necessarily there isn't one before because of the snake constraint)
    graph.add_edge(s, t)
    
    # Find all biconnected components
    # bccs = list(nx.biconnected_components(graph)) # for debug
    # print(f"graph has {len(bccs)} biconnected components")
    for bcc in nx.biconnected_components(graph):
        if s in bcc and t in bcc and v in bcc:
            # snake head heuristic inside this BCC
            graph.remove_edge(s, t)
            nodes_to_remove = [n for n in graph if n not in bcc or n in graph[s] or n in graph[t]]
            if s in nodes_to_remove or t in nodes_to_remove:
                raise ValueError(f"Error: trying to remove head or goal from the graph. s: {s}, t: {t}, nodes_to_remove: {nodes_to_remove}")
            graph.remove_nodes_from(nodes_to_remove)
            
            if calc_Y_heuristic: return Y_heuristic(graph) + 1 # add 1 for neighbor of head or neighbor of goal
            else: return max(0, len(graph.nodes) + 1)
            # return max(0, len(comp) - 1)
    return 0






from collections import defaultdict
from itertools import combinations
import networkx as nx


# ============================================================
# Small helper functions
# ============================================================

def _illegal_contains(illegal, v):
    if isinstance(illegal, int):
        return bool((illegal >> v) & 1)
    return v in illegal


def _common_remaining_graph(state_F, state_B):
    """
    Vertices that are still available to BOTH frontiers.
    The two current heads are explicitly retained.
    """
    G = state_F.graph
    s = state_F.head
    t = state_B.head

    if isinstance(state_F.illegal, int):
        illegal_mask = state_F.illegal | state_B.illegal

        Qn = {
            v for v in G.nodes
            if not (illegal_mask & (1 << v))
        }

    else:
        Qn = (
            set(G.nodes)
            - set(state_F.illegal)
            - set(state_B.illegal)
        )

    return G.subgraph(Qn | {s, t}).copy()


# ============================================================
# Block-cut path
# ============================================================

def _block_cut_path(G, s, t):
    """
    Find the biconnected blocks on the block-cut-tree path
    from s to t.

    Returns:
        [
            (block_graph, entry_vertex, exit_vertex),
            ...
        ]

    or None if s and t are disconnected.
    """

    if s == t:
        return []

    if s not in G or t not in G:
        return None

    if not nx.has_path(G, s, t):
        return None

    components = [
        set(c)
        for c in nx.biconnected_components(G)
    ]

    if not components:
        return None

    # Record which blocks contain each vertex
    membership = defaultdict(list)

    for i, comp in enumerate(components):
        for v in comp:
            membership[v].append(i)

    articulation = set(nx.articulation_points(G))

    # Construct block-cut tree.
    #
    # ('B', i) = biconnected block i
    # ('A', v) = articulation vertex v

    T = nx.Graph()

    for i, comp in enumerate(components):

        block_node = ("B", i)
        T.add_node(block_node)

        for v in comp & articulation:
            T.add_edge(
                block_node,
                ("A", v)
            )

    def endpoint_node(v):

        if v in articulation:
            return ("A", v)

        if not membership[v]:
            return None

        return ("B", membership[v][0])

    src = endpoint_node(s)
    dst = endpoint_node(t)

    if src is None or dst is None:
        return None

    # Special case: both endpoints belong to one block
    shared = set(membership[s]) & set(membership[t])

    if shared:
        i = next(iter(shared))

        return [
            (
                G.subgraph(components[i]).copy(),
                s,
                t
            )
        ]

    if src not in T or dst not in T:
        return None

    path = nx.shortest_path(T, src, dst)

    result = []

    for pos, node in enumerate(path):

        if node[0] != "B":
            continue

        block_index = node[1]
        comp = components[block_index]

        # Entry into this block
        if pos == 0:
            entry = s

        else:
            previous = path[pos - 1]

            if previous[0] == "A":
                entry = previous[1]
            else:
                entry = s

        # Exit from this block
        if pos == len(path) - 1:
            exit_ = t

        else:
            next_node = path[pos + 1]

            if next_node[0] == "A":
                exit_ = next_node[1]
            else:
                exit_ = t

        B = G.subgraph(comp).copy()

        result.append(
            (B, entry, exit_)
        )

    return result


# ============================================================
# Sage/SPQR helper functions
# ============================================================

def _sage_edges(H):
    """
    Return Sage skeleton edges as:
        (u, v, label)
    """

    try:
        return list(
            H.edges(sort=False, labels=True)
        )

    except TypeError:
        # Older Sage versions
        return list(
            H.edges(labels=True)
        )


def _spqr_kind(node):
    """
    Normalize Sage SPQR component names.
    """

    kind = str(node[0]).upper()

    aliases = {
        "POLYGON": "S",
        "BOND": "P",
        "TRICONNECTED": "R",
    }

    return aliases.get(kind, kind)


def _other_spqr_node(
    label_to_nodes,
    label,
    current
):
    """
    A virtual edge occurs in exactly two SPQR skeletons.
    Return the component on the other side.
    """

    nodes = label_to_nodes[label]

    others = [
        node
        for node in nodes
        if node != current
    ]

    if len(others) != 1:

        raise RuntimeError(
            f"Virtual edge {label!r} should occur "
            f"in exactly two SPQR components."
        )

    return others[0]


# ============================================================
# 2-edge cut helper
# ============================================================

def _two_virtual_edges_disconnect(
    cminus_edges,
    vertices,
    edge1,
    edge2
):
    """
    Test whether edge1 and edge2 form a 2-edge cut.

    The paper uses FIND2EDGECUTS in linear time.
    This implementation checks pairs directly.

    Therefore it is simpler, but slower than the paper's
    optimized implementation.
    """

    H = nx.MultiGraph()

    H.add_nodes_from(vertices)

    edge_key = {}

    for i, (u, v, label) in enumerate(cminus_edges):

        H.add_edge(
            u,
            v,
            key=i
        )

        if label is not None:
            edge_key[label] = i

    for edge in (edge1, edge2):

        u, v, label = edge

        key = edge_key[label]

        H.remove_edge(
            u,
            v,
            key=key
        )

    if H.number_of_nodes() <= 1:
        return False

    return not nx.is_connected(H)


# ============================================================
# Exact h_MIS for ONE biconnected block
# ============================================================

def _hmis_spqr_block_sage(
    B,
    s,
    t,
    opt_constraint=None
):
    """
    Compute the 2024 exact SPQR/MIS heuristic h_MIS
    for one biconnected block.

    The returned value is an upper bound on the number
    of remaining EDGES from s to t inside B.

    For LSP:
        opt_constraint = None

    For Snake:
        opt_constraint(u, v) should be True when u-v is
        a real edge of B.

    Based on Algorithm 1 of:
        Dahan, Tabib, Shimony & Dinitz.
    """

    if s == t:
        return 0

    # For Snake, adjacent endpoints imply that the only legal
    # connection is the direct edge.
    if (
        opt_constraint is not None
        and B.has_edge(s, t)
        and opt_constraint(s, t)
    ):
        return 1

    # --------------------------------------------------------
    # Sage is needed for SPQR decomposition
    # --------------------------------------------------------

    try:

        from sage.all import Graph as SageGraph
        from sage.graphs.connectivity import spqr_tree

    except ImportError as e:

        raise ImportError(
            "This heuristic requires SageMath for the SPQR "
            "decomposition. Run the program with Sage's Python, "
            "for example:\n\n"
            "    sage -python your_script.py"
        ) from e

    # --------------------------------------------------------
    # Build Sage graph
    # --------------------------------------------------------

    SG = SageGraph(
        multiedges=True,
        loops=False
    )

    SG.add_vertices(
        list(B.nodes)
    )

    for u, v in B.edges:
        SG.add_edge(u, v, None)

    # --------------------------------------------------------
    # Add auxiliary (s,t) edge
    #
    # This is the key construction from the 2024 paper.
    # --------------------------------------------------------

    AUX = "__F2F_SPQR_AUX_EDGE__"

    SG.add_edge(
        s,
        t,
        AUX
    )

    # --------------------------------------------------------
    # SPQR decomposition
    # --------------------------------------------------------

    T = spqr_tree(SG)

    spqr_nodes = list(T)

    # Map every virtual edge label to the two components
    # in which it occurs.

    label_to_nodes = defaultdict(list)

    for node in spqr_nodes:

        skeleton = node[1]

        for u, v, label in _sage_edges(skeleton):

            if (
                label is not None
                and label != AUX
            ):
                label_to_nodes[label].append(node)

    # --------------------------------------------------------
    # Root the SPQR tree at the component containing
    # the auxiliary (s,t) edge.
    # --------------------------------------------------------

    root = None

    for node in spqr_nodes:

        skeleton = node[1]

        for u, v, label in _sage_edges(skeleton):

            if (
                label == AUX
                and (
                    (u == s and v == t)
                    or
                    (u == t and v == s)
                )
            ):

                root = node
                break

        if root is not None:
            break

    if root is None:

        raise RuntimeError(
            "Could not find the auxiliary (s,t) "
            "edge in the SPQR tree."
        )

    # ========================================================
    # Recursive TRAVERSE from Algorithm 1
    # ========================================================

    def traverse(
        node,
        parent,
        incoming_label,
        incoming_uv
    ):

        s0, t0 = incoming_uv

        # ----------------------------------------------------
        # OPTCONSTRAINT
        #
        # For Snake:
        # if s0 and t0 are adjacent by a REAL graph edge,
        # this subtree contributes no internal vertices.
        # ----------------------------------------------------

        if (
            opt_constraint is not None
            and opt_constraint(s0, t0)
        ):
            return 0

        skeleton = node[1]

        all_edges = _sage_edges(skeleton)

        # ----------------------------------------------------
        # C^- = C minus the incoming edge
        # ----------------------------------------------------

        cminus_edges = []

        removed = False

        for edge in all_edges:

            u, v, label = edge

            if incoming_label == AUX:

                is_incoming = (
                    label == AUX
                    and (
                        (u == s0 and v == t0)
                        or
                        (u == t0 and v == s0)
                    )
                )

            else:

                is_incoming = (
                    label == incoming_label
                )

            if is_incoming and not removed:

                removed = True
                continue

            cminus_edges.append(edge)

        if not removed:

            raise RuntimeError(
                "Incoming edge not found in "
                "SPQR skeleton."
            )

        # ----------------------------------------------------
        # Virtual edges belonging to child subtrees
        # ----------------------------------------------------

        virtual_edges = [

            edge

            for edge in cminus_edges

            if (
                edge[2] is not None
                and edge[2] != AUX
                and edge[2] in label_to_nodes
            )
        ]

        # Recursively compute the MIS contribution
        # hanging from every virtual edge.

        child_value = {}

        for u, v, label in virtual_edges:

            child = _other_spqr_node(
                label_to_nodes,
                label,
                node
            )

            if child == parent:
                continue

            child_value[label] = traverse(
                child,
                node,
                label,
                (u, v)
            )

        component_type = _spqr_kind(node)

        vertices = list(
            skeleton.vertices()
        )

        # ----------------------------------------------------
        # Q component
        # ----------------------------------------------------

        if component_type == "Q":
            return 0

        # ----------------------------------------------------
        # P component
        #
        # Only ONE parallel subtree can contribute.
        # ----------------------------------------------------

        if component_type == "P":

            return max(
                child_value.values(),
                default=0
            )

        # ----------------------------------------------------
        # S component
        #
        # All subtrees can contribute.
        # ----------------------------------------------------

        if component_type == "S":

            return (
                max(0, len(vertices) - 2)
                +
                sum(child_value.values())
            )

        # ----------------------------------------------------
        # R component
        # ----------------------------------------------------

        if component_type != "R":

            raise RuntimeError(
                f"Unknown SPQR component type: "
                f"{node[0]!r}"
            )

        # All ordinary internal vertices of C
        total = max(
            0,
            len(vertices) - 2
        )

        remaining_labels = set(
            child_value
        )

        edge_by_label = {
            edge[2]: edge
            for edge in virtual_edges
            if edge[2] in child_value
        }

        def incident(edge, x):
            return (
                edge[0] == x
                or edge[1] == x
            )

        # ====================================================
        # R CASE 1
        #
        # Virtual edges all incident on s0 or on t0
        # form an exclusion clique.
        #
        # Therefore only the largest subtree contributes.
        # ====================================================

        for x in (s0, t0):

            group = [

                label

                for label in list(
                    remaining_labels
                )

                if incident(
                    edge_by_label[label],
                    x
                )
            ]

            if len(group) >= 2:

                total += max(
                    child_value[label]
                    for label in group
                )

                remaining_labels.difference_update(
                    group
                )

        # ====================================================
        # R CASE 2
        #
        # Pairs of virtual edges forming a 2-edge cut.
        # ====================================================

        labels = list(
            remaining_labels
        )

        cut_pairs = []

        for a, b in combinations(
            labels,
            2
        ):

            edge_a = edge_by_label[a]
            edge_b = edge_by_label[b]

            both_at_s = (
                incident(edge_a, s0)
                and incident(edge_b, s0)
            )

            both_at_t = (
                incident(edge_a, t0)
                and incident(edge_b, t0)
            )

            # These belong to case 1 instead.
            if both_at_s or both_at_t:
                continue

            if _two_virtual_edges_disconnect(
                cminus_edges,
                vertices,
                edge_a,
                edge_b
            ):

                cut_pairs.append(
                    (a, b)
                )

        # The relevant 2-edge cuts in an R component
        # are edge-disjoint.

        for a, b in cut_pairs:

            if (
                a in remaining_labels
                and b in remaining_labels
            ):

                total += max(
                    child_value[a],
                    child_value[b]
                )

                remaining_labels.remove(a)
                remaining_labels.remove(b)

        # ====================================================
        # Everything not involved in an exclusion clique
        # contributes additively.
        # ====================================================

        total += sum(
            child_value[label]
            for label in remaining_labels
        )

        return total

    # h_MIS = alpha(G_ex) + 1
    return (
        traverse(
            root,
            None,
            AUX,
            (s, t)
        )
        + 1
    )


# ============================================================
# FRONT-TO-FRONT SPQR HEURISTIC
# ============================================================

def F2F_spqr_snake_heuristic_paper(
    state_F,
    state_B,
    *,
    combine_with_bcc_y=True
):
    """
    Front-to-front SPQR heuristic for Snake.

    Returns:
        heuristic_value, info

    Interpretation:
        The two current heads are the temporary endpoints
        s' and t'.

    The heuristic:
        1. combines the illegal footprints of F and B,
        2. keeps only the common remaining graph,
        3. finds the biconnected blocks on the s'-t' path,
        4. computes exact SPQR h_MIS for every block,
        5. sums the block bounds.

    If combine_with_bcc_y=True, it also takes the minimum
    with F2F_bcc_snake_heuristic_paper().  This is a safe
    way to retain your Y-pattern bound without double
    counting it with SPQR exclusion pairs.
    """

    info = {}

    G = state_F.graph

    head_F = state_F.head
    head_B = state_B.head

    # ========================================================
    # 1. Heads have met
    # ========================================================

    if head_F == head_B:
        return 0, info

    # ========================================================
    # 2. Intersection / illegal-footprint test
    # ========================================================

    if (
        _illegal_contains(
            state_F.illegal,
            head_B
        )
        or
        _illegal_contains(
            state_B.illegal,
            head_F
        )
    ):
        return -1, info

    # ========================================================
    # 3. Adjacent heads
    #
    # For Snake this really IS exactly one edge:
    #
    # if we leave head_F through another vertex, then
    # head_B becomes illegal because it is a neighbor of
    # head_F.
    # ========================================================

    if G.has_edge(
        head_F,
        head_B
    ):
        return 1, info

    # ========================================================
    # 4. Graph available to both search directions
    # ========================================================

    remaining_graph = _common_remaining_graph(
        state_F,
        state_B
    )

    if (
        head_F not in remaining_graph
        or
        head_B not in remaining_graph
        or
        not nx.has_path(
            remaining_graph,
            head_F,
            head_B
        )
    ):
        return -1, info

    # ========================================================
    # 5. Biconnected blocks on the front-to-front path
    #
    # This corresponds to Algorithm 1 in the 2022 paper.
    # ========================================================

    block_path = _block_cut_path(
        remaining_graph,
        head_F,
        head_B
    )

    if not block_path:
        return -1, info

    info["num_blocks"] = len(
        block_path
    )

    info["block_sizes"] = [
        len(B)
        for B, _, _ in block_path
    ]

    # ========================================================
    # 6. SPQR bound in every block
    # ========================================================

    block_bounds = []

    for B, entry, exit_ in block_path:

        if entry == exit_:

            h_block = 0

        elif B.has_edge(
            entry,
            exit_
        ):

            # Snake OPTCONSTRAINT:
            # adjacent entry/exit means internal vertices
            # of this block cannot be used.
            h_block = 1

        elif len(B) <= 2:

            h_block = 1

        else:

            h_block = _hmis_spqr_block_sage(

                B,
                entry,
                exit_,

                # For Snake:
                # OPTCONSTRAINT(G,x,y) is true exactly
                # when (x,y) is an edge of the graph.
                opt_constraint=(
                    lambda u, v, _B=B:
                    _B.has_edge(u, v)
                )
            )

        block_bounds.append(
            h_block
        )

    # The paper sums the heuristic over blocks.
    h_spqr = sum(
        block_bounds
    )

    info["SPQR_blocks"] = (
        block_bounds
    )

    info["SPQR"] = h_spqr

    # ========================================================
    # 7. Optionally combine with your current BCC + Y bound
    #
    # For MAX search, both are UPPER bounds.
    #
    # min(upper_bound_1, upper_bound_2)
    #
    # is therefore still an admissible upper bound.
    # ========================================================

    if combine_with_bcc_y:

        bcc_function = globals().get(
            "F2F_bcc_snake_heuristic_paper"
        )

        if bcc_function is not None:

            h_bcc_y, bcc_info = (
                bcc_function(
                    state_F,
                    state_B
                )
            )

            info["BCC_Y"] = h_bcc_y
            info["BCC_Y_info"] = bcc_info

            if h_bcc_y == -1:
                return -1, info

            h_combined = min(
                h_spqr,
                h_bcc_y
            )

            info["combined"] = (
                h_combined
            )

            return (
                h_combined,
                info
            )

    return h_spqr, info






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

def bcc_snake_heuristic_paper(state, goal):
    calc_Y_heuristic = True    # True # False
    info = {}
    graph = state.graph
    head = state.head

    if goal == head: 
        return 0, info
    # this line means: if goal in state.illegal:
    if (state.illegal >> goal) & 1: 
        return -1, info
    if graph.has_edge(head, goal):
        return 1, info

    # Step 1: Compute Rn - vertices reachable from state.head
    if isinstance(state.illegal, int):
        Qn = {v for v in graph.nodes if not (state.illegal & (1 << v))}
    else:
        Qn = set(graph.nodes) - state.illegal

    if isinstance(goal, State):
        if isinstance(goal.illegal, int):
            Qn = {v for v in Qn if not (goal.illegal & (1 << v))}
        else:
            Qn = Qn - goal.illegal
        Qn.add(goal.head)
        goal = goal.head

    Qn_subgraph = graph.subgraph(Qn | {head}).copy()
    Qn_subgraph.add_edge(head, goal)

    # reachable_nodes = nx.single_source_shortest_path_length(Qn_subgraph, head)
    # Rn = set(reachable_nodes.keys()) & Qn

    # if goal not in Rn: 
    #     return -1

    # # Step 2: Construct BCT from the graph induced by Rn U {state.head}
    # Rn_graph = nx.induced_subgraph(graph,Rn | {head})
    # if head not in Rn_graph or goal not in Rn_graph:
    #     return 0/0
    
    # Rn_graph.add_edge(head, goal)
    Bn = None
    for comp in nx.biconnected_components(Qn_subgraph): #(Rn_graph)
        if head in comp and goal in comp:
            Bn = set(comp)
            break
    if not Bn:
        return 0/0
    
    info['Bn'] = len(Bn)
    # Step 3: Remove head, neighbors of head, and neighbors of goal from Bn (we will add 2 to the heuristic later)
    # HGP = Bn - {head} - set(graph.neighbors(head)) - set(graph.neighbors(goal))
    Qn_subgraph.remove_nodes_from(set(Qn_subgraph.nodes) - Bn) # keep only Bn
    Qn_subgraph.remove_nodes_from({head} | set(graph.neighbors(head)) | set(graph.neighbors(goal)))
    info['HGP'] = len(Qn_subgraph.nodes)
    Y = len(Qn_subgraph.nodes)
    # HGP_graph = Qn_subgraph.subgraph(HGP)

    # Step 4: count number of vertices in HGP, Y patterns count as 3.
    if calc_Y_heuristic: Y = Y_heuristic(Qn_subgraph) 

    return Y + 2, info # add 1 for neighbor of head, 1 for neighbor of goal

def bcc_snake_heuristic(state, goal):
    calc_Y_heuristic = True
    info = {}
    graph = state.graph
    head = state.head

    if goal == head: 
        return 0, info
    if goal in state.illegal: 
        return -1, info

    # Step 1: Compute Rn - the vertices in Qn that are reachable from state.head
    Qn = set(graph.nodes) - state.illegal
    if isinstance(goal,State):
        Qn = (Qn - goal.illegal)|{goal.head}
        goal = goal.head
    Qn_subgraph = graph.subgraph(Qn|{head}).copy()
    reachable_nodes = nx.single_source_shortest_path_length(Qn_subgraph, head)
    Rn = set(reachable_nodes.keys()) & Qn

    if goal not in Rn: 
        return -1, info

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
        return 0, info

    Bn = set()
    for element in bct_path:
        if not isinstance(element, int): # element might be a cut-point. in this case don't add its vertices.
            Bn.update(bct.nodes[element]["vertices"])
    info['Bn'] = len(Bn)

    # Step 3: Remove head, neighbors of head, and neighbors of goal from Bn (we will add 2 to the heuristic later)
    HGP = Bn - {head} - set(graph.neighbors(head)) - set(graph.neighbors(goal))
    info['HGP'] = len(HGP)
    HGP_graph = graph.subgraph(HGP).copy()

    # Step 4: count number of vertices in HGP, Y patterns count as 3.
    if calc_Y_heuristic: Y = Y_heuristic(HGP_graph)
    else: Y = len(HGP)

    # print(f"len(HGP)={len(HGP)} , Y={Y}")

    return Y + 2, info # add 1 for neighbor of head, 1 for neighbor of goal

def find_component_containing_vertex(tric, vertex):
    # Iterate over triconnected components to find which contains the vertex
    for component in tric.get_triconnected_components():
        for edge in component:
            # Each component is represented as a tuple of (node1, node2, ...). Check if the vertex is part of any edge.
            if vertex in edge[:2]:  # Check the first two elements, which represent vertices in the edge
                return component
    return None

def heuristic(state, goal, heuristic_name, snake, args=None, h_graph=None):
    # print(f"Calculating heuristic {heuristic_name} for {state.materialize_path()} and {goal.materialize_path() if isinstance(goal, State) else goal} with snake={snake}")
    if heuristic_name == "heuristic0": return len(h_graph.nodes) if h_graph else len(state.graph.nodes)
    #F2F heuristics
    if isinstance(goal, State) and isinstance(state, State): #F2F heuristic
        if heuristic_name == "reachable_heuristic":
            return f2f_reachable_heuristic(state, goal, args)
        elif heuristic_name == "bcc_heuristic":
            if args.solution_vertices and len(args.solution_vertices) > 1:
                if snake: return F2F_bcc_snake_heuristic_solVert(state, goal, args.solution_vertices[1], h_graph)
                else: return F2F_bcc_heuristic_solVert(state, goal, args.solution_vertices[1], h_graph)
            else:
                if snake: 
                    return F2F_bcc_snake_heuristic_paper(state, goal)[0]
                    # return F2F_bcc_snake_heuristic(state, goal, h_graph)
                    # print(f"F2F_bcc_snake_heuristic: {h}")
                    # return h
                else: return F2F_bcc_heuristic(state, goal, h_graph)
        elif heuristic_name == "mis_heuristic":
            if snake: 
                spqr_heuristic = F2F_spqr_snake_heuristic_paper(state, goal)[0]
                # bcc_heuristic = F2F_bcc_snake_heuristic_paper(state, goal)[0]
                # args.stats['h_values'].append((2*state.g, spqr_heuristic, bcc_heuristic, bcc_heuristic - spqr_heuristic))
                return spqr_heuristic
            else: return mis_heuristic(state,goal)
        
    if not isinstance(goal,int):
        if not snake: goal = max(goal)
        else: goal = State(state.graph, goal, [], snake)
    
    #F2E heuristic
    if heuristic_name == "heuristic0":
        return heuristic0(state)
    
    elif heuristic_name == "reachable_heuristic":
        return reachable_heuristic(state)
    
    elif heuristic_name == "bcc_heuristic":
        if snake: 
            h, info = bcc_snake_heuristic_paper(state, goal)
            return h # return bcc_snake_heuristic(state, goal)
        else: 
            return bcc_heuristic_paper(state, goal, h_graph) # return bcc_heuristic(state,goal)
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
