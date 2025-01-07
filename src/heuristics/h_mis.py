import itertools
import networkx as nx
from sage.graphs.connectivity import spqr_tree
from sage.graphs.connectivity import TriconnectivitySPQR
from sage.graphs.graph import Graph
import random


# Compute the difference between two lists, returning elements that are in the first list but not in the second
def diff(li1, li2):
    """
    Compute the difference between two lists, returning elements that are in the first list but not in the second.

    This function converts the lists to sets to find the difference and then converts the result back to a list.

    Parameters:
    - li1 (list): The first list from which elements will be taken.
    - li2 (list): The second list whose elements will be excluded.

    Returns:
    - list: A list containing elements that are present in `li1` but not in `li2`.
    """
    return list(set(li1) - set(li2))


# Compute the intersection of two lists, returning elements that are common to both
def intersection(lst1, lst2):
    """
    Compute the intersection of two lists, returning elements that are common to both.

    This function uses a set to speed up the membership testing, making it efficient for large lists.

    Parameters:
    - lst1 (list): The first list.
    - lst2 (list): The second list.

    Returns:
    - list: A list containing elements that are present in both `lst1` and `lst2`.
    """
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3


# Flatten a list of lists into a single list
def flatten(lst):
    """
    Flatten a list of lists into a single list.

    This function takes a list of lists and concatenates all sub-lists into a single list, effectively "flattening" the structure.

    Parameters:
    - lst (list of lists): A list where each element is a list.

    Returns:
    - list: A flattened list containing all elements from the sub-lists.
    """
    res = []
    for x in lst:
        res += x
    return res


# Find pairs of edges that, when removed, disconnect the graph
def get_relevant_cuts(graph, possible_edges):
    """
    Find pairs of edges that, when removed, disconnect the graph.

    This function iterates through all pairs of edges from the list of possible edges and checks
    if removing a pair of edges would result in a disconnected graph. Such pairs are considered
    as cut edges.

    Parameters:
    - graph (networkx.Graph): The input graph in which cuts are to be found.
    - possible_edges (list): A list of edges to consider for forming pairs.

    Returns:
    - cut_edges (list of tuples): A list of tuples, where each tuple contains a pair of edges
      that, if removed, would disconnect the graph.
    """
    cut_edges = [(e1, e2) for e1, e2 in itertools.combinations(possible_edges, 2) if is_edge_cut((e1, e2), graph)]
    return cut_edges


# Check if a given set of edges form a cut-set in the graph
def is_edge_cut(edges, graph):
    """
    Check if a given set of edges form a cut-set in the graph.

    A cut-set is a set of edges whose removal makes the graph disconnected.
    This function creates a copy of the input graph, removes the specified edges, 
    and then checks if the graph remains connected.

    Parameters:
    - edges (list of tuples): A list of edges to be removed, where each edge is a tuple of two nodes.
    - graph (networkx.Graph): The input graph from which edges are removed.

    Returns:
    - bool: True if the removal of the edges disconnects the graph, False otherwise.
    """
    g = graph.copy()
    for u, v in edges:
        g.remove_edge(u, v)
    return not nx.is_connected(g)


# Find the root subgraph node (sn) in an SPQR tree that contains both the specified nodes
def find_root_sn(tree, s, t):
    """
    Find the root subgraph node (sn) in an SPQR tree that contains both the specified nodes.

    This function searches for a subgraph node in the SPQR tree that contains both nodes `s` and `t`.
    It returns the first such subgraph node that meets the condition.

    Parameters:
    - tree (list): A list of subgraph nodes, where each node is a tuple containing a type and a networkx graph.
    - s (int): The first node to search for.
    - t (int): The second node to search for.

    Returns:
    - tuple: The subgraph node from the tree that contains both `s` and `t`.
    """
    return [x for x in tree if ({s, t} & set(x[1].networkx_graph().nodes)) == {s, t}][0]


# Compute edge separators between subgraph nodes in an SPQR tree
def edge_seperators(tree):
    """
    Compute edge separators between subgraph nodes in an SPQR tree.

    An edge separator is defined as a pair of nodes shared by two neighboring subgraph nodes in the tree.
    This function creates a dictionary that maps each pair of neighboring subgraphs to the node pairs
    that separate them.

    Parameters:
    - tree (sage.graphs.spqr_tree): An SPQR tree where each edge connects two subgraph nodes.

    Returns:
    - sp_dict (dict): A dictionary where the keys are tuples representing neighboring subgraph nodes,
      and the values are tuples representing the edge separators between those subgraph nodes.
    """
    sp_dict = {}
    for (t1, g1), (t2, g2) in tree.networkx_graph().edges:
        sp = tuple(intersection(g1.networkx_graph().nodes, g2.networkx_graph().nodes))
        sp = min(sp), max(sp)
        sp_dict[((t1, g1), (t2, g2))] = sp
        sp_dict[((t2, g2), (t1, g1))] = sp
    return sp_dict


# Retrieve the nodes associated with a specific edge from the separator dictionary
def get_edge_nodes(e, sp_dict):
    """
    Retrieve the nodes associated with a specific edge from the separator dictionary.

    This function looks up the nodes associated with an edge from a dictionary of edge separators.
    If the edge is not found directly, it attempts to find it in the reversed order.

    Parameters:
    - e (tuple): A tuple representing the edge (subgraph1, subgraph2).
    - sp_dict (dict): A dictionary where keys are edge tuples and values are node pairs.

    Returns:
    - tuple: A tuple representing the nodes associated with the edge.
    """
    return sp_dict[e] if e in sp_dict else sp_dict[(e[1], e[0])]


# Handles traversal through the SPQR tree to identify nodes relevant to the heuristic.
# Dispatches the specific computation based on the type of SPQR tree node ('R', 'P', or 'S') by calling nodes_r, nodes_p, or nodes_s.
def spqr_nodes(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict):
    """
    Recursively traverses the SPQR tree to identify nodes relevant to the path between in_node and out_node.

    Parameters:
    - current_sn: The current SPQR tree node being processed.
    - parent_sn: The parent SPQR tree node from which the traversal originated.
    - tree: The SPQR tree of the graph.
    - g: The original graph.
    - in_node: The entry node into the current component.
    - out_node: The exit node from the current component.
    - sp_dict: A dictionary containing the separation pairs between components in the SPQR tree.

    Returns:
    - A list of nodes that are part of the identified path or structure.
    """
    if current_sn[0] == 'R':
        return nodes_r(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict)
    if current_sn[0] == 'P':
        return nodes_p(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict)
    if current_sn[0] == 'S':
        return nodes_s(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict)
    return []


# Handles 'S' (series) components by considering paths in a series and determining the longest sequences.
def nodes_s(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict):
    """
    Processes 'S' (series) components in the SPQR tree to identify relevant nodes.

    Parameters:
    - current_sn: The current series node in the SPQR tree.
    - parent_sn: The parent node in the SPQR tree from which this node was accessed.
    - tree: The SPQR tree of the graph.
    - g: The original graph.
    - in_node: The entry node into the component.
    - out_node: The exit node from the component.
    - sp_dict: A dictionary containing the separation pairs between components in the SPQR tree.

    Returns:
    - A list of nodes that form the longest sequence between in_node and out_node.
    """
    sn_sp = [(neighbor_sn, sp_dict[(neighbor_sn, current_sn)]) for neighbor_sn in tree.networkx_graph().neighbors(current_sn) if neighbor_sn != parent_sn]
    super_n_nodes = [((i, o), spqr_nodes(neighbor_sn, current_sn, tree, g, i, o, sp_dict)) for neighbor_sn, (i, o) in sn_sp]
    in_out_sn = [n for (i, o), n in super_n_nodes if (i, o) == (in_node, out_node)]
    ret = []
    if in_out_sn:
        # Print debug information for in-out sequence.
        print('iosn', in_node, out_node, in_out_sn)
        print(g.has_edge(in_node, out_node))
        other_path = flatten([n for (i, o), n in super_n_nodes if (i, o) != (in_node, out_node)]) + list(current_sn[1].networkx_graph().nodes)
        ret = max((in_out_sn[0] + [in_node, out_node], other_path), key=len)
    else:
        ret = flatten([n for (i, o), n in super_n_nodes]) + list(current_sn[1].networkx_graph().nodes)
    ret = ret if not parent_sn else diff(ret, [in_node, out_node])
    return ret


# Handles 'P' (parallel) components by identifying paths and taking the maximal node sets.
def nodes_p(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict):
    """
    Processes 'P' (parallel) components in the SPQR tree to identify relevant nodes.

    Parameters:
    - current_sn: The current parallel node in the SPQR tree.
    - parent_sn: The parent node in the SPQR tree from which this node was accessed.
    - tree: The SPQR tree of the graph.
    - g: The original graph.
    - in_node: The entry node into the component.
    - out_node: The exit node from the component.
    - sp_dict: A dictionary containing the separation pairs between components in the SPQR tree.

    Returns:
    - A list of nodes that form the largest path between in_node and out_node within the parallel structure.
    """
    sn_sp = [(neighbor_sn, sp_dict[(neighbor_sn, current_sn)]) for neighbor_sn in tree.networkx_graph().neighbors(current_sn) if neighbor_sn != parent_sn]
    super_n_nodes = [spqr_nodes(neighbor_sn, current_sn, tree, g, i, o, sp_dict) for neighbor_sn, (i, o) in sn_sp]
    ret = max(super_n_nodes, key=len)
    ret = ret + [in_node, out_node] if not parent_sn else ret
    return ret


# Handles 'R' (rigid) components, building subgraphs and computing relevant nodes by analyzing cut edges.
def nodes_r(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict):
    """
    Processes 'R' (rigid) components in the SPQR tree to identify relevant nodes.

    Parameters:
    - current_sn: The current rigid node in the SPQR tree.
    - parent_sn: The parent node in the SPQR tree from which this node was accessed.
    - tree: The SPQR tree of the graph.
    - g: The original graph.
    - in_node: The entry node into the component.
    - out_node: The exit node from the component.
    - sp_dict: A dictionary containing the separation pairs between components in the SPQR tree.

    Returns:
    - A list of nodes that form the longest path or set of nodes relevant to the heuristic within the rigid structure.
    """
    sn_sp = [(neighbor_sn, sp_dict[(neighbor_sn, current_sn)]) for neighbor_sn in tree.networkx_graph().neighbors(current_sn) if neighbor_sn != parent_sn]
    super_n_nodes_dict = [((i, o), spqr_nodes(neighbor_sn, current_sn, tree, g, i, o, sp_dict)) for neighbor_sn, (i, o) in sn_sp]
    super_n_nodes_dict = dict(super_n_nodes_dict)
    i_o_sn = super_n_nodes_dict.get((in_node, out_node), [])
    super_n_nodes_dict.pop((in_node, out_node), None)

    # Extract nodes from the current component.
    sp_nodes = list(current_sn[1].networkx_graph().nodes)
    local_g = g.subgraph(sp_nodes).copy()
    local_g.add_edges_from([e for e in super_n_nodes_dict.keys() if e not in local_g.edges])

    # Identify edges connected to in_node and out_node.
    in_edges = [e for e in super_n_nodes_dict.keys() if in_node in e]
    out_edges = [e for e in super_n_nodes_dict.keys() if out_node in e]

    # Remove the direct edge between in_node and out_node if it exists.
    if local_g.has_edge(in_node, out_node):
        local_g.remove_edge(in_node, out_node)

    # Identify relevant cut edges.
    cut_edges = get_relevant_cuts(local_g, super_n_nodes_dict.keys())

    # Extract nodes that can be easily determined from the cut edges.
    in_out_edges = in_edges + out_edges
    cut_maxes = [(max((e1, e2), key=lambda x: len(super_n_nodes_dict[x])), (e1, e2)) for e1, e2 in cut_edges]
    easy_cut_nodes = [(super_n_nodes_dict[e], (e1, e2)) for e, (e1, e2) in cut_maxes if e not in in_out_edges]
    easy_nodes = [n for n, (e1, e2) in easy_cut_nodes]
    easy_nodes += [super_n_nodes_dict[e] for e in diff(super_n_nodes_dict.keys(), flatten(cut_edges) + in_out_edges)]

    # Compute exclusion nodes and select the longest set of relevant nodes.
    relevant_cuts = diff(cut_edges, [t[1] for t in easy_cut_nodes])
    in_cuts = [(e1, e2) for e1, e2 in relevant_cuts if (in_node in e1 or in_node in e2) and not (in_node in e1 and in_node in e2)]
    out_cuts = [(e1, e2) for e1, e2 in relevant_cuts if (out_node in e1 or out_node in e2) and not (out_node in e1 and out_node in e2)]
    c = get_max_comb(in_cuts, in_edges, super_n_nodes_dict) + get_max_comb(out_cuts, out_edges, super_n_nodes_dict)

    ret = max((flatten(easy_nodes + [super_n_nodes_dict[e] for e in c]) + sp_nodes, i_o_sn + [in_node, out_node]), key=len)
    ret = ret if not parent_sn else diff(ret, [in_node, out_node])
    return ret



# Builds an exclusion graph using networkx.
# Creates a graph where edges represent exclusion pairs (2 nodes that cannot coexist on a path - due to the exclusion properties derived from SPQR tree analysis).
def get_max_comb(relevant_cuts, node_edges, super_n_nodes_dict):
    """
    Constructs an exclusion graph and finds the maximum independent set (MIS) by analyzing relevant cuts and node edges.

    Parameters:
    - relevant_cuts: A list of tuples representing edges that cannot coexist due to cut properties between nodes.
    - node_edges: A list of edges connected to the in_node or out_node, indicating potential exclusion relationships.
    - super_n_nodes_dict: A dictionary mapping edges to their associated node lists.

    Returns:
    - c: A list of edges that form the largest clique in the complement of the exclusion graph,
         which corresponds to the maximum independent set (MIS) in the original graph.
    """
    # Create a new graph to represent the exclusion relationships.
    exg = nx.Graph()

    # Combine node edges with flattened relevant cuts to create the initial set of edges for the exclusion graph.
    ex_edges = node_edges + flatten(relevant_cuts)
    exg.add_nodes_from(ex_edges)

    # Generate all pairs of node edges and add them as edges in the exclusion graph.
    # These edges represent pairs of nodes that cannot be on the same path.
    exg_pairs = list(itertools.combinations(node_edges, 2)) + relevant_cuts
    exg.add_edges_from(exg_pairs)

    # Construct the complement of the exclusion graph.
    # The complement graph contains edges between nodes that can coexist in a path.
    complement_exg = nx.complement(exg)

    # Assign weights to the nodes in the complement graph based on the length of their corresponding node lists.
    for node in complement_exg:
        complement_exg.nodes[node]['l'] = len(super_n_nodes_dict[node])

    # Find the maximum weight clique in the complement graph.
    # The clique corresponds to a set of nodes that can coexist, and thus to the largest independent set in the original exclusion graph.
    c, w = nx.max_weight_clique(complement_exg, weight='l')
    return c



# Finds the root component in the SPQR tree that contains both the source and target nodes
def find_root_sn(tree, s, t, sp_dict):
    """
    Finds the root component in the SPQR tree that contains both the source and target nodes.

    Parameters:
    - tree: The SPQR tree of a graph.
    - s: The source node in the graph.
    - t: The target node in the graph.
    - sp_dict: A dictionary containing the separation pairs between components in the SPQR tree.

    Returns:
    - The root component of the SPQR tree that contains both the source (s) and target (t) nodes.
      The function prefers a 'P' (parallel) node over other types when multiple components match.
    """
    return sorted(
        [x for x in tree if ({s, t} & set(x[1].networkx_graph().nodes)) == {s, t}],
        key=lambda x: 0 if x[0] == 'P' else 1
    )[0]


# compute ℎ_𝑀𝐼𝑆 (with an appropriate graph component and starting/ending nodes)
# Computes the nodes relevant for the longest path problem using the SPQR tree of the component
def get_max_nodes_spqr_recursive(component, in_node, out_node, return_nodes=False):
    """
    Computes the nodes relevant for the longest path problem using the SPQR tree of the component.

    Parameters:
    - component: The graph component being analyzed.
    - in_node: The entry node into the component.
    - out_node: The exit node from the component.
    - return_nodes: If True, returns the list of nodes in the path; otherwise, returns the count of nodes.

    Returns:
    - A list of nodes that form the relevant part of the path if return_nodes is True.
    - An integer representing the number of nodes in the path if return_nodes is False.
    """
    # Create a copy of the component to avoid modifying the original.
    comp = component.copy()
    
    # If the component has only two nodes, return them directly.
    if len(comp.nodes) == 2:
        return comp.nodes if return_nodes else len(comp.nodes)
    
    # Ensure the component includes an edge between in_node and out_node.
    if not comp.has_edge(in_node, out_node):
        comp.add_edge(in_node, out_node)
    
    # Create an SPQR tree for the component using SageMath.
    comp_sage = Graph(comp)
    try:
        tree = spqr_tree(comp_sage)
    except ValueError as e:
        if "graph is not biconnected" in str(e) or "cut vertex" in str(e):
            return len(comp.nodes)
        else:
            print(f"Tzur - An Error Occurred: {e}")
    
    # Generate the dictionary of separation pairs between components.
    sp_dict = edge_seperators(tree)
    
    # Identify the root node of the SPQR tree that contains both in_node and out_node.
    root_node = find_root_sn(tree, in_node, out_node, sp_dict)
    
    # Traverse the SPQR tree from the root to determine the nodes relevant for the path.
    nodes = spqr_nodes(root_node, [], tree, comp, min(in_node, out_node), max(in_node, out_node), sp_dict)
    
    return nodes if return_nodes else len(nodes)


# SNAKE IRRELEVANT?


def max_disj_constraints_actual_set(graph, s, t, y_filter=True, rectangle_filter=True, x_filter=False):
    g = graph.copy()
    g.remove_node(s)
    g.remove_node(t)
    if y_filter:
        while g.nodes:
            x = max(g.nodes, key=lambda x: g.degree[x])
            if g.degree[x] < 3:
                break
            y = []
            for n in [x] + list(random.sample(list(g.neighbors(x)), 3)):
                y += [n]
                g.remove_node(n)
    elif x_filter:
        while g.nodes:
            x = max(g.nodes, key=lambda x: g.degree[x])
            if g.degree[x] < 4:
                break
            for n in [x] + list(g.neighbors(x)):
                g.remove_node(n)
    if rectangle_filter:
        while g.nodes:
            cycles = nx.simple_cycles(g)
            if not cycles:
                break
            for c in cycles:
                if includes(g.nodes, c):
                    loser_node = random.choice(c)
                    g.remove_node(loser_node)
    return list(g.nodes)


def snake_spqr_nodes(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict):
    if current_sn[0] == 'R':
        return snake_nodes_r(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict)
    if current_sn[0] == 'P':
        return snake_nodes_p(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict)
    if current_sn[0] == 'S':
        return snake_nodes_s(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict)
    return []


def snake_nodes_s(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict):
    sn_sp = [(neighbor_sn, sp_dict[(neighbor_sn, current_sn)]) for neighbor_sn in tree.networkx_graph().neighbors(current_sn) if neighbor_sn != parent_sn]
    super_n_nodes = [((i ,o), snake_spqr_nodes(neighbor_sn, current_sn, tree, g, i, o, sp_dict)) for neighbor_sn, (i, o) in sn_sp]
    in_out_sn = [n for (i ,o), n in super_n_nodes if (i ,o) == (in_node, out_node)]
    ret = []
    if in_out_sn:
        print('iosn', in_node, out_node, in_out_sn)
        print(g.has_edge(in_node, out_node))
        # with open('D:/Heuristic Tests/improved_spqr_results/'+str(cur_t)+'weird_s.txt', "a+") as f:
        #     f.write(f'{pair_i} \nsource, target = {in_node, out_node} \nnodes = {list(g.nodes)}\nedges = {list(g.edges)} \n')
        #     f.write(f'itn = {str(index_to_node)}\n')
        #     f.write('\n\n')
        other_path = flatten([n for (i ,o), n in super_n_nodes if (i ,o) != (in_node, out_node)]) + list \
            (current_sn[1].networkx_graph().nodes)
        ret = max((in_out_sn[0] + [in_node, out_node], other_path), key=len)
    else:
        ret = flatten([n for (i ,o), n in super_n_nodes]) + list(current_sn[1].networkx_graph().nodes)
    ret = ret if not parent_sn else diff(ret, [in_node, out_node])
    #     print('s', ret)
    return ret


def snake_nodes_p(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict):
    sn_sp = [(neighbor_sn, sp_dict[(neighbor_sn, current_sn)]) for neighbor_sn in tree.networkx_graph().neighbors(current_sn) if neighbor_sn != parent_sn]
    super_n_nodes = [snake_spqr_nodes(neighbor_sn, current_sn, tree, g, i, o, sp_dict) for neighbor_sn, (i, o) in sn_sp]
    ret = max(super_n_nodes, key=len)
    ret = ret + [in_node, out_node] if not parent_sn else ret
    #     print('p', ret)
    return ret


def snake_nodes_r(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict):
    #     print('hi')
    sn_sp = [(neighbor_sn, sp_dict[(neighbor_sn, current_sn)]) for neighbor_sn in tree.networkx_graph().neighbors(current_sn) if neighbor_sn != parent_sn]
    super_n_nodes_dict = [((i ,o) , snake_spqr_nodes(neighbor_sn, current_sn, tree, g, i, o, sp_dict)) for neighbor_sn, (i, o) in sn_sp]
    super_n_nodes_dict = dict(super_n_nodes_dict)
    i_o_sn = super_n_nodes_dict[(in_node, out_node)] if (in_node, out_node) in super_n_nodes_dict.keys() else []
    super_n_nodes_dict.pop((in_node, out_node), None)

    #     with open('D:/Heuristic Tests/improved_spqr_results/'+str(cur_t)+'r_size.txt', "a+") as f:
    #         f.write(f'\tr -- {len(super_n_nodes_dict)}\n')

    # with open('D:/Heuristic Tests/improved_spqr_results/'+str(cur_t)+'r_neighbors_count.txt', "a+") as f:
    #         f.write(f'{len(super_n_nodes_dict)}\n')
    sp_nodes = list(current_sn[1].networkx_graph().nodes)
    #     print(sp_nodes)
    local_g = g.subgraph(sp_nodes).copy()
    local_g_og = local_g.copy()

    local_g.add_edges_from([e for e in super_n_nodes_dict.keys() if e not in local_g.edges])

    # in_node edges
    in_edges = [e for e in super_n_nodes_dict.keys() if in_node in e]

    # out_node edges
    out_edges = [e for e in super_n_nodes_dict.keys() if out_node in e]

    if local_g.has_edge(in_node, out_node):
        local_g.remove_edge(in_node, out_node)

    # cut edges
    cut_edges = get_relevant_cuts(local_g, super_n_nodes_dict.keys())

    # easy nodes
    in_out_edges = in_edges + out_edges
    cut_maxes = [(max((e1, e2), key=lambda x: len(super_n_nodes_dict[x])), (e1 ,e2)) for e1 ,e2 in cut_edges]
    easy_cut_nodes = [(super_n_nodes_dict[e], (e1 ,e2)) for e ,(e1 ,e2) in cut_maxes if e not in in_out_edges]
    easy_nodes = [n for n, (e1, e2) in easy_cut_nodes]
    easy_nodes += [super_n_nodes_dict[e] for e in diff(super_n_nodes_dict.keys(), flatten(cut_edges) + in_out_edges)]

    # s_t_cuts = [(e1, e2) for e1, e2 in cut_edges if (e1 in in_edges and e2 in out_edges) or (e2 in in_edges and e1 in out_edges)]
    # if s_t_cuts:
    #     print(s_t_cuts)
    #     with open('D:/Heuristic Tests/improved_spqr_results/ ' +str(cur_t ) +'s_t_cut.txt', "a+") as f:
    #         f.write \
    #             (f'{s_t_cuts} \nsource, target = {in_node, out_node} \nnodes = {list(g.nodes)}\nedges = {list(g.edges)} \n')
    #         f.write(f'itn = {str(index_to_node)}\n')
    #         f.write('\n\n')

    # draw_grid('', 'rc', g, [[0]*20]*20, in_node, out_node, itn, path=local_g.nodes)
    # print('in , out - ', in_node, out_node)
    # print('edges -', list(local_g.edges))
    # print('cuts -', get_relevant_cuts(local_g, local_g.edges))
    # print('ve')
    # for k,v in super_n_nodes_dict.items():
    #     print(k, '\t', v)

    # exclusion nodes
    relevant_cuts = diff(cut_edges, [t[1] for t in easy_cut_nodes])

    in_cuts = [(e1, e2) for e1, e2 in relevant_cuts if (in_node in e1 or in_node in e2) and not (in_node in e1 and in_node in e2)]

    out_cuts = [(e1, e2) for e1, e2 in relevant_cuts if (out_node in e1 or out_node in e2) and not (out_node in e1 and out_node in e2)]

    c = get_max_comb(in_cuts, in_edges, super_n_nodes_dict) + get_max_comb(out_cuts, out_edges, super_n_nodes_dict)

    sp_nodes_filtered = max_disj_constraints_actual_set(local_g_og, in_node, out_node, y_filter=True, rectangle_filter=True, x_filter=False)

    #     print(len(complement_exg.nodes))
    ret = max((flatten(easy_nodes + [super_n_nodes_dict[e] for e in c]) + sp_nodes, i_o_sn + [in_node, out_node]), key=len)
    ret = ret if not parent_sn else diff(ret, [in_node, out_node])
    #     print('r', ret)
    return ret


def snake_get_max_nodes_spqr_recursive(component, in_node, out_node, return_nodes=False):
    # print(f"s:{s}, t:{t}")
    #     with open('D:/Heuristic Tests/improved_spqr_results/'+str(cur_t)+'r_size.txt', "a+") as f:
    #         f.write(f'\ncomp len -- {len(component)}\n')
    comp = component.copy()
    if len(comp.nodes) == 2:
        return comp.nodes if return_nodes else len(comp.nodes)
    if not comp.has_edge(in_node, out_node):
        #         print(f'adding {(in_node, out_node)}')
        comp.add_edge(in_node, out_node)
    comp_sage = Graph(comp)
    tree = spqr_tree(comp_sage)
    # for x in tree:
    #     print(x[0], list(x[1].networkx_graph().nodes))
    sp_dict = edge_seperators(tree)
    root_node = find_root_sn(tree, in_node, out_node, sp_dict)
    #     print(root_node)
    nodes = snake_spqr_nodes(root_node, [], tree, comp, min(in_node, out_node), max(in_node, out_node), sp_dict)
    # print('ret', res)
    return nodes if return_nodes else len(nodes)


# SNAKE


def get_clique(node, graph):
    g = graph.subgraph(list(graph.neighbors(node)))
    if not g.nodes:
        return [node]
    else:
        x = max(g.nodes, key=lambda x: g.degree[x])
        return [node] + get_clique(x, g)


def max_disj_set_upper_bound(nodes, pairs, x_filter=False, y_filter=False, rectangle_filter=False, og_g=None):
    g = nx.Graph()
    for x in nodes:
        g.add_node(x)
    for s, t in pairs:
        g.add_edge(s, t)
    degrees = g.degree
    counter = 0
    # print("nodes", len(list(g.nodes)))
    # print("pairs", pairs)
    while g.nodes:
        x = max(g.nodes, key=lambda x: degrees[x])
        c = get_clique(x, g)
        # print(len(c))
        if len(c) == 1:
            break
        for n in c:
            g.remove_node(n)
        counter += 1
    if og_g:
        og_g = og_g.subgraph(g.nodes).copy()
        if y_filter:
            while og_g.nodes:
                x = max(og_g.nodes, key=lambda x: og_g.degree[x])
                if og_g.degree[x] < 3:
                    break
                # print('---------')
                y = []
                for n in [x] + list(random.sample(list(og_g.neighbors(x)), 3)):
                    y += [n]
                    g.remove_node(n)
                    og_g.remove_node(n)
                # print([index_to_node[f] for f in y])
                counter += 3
        elif x_filter:
            while og_g.nodes:
                x = max(og_g.nodes, key=lambda x: og_g.degree[x])
                if og_g.degree[x] < 4:
                    break
                for n in [x] + list(og_g.neighbors(x)):
                    g.remove_node(n)
                    og_g.remove_node(n)
                counter += 4
        if rectangle_filter:
            while og_g.nodes:
                x = max(og_g.nodes, key=lambda x: og_g.degree[x])
                if og_g.degree[x] < 2:
                    break
                for n in [x] + list(og_g.neighbors(x)):
                    g.remove_node(n)
                    og_g.remove_node(n)
                counter += 4

    counter += len(g.nodes)
    # print('counter', counter)
    # print('----------------------------------')
    return counter


def get_max_nodes_spqr_snake(component, in_node, out_node, x_filter=False, y_filter=False, in_neighbors=False, out_neighbors=False, return_pairs=False):
    # print(f"s:{s}, t:{t}")
    component = component.copy()
    # if s and t are connected, the snake must go directly to t
    if component.has_edge(in_node, out_node):
        return 2
    else:
        component.add_edge(in_node, out_node)
    comp_sage = Graph(component)
    tree = spqr_tree(comp_sage)
#     for x in tree:
#         print(x)
#     print('--------------------------------')
    pairs = get_all_spqr_pairs_new(tree, component, in_node, out_node)
    # if in_neighbors:
    #     pairs.update(get_neighbors_pairs(component, in_node))
    # if out_neighbors:
    #     pairs.update(get_neighbors_pairs(component, out_node))
    res = pairs if return_pairs else max_disj_set_upper_bound(component.nodes, pairs, x_filter, y_filter, component)
    # print('ret', res)
    return res


def nodes_of_sn(current_sn, parent_sn, tree):
    nodes = list(current_sn[1].networkx_graph().nodes)
    for neighbor in tree.neighbors(current_sn):
        if neighbor == parent_sn:
            continue
        nodes += nodes_of_sn(neighbor, current_sn, tree)
    return nodes


def get_all_spqr_pairs_new(tree, component, in_node, out_node):
    pairs = []
    for c in tree:
        pairs += pairs_spqr_new(c, tree, component, in_node, out_node)
    pairs = set(pairs)
    return pairs


def pairs_spqr_new(current_sn, tree, g, s, t):
    if current_sn[0] == 'R':
        pairs = pairs_r(current_sn, tree, g, s, t)
        return pairs
    if current_sn[0] == 'P':
        return pairs_p(current_sn, tree, g, s, t)
    return []


def pairs_p(current_sn, tree, g, s, t):
    sub_nodes = [diff(nodes_of_sn(neighbor_sn, current_sn, tree), current_sn[1].networkx_graph().nodes) for neighbor_sn in tree.networkx_graph().neighbors(current_sn)]
    sub_nodes = [nodes for nodes in sub_nodes if ((s not in nodes or s in current_sn[1].networkx_graph().nodes) and (t not in nodes or t in current_sn[1].networkx_graph().nodes))]
    pairs = []
    for i in range(len(sub_nodes)):
        for j in range(i+1, len(sub_nodes)):
            pairs += flatten([[(n1, n2) if n1 < n2 else (n2, n1) for n1 in sub_nodes[i]] for n2 in sub_nodes[j]])
    return pairs


def get_all_r_pairs(comp):
    # print('start')
    comp_g = Graph(comp)
    t = spqr_tree(comp_g)
    ps = [all_pairs(list(g.networkx_graph().nodes)) for t, g in t if t == 'R']
    res = set()
    for s in ps:
        res = res.union(s)
    # print('end')
    return res


def pairs_r(current_sn, tree, g, s, t):
    # print(f's,t - {(s,t)}')
    super_n_nodes = [nodes_of_sn(neighbor_sn, current_sn, tree) for neighbor_sn in
                     tree.networkx_graph().neighbors(current_sn)]
    super_edges = [tuple(set(intersection(nodes, current_sn[1].networkx_graph().nodes))) for nodes in super_n_nodes]
    super_edges = dict([(e, diff(nodes, e)) for e, nodes in zip(super_edges, super_n_nodes)])
    super_edges.pop((s, t), None)
    super_edges.pop((t, s), None)
    # print(super_edges)

    sp_nodes = list(current_sn[1].networkx_graph().nodes)
    local_g = g.subgraph(sp_nodes).copy()

    local_g.add_edges_from([e for e in super_edges.keys() if e not in local_g.edges])

    try:
        # s_,t_ = [e for e in local_g.edges if (((s in e) or (e in super_edges.keys() and s in super_edges[e])) and ((t in e) or (e in super_edges.keys() and t in super_edges[e])))][0]
        potential_es = [e for e, ns in super_edges.items() if ((s in e or s in ns) and (t in e or t in ns))] + [e for e
                                                                                                                in
                                                                                                                local_g.edges
                                                                                                                if (
                                                                                                                            s in e and t in e)]
        s_, t_ = potential_es[0]
    except Exception as e:
        print(f's,t - {(s, t)}')
        print(super_edges)
        print(potential_es)
        raise e

    if s not in (s_, t_) or t not in (s_, t_):
        super_edges.pop((s_, t_), None)
        s, t = s_, t_

    # print(f'{(s,t)}, {list(local_g.edges)}')

    # out of s pairs:
    s_edges_nodes = [super_edges[e] for e in super_edges.keys() if s in e]
    s_edge_pairs = []
    for i in range(len(s_edges_nodes)):
        for j in range(i + 1, len(s_edges_nodes)):
            s_edge_pairs += flatten(
                [[(n1, n2) if n1 < n2 else (n2, n1) for n1 in s_edges_nodes[i]] for n2 in s_edges_nodes[j]])

    # out of t pairs:
    t_edges_nodes = [super_edges[e] for e in super_edges.keys() if t in e]
    #     print(t_edges_nodes)
    t_edge_pairs = []
    for i in range(len(t_edges_nodes)):
        for j in range(i + 1, len(t_edges_nodes)):
            t_edge_pairs += flatten(
                [[(n1, n2) if n1 < n2 else (n2, n1) for n1 in t_edges_nodes[i]] for n2 in t_edges_nodes[j]])

    # print(111)
    # edge cut pairs:
    #     print(s,t)
    #     print(super_edges.keys())
    if (s, t) not in super_edges.keys() and (t, s) not in super_edges.keys():
        local_g.remove_edge(s, t)
    cut_pairs = []
    #     print(get_relevant_cuts(local_g, super_edges.keys()))
    for p1, p2 in get_relevant_cuts(local_g, super_edges.keys()):
        #         print(f'1 - {super_edges[p1]}    2 - {super_edges[p2]}')
        cut_pairs += flatten([[(n1, n2) if n1 < n2 else (n2, n1) for n1 in super_edges[p1]] for n2 in super_edges[p2]])
    #     # print(222)
    # draw_grid('', 'r', g, [[0]*20]*20, s, t, itn, path=local_g.nodes)
    # print('in , out - ', s, t)
    # print('edges -', list(local_g.edges))
    # print('cuts -', get_relevant_cuts(local_g, local_g.edges))
    # print('ve')
    # for k,v in super_edges.items():
    #     print(k, '\t', v)
    return list(set(s_edge_pairs + t_edge_pairs + cut_pairs))


def get_neighbors_pairs(component, node):
    node_neighbors = list(component.neighbors(node))
#     print(in_node_neighbors)
    pairs = set(combinations(node_neighbors, 2))
#     print(pairs)
    return pairs


def get_max_nodes_spqr_new(component, in_node, out_node, x_filter=False, y_filter=False, in_neighbors=False, out_neighbors=False, return_pairs=False):
    # print(f"s:{s}, t:{t}")
    component = component.copy()
    if not component.has_edge(in_node, out_node):
#         print(11)
        component.add_edge(in_node, out_node)
    comp_sage = Graph(component)
    tree = spqr_tree(comp_sage)
#     for x in tree:
#         print(x)
#     print('--------------------------------')
    pairs = get_all_spqr_pairs_new(tree, component, in_node, out_node)
    # COMMON.pairs_idk = pairs
    if in_neighbors:
        pairs.update(get_neighbors_pairs(component, in_node))
    if out_neighbors:
        pairs.update(get_neighbors_pairs(component, out_node))
    res = pairs if return_pairs else max_disj_set_upper_bound(component.nodes, pairs, x_filter, y_filter, component)
    # print('ret', res)
    return res