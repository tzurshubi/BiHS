import heapq, time
from heuristics.heuristic import heuristic
from models.state import State
from models.heapq_state import HeapqState
from utils.utils import *
import matplotlib.pyplot as plt
from collections import defaultdict

def XA(graph, start, goal, heuristic_name, snake, args):
    stats = args.stats
    logger = args.logger
    cube = args.graph_type == "cube"
    buffer_dim = args.cube_buffer_dim if cube else None
    V = len(graph.nodes)
    lookahead = args.lookahead if args.lookahead >= 1 else 1

    # For Plotting
    g_degree_pairs = []

    open_set = HeapqState()
    initial_state = State(graph, [start], [], snake, args) if isinstance(start, int) else State(graph, start, [], snake, args)

    initial_h_value = heuristic(initial_state, goal, heuristic_name, snake, args) if heuristic_name else V
    initial_f_value = initial_state.g + initial_h_value
    open_set.push(initial_state, initial_f_value, initial_f_value)

    if args.bsd:
        state_key = (initial_state.head, initial_state.path_vertices_and_neighbors if snake else initial_state.path_vertices)
        FNV = {state_key: initial_state.g}

    best_path = None
    best_path_length = -1

    def get_lookahead_successors(cur_state, cur_h_graph, remaining):
        """
        Recursively advances the frontier down to depth `lookahead` from the
        state currently being expanded, mirroring the scheme used by
        XDFBnB/XIDA: the heuristic is only evaluated at the leaves of this
        local depth-k subtree, not at every intermediate node. A branch that
        reaches the goal ends immediately (going past the goal makes no sense
        for a longest simple s-t path); a branch that dead-ends before depth k
        is dropped. `cur_h_graph` is threaded/shrunk alongside the recursion
        (each step removes the head that was just left behind) so the
        heuristic never has to recompute the tail-removed graph from scratch.
        """
        nonlocal best_path, best_path_length

        if remaining == 0:
            h_val = V
            if heuristic_name:
                h_val = heuristic(cur_state, goal, heuristic_name, snake, args, cur_h_graph.copy() if snake else cur_h_graph)
            return [(h_val, cur_state, cur_h_graph)]

        succs = cur_state.generate_successors(args, snake, True)
        stats["generated"] += len(succs)
        if remaining == lookahead:
            g_degree_pairs.append((cur_state.g, len(succs)))

        # Graph copy hoisted outside the loop for performance
        next_h_graph = cur_h_graph.copy()
        if cur_state.head in next_h_graph: next_h_graph.remove_node(cur_state.head)

        all_leaves = []
        for succ in succs:
            if buffer_dim is not None and has_bridge_edge_across_dim(cur_state, succ, buffer_dim):
                if succ.traversed_buffer_dimension: continue
                succ.traversed_buffer_dimension = True

            if succ.head == goal:
                # Reaching the goal ends this branch immediately (it's the only
                # valid endpoint) - matches XA's existing check-on-generation.
                if succ.g > best_path_length:
                    best_path = succ
                    best_path_length = succ.g
                    if cube: logger(f"Expansion {stats['expansions']}: New longest path found with length {best_path_length}: {best_path.materialize_path()}")
                all_leaves.append((0, succ, next_h_graph))
                continue

            if remaining == 1:
                # Fast-path for the final lookahead layer
                h_val = V
                if heuristic_name:
                    h_val = heuristic(succ, goal, heuristic_name, snake, args, next_h_graph.copy() if snake else next_h_graph)
                all_leaves.append((h_val, succ, next_h_graph))
            else:
                all_leaves.extend(get_lookahead_successors(succ, next_h_graph, remaining - 1))

        return all_leaves

    while len(open_set) > 0:
        priority, f_value, g_value, current_state = open_set.pop()

        stats["expansions"] += 1
        if stats["expansions"] % 10_000 == 0:
            logger(f"Expansion {stats['expansions']}: state {current_state.path}, f={f_value}, g={g_value}")

        if current_state.head == goal:
            if g_value > best_path_length:
                best_path = current_state
                best_path_length = g_value
                if cube: logger(f"Expansion {stats['expansions']}: New longest path found with length {best_path_length}: {best_path.materialize_path()}")
            continue

        if f_value <= best_path_length:
            break

        # A*'s open list is a heap, not a call stack, so unlike XDFBnB's
        # h_graph (threaded continuously across the whole recursive descent)
        # this has to be rebuilt fresh for whichever state was just popped -
        # tail nodes removed once here, then threaded/shrunk through the
        # lookahead recursion below.
        h_graph = graph.copy()
        h_graph.remove_nodes_from(current_state.tail())

        leaves = get_lookahead_successors(current_state, h_graph, lookahead)

        gui_successors = [] if args.gui_logger.video else None

        for h_val, leaf, leaf_h_graph in leaves:
            if leaf.head == goal:
                # best_path was already updated when this leaf was generated
                # (above) - there's nothing left to search past the goal.
                continue

            if args.bsd:
                state_key = (leaf.head, leaf.path_vertices_and_neighbors if snake else leaf.path_vertices)
                if state_key in FNV and FNV[state_key] >= leaf.g:
                    stats["symmetric_states_removed"] += 1
                    continue
                # Record the new best arrival length
                FNV[state_key] = leaf.g

            f_leaf = leaf.g + h_val
            if f_leaf <= best_path_length:
                # Already dominated by the current incumbent - not worth
                # keeping in the open list.
                continue

            if gui_successors is not None:
                gui_successors.append((leaf, f_leaf, h_val))

            # min(f_leaf, f_value) enforces pathmax/monotonicity
            open_set.push(leaf, min(f_leaf, f_value), min(f_leaf, f_value))

        if gui_successors is not None:
            args.gui_logger.gui_log_expansion(current_state, f_value, f_value - g_value, gui_successors)

    return best_path.materialize_path() if best_path else None, stats
