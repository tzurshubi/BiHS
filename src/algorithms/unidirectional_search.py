import heapq, time
from heuristics.heuristic import heuristic
from models.state import State
from models.heapq_state import HeapqState
from utils.utils import *
import matplotlib.pyplot as plt
from collections import defaultdict

def unidirectional_search(graph, start, goal, heuristic_name, snake, args):
    stats = {"expansions": 0, "generated": 0, "symmetric_states_removed": 0, "dominated_states_removed": 0}
    logger = args.logger
    cube = args.graph_type == "cube"
    buffer_dim = args.cube_buffer_dim if cube else None

    # For Plotting
    g_degree_pairs = []

    open_set = HeapqState()
    initial_state = State(graph, [start], [], snake, args) if isinstance(start, int) else State(graph, start, [], snake, args)

    initial_h_value = heuristic(initial_state, goal, heuristic_name, snake, args, graph)
    initial_f_value = initial_state.g + initial_h_value
    open_set.push(initial_state, initial_f_value, initial_f_value)

    if args.bsd:
        state_key = (initial_state.head, initial_state.path_vertices_and_neighbors if snake else initial_state.path_vertices)
        FNV = {state_key: initial_state.g}

    best_path = None
    best_path_length = -1

    while len(open_set) > 0:
        priority, f_value, g_value, current_state = open_set.pop()

        stats["expansions"] += 1
        # if stats["expansions"] % 10_000 == 0:
        #     logger(f"Expansion {stats['expansions']}: state {current_state.path}, f={f_value}, g={g_value}")
            
        if current_state.head == goal:
            if g_value > best_path_length:
                best_path = current_state
                best_path_length = g_value
            continue

        if f_value <= best_path_length:
            break

        successors = current_state.generate_successors(args, snake, True)
        g_degree_pairs.append((current_state.g, len(successors)))

        for successor in successors:
            if args.bsd:
                state_key = (successor.head, successor.path_vertices_and_neighbors if snake else successor.path_vertices)
                if state_key in FNV and FNV[state_key] >= successor.g:
                    stats["symmetric_states_removed"] += 1
                    continue
                # Record the new best arrival length
                FNV[state_key] = successor.g

            if buffer_dim is not None and has_bridge_edge_across_dim(current_state, successor, buffer_dim):
                if successor.traversed_buffer_dimension: continue
                successor.traversed_buffer_dimension = True

            stats["generated"] += 1
            
            if successor.head == goal:
                h_successor = 0
                if successor.g > best_path_length:
                    best_path = successor
                    best_path_length = successor.g
                    if f_value <= best_path_length:
                        break
            else:
                h_successor = heuristic(successor, goal, heuristic_name, snake, args, graph)
                
            g_successor = successor.g
            f_successor = g_successor + h_successor
            
            # min(f_successor, f_value) enforces pathmax/monotonicity 
            open_set.push(successor, min(f_successor, f_value), min(f_successor, f_value))

    return best_path.materialize_path() if best_path else None, stats