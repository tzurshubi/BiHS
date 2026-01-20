import heapq,time
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
    g_degree_pairs = []  # Store (g, degree) for each expanded state

    # Initialize custom priority queue
    open_set = HeapqState()

    # Initial state
    initial_state = State(graph, [start], [], snake) if isinstance(start, int) else State(graph, start, [], snake)

    # Initial f_value
    initial_h_value = heuristic(initial_state, goal, heuristic_name, snake)
    initial_f_value = initial_state.g + initial_h_value

    # Push initial state with priority based on f_value
    open_set.push(initial_state, initial_f_value)

    # Basic symmetry detection - a dictionary with the key (head,nodes)
    FNV = {(initial_state.head, initial_state.path_vertices_and_neighbors_bitmap if snake else initial_state.path_vertices_bitmap)}

    # The best path found
    best_path = None
    best_path_length = -1

    # Expansion counter, generated counter
    stats["expansions"] = 0
    stats["generated"] = 0

    while len(open_set) > 0:
        # Pop the state with the highest priority (g(N) + h(N))
        f_value, g_value, current_state = open_set.pop()
        current_path_length = g_value

        # Increment the expansion counter
        stats["expansions"] += 1
        if stats["expansions"] % 10_000 == 0:
            logger(f"Expansion {stats['expansions']}: state {current_state.path}, f={f_value}, g={g_value}")
            
        # Check if the current state is the goal state
        if current_state.head == goal:
            if current_path_length > best_path_length:
                best_path = current_state
                best_path_length = current_path_length
                if snake:
                    logger(f"Expansion {stats['expansions']}: Found path of length {best_path_length}: {best_path.materialize_path()}. f_max={f_value}, generated={stats['generated']}")
            continue

        # Finish if the f_value is smaller than the best path length found so far
        if f_value <= best_path_length:
            # print(f"apparently we won't find a path longer than {best_path_length}")
            break

        # Generate successors
        successors = current_state.generate_successors(args, snake, True)

        # For Plotting
        g_degree_pairs.append((current_state.g, len(successors)))

        for successor in successors:
            if args.bsd and (successor.head, successor.path_vertices_and_neighbors_bitmap if snake else successor.path_vertices_bitmap) in FNV:
                # print(f"symmetric state removed: {successor.path}")
                stats["symmetric_states_removed"] += 1
                continue

            # Check if successor traverses the buffer dimension in cube graphs
            if has_bridge_edge_across_dim(current_state, successor, buffer_dim):
                if successor.traversed_buffer_dimension: continue  # already traversed buffer dimension
                successor.traversed_buffer_dimension = True

            stats["generated"] += 1
            
            # Check if successor reached the goal
            if successor.head == goal:
                h_successor = 0
                if current_path_length > best_path_length:
                    best_path = successor
                    best_path_length = current_path_length
                    if f_value <= best_path_length:
                        break
            else:
                # Calculate the heuristic value
                h_successor = heuristic(successor, goal, heuristic_name, snake)
            # Calculate the g_successor
            g_successor = successor.g
            # Calculate the f_value
            f_successor = g_successor + h_successor
            # Push the successor to the priority queue with the priority as - (g(N) + h(N))
            open_set.push(successor, min(f_successor, f_value))
            FNV.add((successor.head, successor.path_vertices_and_neighbors_bitmap if snake else successor.path_vertices_bitmap))

    return best_path.path, stats
