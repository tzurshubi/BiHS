import heapq,time
from heuristics.heuristic import heuristic
from models.state import State
from models.heapq_state import HeapqState
from utils.utils import *


def uniHS_for_LSP(graph, start, goal, heuristic_name, snake, args):
    # Initialize custom priority queue
    open_set = HeapqState()

    # Initial state
    initial_state = State(graph, [start], snake) if isinstance(start, int) else State(graph, start, snake)

    # Initial f_value
    initial_f_value = heuristic(initial_state, goal, heuristic_name, snake)

    # Push initial state with priority based on f_value
    open_set.push(initial_state, initial_f_value)

    # The best path found
    best_path = None
    best_path_length = -1

    # Expansion counter
    expansions = 0

    while len(open_set) > 0:
        # Pop the state with the highest priority (g(N) + h(N))
        _, _, current_state, f_value = open_set.pop()
        current_path_length = len(current_state.path) - 1

        # Increment the expansion counter
        expansions += 1
        # if expansions % 10000 == 0:
        #     print(
        #         f"Expansion #{expansions}: state {current_state.path}, f={f_value}, len={len(current_state.path)}"
        #     )

        # Check if the current state is the goal state
        if current_state.head == goal:
            # Update the best path if the current one is longer
            if current_path_length > best_path_length:
                best_path = current_state.path
                best_path_length = current_path_length
                print(f"[{time2str(args.start_time,time.time())} expansion {expansions}] Found path of length {best_path_length}. {best_path}")
                with open(f"gf2_symbr_bctis_{args.date}_{args.graph_type}_{args.number_of_graphs}_uni.txt", 'a') as file:
                    file.write(f"[{time2str(args.start_time,time.time())} expansion {expansions}] Found path of length {best_path_length}. {best_path}\n")
    
            continue

        # Finish if the f_value is smaller than the best path length found so far
        if f_value <= best_path_length:
            # print(f"apparently we won't find a path longer than {best_path_length}")
            break

        # Generate successors
        successors = current_state.successor(snake, True)
        for successor in successors:
            # Calculate the heuristic value
            h_value = heuristic(successor, goal, heuristic_name, snake)
            # Calculate the g_value
            g_value = current_path_length + 1
            # Calculate the f_value
            f_value = g_value + h_value
            # Push the successor to the priority queue with the priority as - (g(N) + h(N))
            open_set.push(successor, f_value)

    return best_path, expansions
