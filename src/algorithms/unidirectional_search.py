import heapq,time
from heuristics.heuristic import heuristic
from models.state import State
from models.heapq_state import HeapqState
from utils.utils import *


def unidirectional_search(graph, start, goal, heuristic_name, snake, args):
    # Initialize custom priority queue
    open_set = HeapqState()

    # Initial state
    initial_state = State(graph, [start], snake) if isinstance(start, int) else State(graph, start, snake)

    # Initial f_value
    initial_h_value = heuristic(initial_state, goal, heuristic_name, snake)
    initial_f_value = initial_state.g + initial_h_value

    # Push initial state with priority based on f_value
    open_set.push(initial_state, initial_f_value)

    # The best path found
    best_path = None
    best_path_length = -1

    # Expansion counter, generated counter
    expansions = 0
    generated = 0

    while len(open_set) > 0:
        # Pop the state with the highest priority (g(N) + h(N))
        _, _, current_state, f_value = open_set.pop()
        current_path_length = len(current_state.path) - 1

        # Increment the expansion counter
        expansions += 1
        # if expansions % 5000 == 0:
        #     print(
        #         f"Expansion #{expansions}: state {current_state.path}, f={f_value}, len={len(current_state.path)}"
        #     )

        # Check if the current state is the goal state
        if current_state.head == goal:
            if current_path_length > best_path_length:
                best_path = current_state.path
                best_path_length = current_path_length
                if snake:
                    print(f"[{time2str(args.start_time,time.time())} expansion {expansions}, {time_ms(args.start_time,time.time())}] Found path of length {best_path_length}. {best_path}. generated: {generated}")
                    with open(args.log_file_name, 'a') as file:
                        file.write(f"[{time2str(args.start_time,time.time())} expansion {expansions}] Found path of length {best_path_length}. {best_path}\n")
    
            continue

        # Finish if the f_value is smaller than the best path length found so far
        if f_value <= best_path_length:
            # print(f"apparently we won't find a path longer than {best_path_length}")
            break

        # Generate successors
        successors = current_state.successor(args, snake, True)
        for successor in successors:
            generated += 1
            # Check if successor reached the goal
            if successor.head == goal:
                h_successor = 0
                if current_path_length > best_path_length:
                    best_path = successor.path
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

    return best_path, expansions, generated
