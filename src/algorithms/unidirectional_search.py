import heapq
from heuristics.heuristic import heuristic

from models.state import State
from models.heapq_state import HeapqState


def uniHS_for_LSP(graph, start, goal, heuristic_name):
    # Initialize custom priority queue
    open_set = HeapqState()

    # Initial state
    initial_state = State(graph, [start])

    # Initial f_value
    initial_f_value = heuristic(initial_state, goal, heuristic_name)

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
        print(f"Expanding state {current_state.path}")

        # Check if the current state is the goal state
        if current_state.head() == goal:
            # Update the best path if the current one is longer
            if current_path_length > best_path_length:
                best_path = current_state.path
                best_path_length = current_path_length
                print(
                    f"This state has head as the goal, and is the longest found, with length {current_path_length}"
                )
            continue

        # Finish if the f_value is smaller than the best path length found so far
        if f_value <= best_path_length:
            print(f"apparently we won't find a path longer than {best_path_length}")
            break

        # Generate successors
        successors = current_state.successor()
        for successor in successors:
            # Calculate the heuristic value
            h_value = heuristic(successor, goal, heuristic_name)
            # Calculate the g_value
            g_value = current_path_length + 1
            # Calculate the f_value
            f_value = g_value + h_value
            # Push the successor to the priority queue with the priority as - (g(N) + h(N))
            open_set.push(successor, f_value)

    return best_path, expansions
