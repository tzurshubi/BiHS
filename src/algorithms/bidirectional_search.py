import matplotlib.pyplot as plt
import heapq
from heuristics.heuristic import heuristic
from models.state import State
from models.heapq_state import HeapqState


def biHS_for_LSP(graph, start, goal, heuristic_name, snake = False):
    # # For Plotting h
    # mis_smaller_flag = []
    # expansions_list = []
    # h_MIS = []
    # h_BCC = []
    # max_f = []

    # Options
    alternate = False
    lastDirectionF = False

    # Initialize meeting point of the two searches
    best_path_meet_point = None

    # Initialize custom priority queues for forward and backward searches
    open_set_F = HeapqState()
    open_set_B = HeapqState()

    # Initial states
    initial_state_F = State(graph, [start], snake)
    initial_state_B = State(graph, [goal], snake)

    # Initial f_values
    initial_f_value_F = heuristic(initial_state_F, goal, heuristic_name, snake)
    initial_f_value_B = heuristic(initial_state_B, start, heuristic_name, snake)

    # Push initial states with priority based on f_value
    open_set_F.push(initial_state_F, initial_f_value_F)
    open_set_B.push(initial_state_B, initial_f_value_B)

    # Best path found and its length
    best_path = None        # S in the pseudocode
    best_path_length = -1   # U in the pseudocode

    # Expansion counter
    expansions = 0

    # Closed sets for forward and backward searches
    closed_set_F = set()
    closed_set_B = set()

    while len(open_set_F) > 0 or len(open_set_B) > 0:
        # Determine which direction to expand
        directionF = None # True - Forward, False - Backward 
        if alternate:
            if lastDirectionF == True:
                directionF = False
            else:
                directionF = True
            lastDirectionF = not lastDirectionF
        else:
            if len(open_set_F) > 0 and (
                len(open_set_B) == 0 or open_set_F.top()[3] >= open_set_B.top()[3]
            ):
                directionF = True
            else:
                directionF = False

        # Pop the best state from OPEN
        if directionF:
            _, _, current_state, f_value = open_set_F.pop()
            closed_set_F.add(current_state)
        else:
            _, _, current_state, f_value = open_set_B.pop()
            closed_set_B.add(current_state)

        # Logs
        current_path_length = len(current_state.path) - 1
        expansions += 1
        # if expansions % 10000 == 0:
        #     print(
        #         f"Expansion #{expansions}: state {current_state.path}, f={f_value}, len={len(current_state.path)}"
        #     )
        #     print(f"closed_F: {len(closed_set_F)}. closed_B: {len(closed_set_B)}")
        #     print(f"open_F: {len(open_set_F)}. open_B: {len(open_set_B)}")

        # Check against CLOSED of the other direction
        if directionF:
            for state in closed_set_B:
                if (
                    current_state.head() == state.head()
                    and not current_state.shares_vertex_with(state, snake)
                    # and current_state.pi().isdisjoint(state.pi())
                ):
                    total_length = current_path_length + len(state.path) - 1
                    if total_length > best_path_length:
                        best_path_length = total_length
                        best_path = current_state.path[:-1] + state.path[::-1]
                        # print(f"Found longer path of length {total_length}")
        else:
            for state in closed_set_F:
                if (
                    current_state.head() == state.head()
                    and not current_state.shares_vertex_with(state, snake)
                    # and current_state.pi().isdisjoint(state.pi())
                ):
                    total_length = current_path_length + len(state.path) - 1
                    if total_length > best_path_length:
                        best_path_length = total_length
                        best_path = state.path[:-1] + current_state.path[::-1]
                        best_path_meet_point = current_state.head()
                        # print(f"Found longer path of length {total_length}")

        # Check if U is the largest it will ever be
        if best_path_length >= max(
            open_set_F.top()[3] if len(open_set_F) > 0 else float("-inf"),
            open_set_B.top()[3] if len(open_set_B) > 0 else float("-inf"),
        ):
            # print(f"Terminating with best path of length {best_path_length}")
            break

        # Generate successors
        successors = current_state.successor(snake)
        for successor in successors:
            h_value = heuristic(
                successor, goal if directionF else start, heuristic_name, snake
            )

            # # For Plotting h
            # h_BCC.append(h_value)
            # h_mis = heuristic(successor, goal if direction == "F" else start, "mis_heuristic")
            # h_MIS.append(h_mis+0.1)
            # mis_smaller_flag.append(-1 if h_value<h_mis else 0 if h_value==h_mis else 1)
            # max_f.append(f_value)
            # expansions_list.append(expansions)

            g_value = current_path_length + 1
            f_value = g_value + h_value

            if directionF:
                open_set_F.push(successor, min(f_value, 2 * h_value)) # MM
            else:
                open_set_B.push(successor, min(f_value, 2 * h_value)) # MM
    
    # # For Plotting h
    # plt.plot(expansions_list, h_MIS, label='h_MIS')
    # plt.plot(expansions_list, h_BCC, label='h_BCC')
    # plt.plot(expansions_list, max_f, label='max_f')
    # plt.plot(expansions_list, mis_smaller_flag, label='mis_smaller_flag')
    # plt.legend()
    # plt.xlabel("# expansions")
    # plt.ylabel("h value")
    # plt.savefig("h_vs_expansions.png")

    return best_path, expansions, best_path_meet_point
