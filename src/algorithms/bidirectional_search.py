import heapq
from heuristics.heuristic import heuristic
from models.state import State
from models.heapq_state import HeapqState


def biHS_for_LSP(graph, start, goal, heuristic_name):
    # Options
    alternate = False
    lastDirectionF = False

    # Initialize meeting point of the two searches
    best_path_meet_point = None

    # Initialize custom priority queues for forward and backward searches
    open_set_F = HeapqState()
    open_set_B = HeapqState()

    # Initial states
    initial_state_F = State(graph, [start])
    initial_state_B = State(graph, [goal])

    # Initial f_values
    initial_f_value_F = heuristic(initial_state_F, goal, heuristic_name)
    initial_f_value_B = heuristic(initial_state_B, start, heuristic_name)

    # Push initial states with priority based on f_value
    open_set_F.push(initial_state_F, initial_f_value_F)
    open_set_B.push(initial_state_B, initial_f_value_B)

    # Best path found and its length
    best_path = None
    best_path_length = -1

    # Expansion counter
    expansions = 0

    # Closed sets for forward and backward searches
    closed_set_F = set()
    closed_set_B = set()

    while len(open_set_F) > 0 or len(open_set_B) > 0:
        # Determine which direction to expand
        direction = None

        if alternate:
            if lastDirectionF == True:
                direction = "B"
            else:
                direction = "F"
            lastDirectionF = not lastDirectionF
        else:
            if len(open_set_F) > 0 and (
                len(open_set_B) == 0 or open_set_F.top()[3] >= open_set_B.top()[3]
            ):
                direction = "F"
            else:
                direction = "B"

        if direction == "F":
            _, _, current_state, f_value = open_set_F.pop()
            closed_set_F.add(current_state)
        else:
            _, _, current_state, f_value = open_set_B.pop()
            closed_set_B.add(current_state)

        # Logs
        current_path_length = len(current_state.path) - 1
        expansions += 1
        # if expansions % 2001 == 0:
        #     print(
        #         f"Expansion #{expansions}: state {current_state.path}, f={f_value}, len={len(current_state.path)}"
        #     )
        #     print(f"closed_F: {len(closed_set_F)}. closed_B: {len(closed_set_B)}")
        #     print(f"open_F: {len(open_set_F)}. open_B: {len(open_set_B)}")

        # Check against CLOSED of the other direction
        if direction == "F":
            for state in closed_set_B:
                if (
                    current_state.head() == state.head()
                    and not current_state.shares_vertex_with(state)
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
                    and not current_state.shares_vertex_with(state)
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
        successors = current_state.successor()
        for successor in successors:
            h_value = heuristic(
                successor, goal if direction == "F" else start, heuristic_name
            )
            g_value = current_path_length + 1
            f_value = g_value + h_value

            if direction == "F":
                open_set_F.push(
                    successor, min(f_value, 2 * h_value)
                )  # 23.7 tzur used to be f_value
            else:
                open_set_B.push(
                    successor, min(f_value, 2 * h_value)
                )  # 23.7 tzur used to be f_value

    return best_path, expansions, best_path_meet_point
