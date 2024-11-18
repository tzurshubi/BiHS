import matplotlib.pyplot as plt
import heapq
from heuristics.heuristic import heuristic
from models.state import State
from models.openvopen import Openvopen
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
    OPEN_F = HeapqState()
    OPEN_B = HeapqState()
    OPENvOPEN = Openvopen(max(graph.nodes)+1)

    # Initial states
    initial_state_F = State(graph, [start], snake)
    initial_state_B = State(graph, [goal], snake)

    # Initial f_values
    initial_f_value_F = heuristic(initial_state_F, goal, heuristic_name, snake)
    initial_f_value_B = heuristic(initial_state_B, start, heuristic_name, snake)

    # Push initial states with priority based on f_value
    OPEN_F.push(initial_state_F, initial_f_value_F)
    OPEN_B.push(initial_state_B, initial_f_value_B)
    OPENvOPEN.insert_state(initial_state_F, True)
    OPENvOPEN.insert_state(initial_state_B, False)

    # Best path found and its length
    best_path = None        # S in the pseudocode
    best_path_length = -1   # U in the pseudocode

    # Expansion counter
    expansions = 0

    # Closed sets for forward and backward searches
    CLOSED_F = set()
    CLOSED_B = set()

    while len(OPEN_F) > 0 or len(OPEN_B) > 0:
        # Determine which direction to expand
        directionF = None # True - Forward, False - Backward 
        if alternate:
            directionF = False if lastDirectionF else True
            lastDirectionF = not lastDirectionF
        else:
            if len(OPEN_F) > 0 and (
                len(OPEN_B) == 0 or OPEN_F.top()[3] >= OPEN_B.top()[3]
            ):
                directionF = True
            else:
                directionF = False

        # Set general variables
        D, D_hat = ('F', 'B') if directionF else ('B', 'F')
        OPEN_D, OPEN_D_hat = (OPEN_F, OPEN_B) if directionF else (OPEN_B, OPEN_F)
        CLOSED_D, CLOSED_D_hat = (CLOSED_F, CLOSED_B) if directionF else (CLOSED_B, CLOSED_F)

        # Get the best state from OPEN_D
        _, _, current_state, f_value = OPEN_D.top()

        # Logs
        current_path_length = len(current_state.path) - 1
        expansions += 1
        if expansions % 1000 == 0:
            print(
                f"Expansion #{expansions}: state {current_state.path}, f={f_value}, len={len(current_state.path)}"
            )
        #     print(f"closed_F: {len(closed_set_F)}. closed_B: {len(closed_set_B)}")
        #     print(f"open_F: {len(open_set_F)}. open_B: {len(open_set_B)}")

        # Check against OPEN of the other direction
        state = OPENvOPEN.find_highest_non_overlapping_state(current_state,directionF, snake)
        if state:
            total_length = current_path_length + len(state.path) - 1
            if total_length > best_path_length:
                best_path_length = total_length
                best_path = current_state.path[:-1] + state.path[::-1]
                best_path_meet_point = current_state.head()
                # print(f"Found longer path of length {total_length}")

        # Check if U is the largest it will ever be
        if best_path_length >= min(
            OPEN_F.top()[3] if len(OPEN_F) > 0 else float("inf"),
            OPEN_B.top()[3] if len(OPEN_B) > 0 else float("inf"),
        ):
            # print(f"Terminating with best path of length {best_path_length}")
            break

        # Get the current state from OPEN_D TO CLOSED_D
        _, _, current_state, f_value = OPEN_D.pop()
        OPENvOPEN.remove_state(current_state, directionF)
        CLOSED_D.add(current_state)

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
            OPEN_D.push(successor, min(f_value, 2 * h_value)) # MM # ,2*(OPT-g_value)
            OPENvOPEN.insert_state(successor,directionF)

    
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
