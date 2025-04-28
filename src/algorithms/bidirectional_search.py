import matplotlib.pyplot as plt
import heapq, time
from heuristics.heuristic import heuristic
from models.state import State
from models.openvopen import Openvopen
from models.heapq_state import HeapqState
from utils.utils import *



def bidirectional_search(graph, start, goal, heuristic_name, snake, args):
    # # For Plotting h
    # mis_smaller_flag = []
    # expansions_list = []
    # h_MIS = []
    # h_BCC = []
    # max_f = []
    calc_h_time = 0

    # Options
    alternate = False # False
    lastDirectionF = False

    # Initialize meeting point of the two searches
    best_path_meet_point = None

    # Initialize custom priority queues for forward and backward searches
    OPEN_F = HeapqState()
    OPEN_B = HeapqState()
    OPENvOPEN = Openvopen(max(graph.nodes)+1)

    # Initial states
    initial_state_F = State(graph, [start], snake) if isinstance(start, int) else State(graph, start, snake)
    initial_state_B = State(graph, [goal], snake) if isinstance(goal, int) else State(graph, goal, snake)

    # Initial f_values
    initial_state_F.h = heuristic(initial_state_F, goal, heuristic_name, snake)
    initial_f_value_F = initial_state_F.g + initial_state_F.h
    initial_state_B.h = heuristic(initial_state_B, start, heuristic_name, snake)
    initial_f_value_B = initial_state_B.g + initial_state_B.h

    # Push initial states with priority based on f_value
    OPEN_F.push(initial_state_F, initial_f_value_F)
    OPEN_B.push(initial_state_B, initial_f_value_B)
    OPENvOPEN.insert_state(initial_state_F, True)
    OPENvOPEN.insert_state(initial_state_B, False)

    # Best path found and its length
    best_path = None        # S in the pseudocode
    best_path_length = -1   # U in the pseudocode

    # Expansion counter, generated counter
    expansions = 0
    generated = 0
    moved_OPEN_to_AUXOPEN = 0

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
        current_path_length = len(current_state.path) - 1
        
        # if expansions % 100000 == 0:
        #     print(
        #         f"Expansion #{expansions}: state {current_state.path}, f={f_value}, len={len(current_state.path)}"
        #     )
        #     print(f"closed_F: {len(closed_set_F)}. closed_B: {len(closed_set_B)}")
        #     print(f"open_F: {len(open_set_F)}. open_B: {len(open_set_B)}")

        # Check against OPEN of the other direction
        state = OPENvOPEN.find_highest_non_overlapping_state(current_state,directionF, best_path_length, snake)
        if state:
            total_length = current_path_length + len(state.path) - 1
            if total_length > best_path_length:
                best_path_length = total_length
                best_path = current_state.path[:-1] + state.path[::-1]
                best_path_meet_point = current_state.head
                if snake and total_length >= f_value-3:
                    print(f"[{time2str(args.start_time,time.time())} expansion {expansions}, {time_ms(args.start_time,time.time())}] Found path of length {total_length}: {best_path}. g_F={current_path_length}, g_B={len(state.path) - 1}, f_max={f_value}, generated={generated}")
                    # with open(args.log_file_name, 'a') as file:
                    #     file.write(f"[{time2str(args.start_time,time.time())} expansion {expansions}] Found path of length {total_length}. {best_path}. g_F={current_path_length}, g_B={len(state.path) - 1}, f_max={f_value}\n")

        # Termination Condition: check if U is the largest it will ever be
        if best_path_length >= min(
            OPEN_F.top()[3] if len(OPEN_F) > 0 else float("inf"),
            OPEN_B.top()[3] if len(OPEN_B) > 0 else float("inf"),
        ):
            # print(f"Terminating with best path of length {best_path_length}")
            break

        # New Check by Shimony. if g > f_max/2 don't expant it, but keep it in OPENvOPEN for checking collision of search from the other side
        # if C* = 20, in the F direction we won't expand S with g > 9, in the B direction we won't expand S with g > 9.5 
        # if C* = 19, in the F direction we won't expand S with g > 8.5, in the B direction we won't expand S with g > 9 
        if (D=='F' and current_state.g > f_value/2 - 1) or (D=='B' and current_state.g > (f_value - 1)/2): 
            OPEN_D.pop()
            moved_OPEN_to_AUXOPEN += 1
            # print(f"Not expanding state {current_state.path} because state.g = {current_state.g}")
            continue

        expansions += 1

        # Get the current state from OPEN_D TO CLOSED_D
        _, _, current_state, f_value = OPEN_D.pop()
        OPENvOPEN.remove_state(current_state, directionF)
        CLOSED_D.add(current_state)

        # Generate successors
        successors = current_state.successor(args, snake, directionF)
        for successor in successors:
            generated += 1
            curr_time = time.time()
            h_successor = heuristic(
                successor, goal if directionF else start, heuristic_name, snake
            )
            calc_h_time += time.time() - curr_time

            # # For Plotting h
            # h_BCC.append(h_value)
            # h_mis = heuristic(successor, goal if direction == "F" else start, "mis_heuristic")
            # h_MIS.append(h_mis+0.1)
            # mis_smaller_flag.append(-1 if h_value<h_mis else 0 if h_value==h_mis else 1)
            # max_f.append(f_value)
            # expansions_list.append(expansions)

            g_successor = current_path_length + 1
            f_successor = g_successor + h_successor
            # if f_value<f_successor:
            #     print(f"f_value {f_value}")
            OPEN_D.push(successor, min(2 * h_successor, f_value, f_successor)) # MM:  min(2 * h_successor, f_value,f_successor)
            # 
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
    # print(f"total time for calculating heuristics: {1000*calc_h_time}")
    return best_path, expansions, generated, moved_OPEN_to_AUXOPEN, best_path_meet_point
