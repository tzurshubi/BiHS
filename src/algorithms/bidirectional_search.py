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
    logger = args.logger 
    cube = args.graph_type == "cube"
    buffer_dim = args.cube_buffer_dimension if cube else None
    calc_h_time = 0
    valid_meeting_check_time = 0
    valid_meeting_checks = 0
    valid_meeting_checks_sum_g_under_f_max = 0
    g_values = []
    BF_values = []

    # Options
    alternate = False # False
    lastDirectionF = False

    # Initialize meeting point of the two searches
    best_path_meet_point = None

    # Initialize custom priority queues for forward and backward searches
    OPEN_F = HeapqState()
    OPEN_B = HeapqState()
    OPENvOPEN = Openvopen(graph, start, goal)

    # Initial states
    initial_state_F = State(graph, [start], [], snake) if isinstance(start, int) else State(graph, start, [], snake)
    initial_state_B = State(graph, [goal], [], snake) if isinstance(goal, int) else State(graph, goal, [], snake)

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
    FNV_F = {(initial_state_F.head, initial_state_F.path_vertices_and_neighbors_bitmap if snake else initial_state_F.path_vertices_bitmap)}
    FNV_B = {(initial_state_B.head, initial_state_B.path_vertices_and_neighbors_bitmap if snake else initial_state_B.path_vertices_bitmap)}

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
        if cube and args.symmetrical_generation_in_other_frontier: 
            directionF = True
            if len(OPEN_F) == 0: break
        elif alternate:
            directionF = False if lastDirectionF else True
            lastDirectionF = not lastDirectionF
        else:
            if len(OPEN_F) > 0 and (
                len(OPEN_B) == 0 or OPEN_F.top()[0] >= OPEN_B.top()[0]
            ):
                directionF = True
            else:
                directionF = False

        # Set general variables
        D, D_hat = ('F', 'B') if directionF else ('B', 'F')
        OPEN_D, OPEN_D_hat = (OPEN_F, OPEN_B) if directionF else (OPEN_B, OPEN_F)
        CLOSED_D, CLOSED_D_hat = (CLOSED_F, CLOSED_B) if directionF else (CLOSED_B, CLOSED_F)
        FNV_D , FNV_D_hat = (FNV_F, FNV_B) if directionF else (FNV_B, FNV_F)

        # Get the best state from OPEN_D
        f_value, g_value, current_state = OPEN_D.top()
        current_path_length = current_state.g

        # Check against OPEN of the other direction, for a valid meeting point
        curr_time = time.time()
        state, _, _, _, num_checks, num_checks_sum_g_under_f_max = OPENvOPEN.find_longest_non_overlapping_state(current_state, directionF, best_path_length, f_value, snake)
        valid_meeting_check_time += time.time() - curr_time
        valid_meeting_checks += num_checks
        valid_meeting_checks_sum_g_under_f_max += num_checks_sum_g_under_f_max
        if state:
            total_length = current_path_length + state.g
            if total_length > best_path_length:
                best_path_length = total_length
                best_path = current_state.materialize_path()[:-1] + state.materialize_path()[::-1]
                best_path_meet_point = current_state.head
                if snake:
                    logger(f"Expansion {expansions}: Found path of length {total_length}: {best_path}. g_F={current_path_length}, g_B={state.g}, f_max={f_value}, generated={generated}")
                    
        # Termination Condition: check if U is the largest it will ever be
        if best_path_length >= min(
            OPEN_F.top()[0] if len(OPEN_F) > 0 else float("inf"),
            OPEN_B.top()[0] if len(OPEN_B) > 0 else float("inf"),
        ):
            logger(f"Upper Bound Terminatation - best path length: {best_path_length}. best path: {best_path}")
            break

        # Skip states that traverse the buffer dimension in cube graphs
        if current_state.traversed_buffer_dimension:
            OPEN_D.pop()
            continue

        # XMM_full: g cutoff. if g > f_max/2 don't expand it, but keep it in OPENvOPEN for checking collision with the other side
        if args.algo == "cutoff" or args.algo == "full":
            if (D=='F' and current_state.g > f_value/2 - 1) or (D=='B' and current_state.g > (f_value - 1)/2): 
                OPEN_D.pop()
                moved_OPEN_to_AUXOPEN += 1
                # logger(f"Not expanding state {current_state.path} because state.g = {current_state.g}")
                continue

        # Logging progress
        if expansions and expansions % 20_000 == 0:
            logger(f"Expansion {expansions}: f={f_value}, g={current_state.g}")
        #     print(f"closed_F: {len(closed_set_F)}. closed_B: {len(closed_set_B)}")
        #     print(f"open_F: {len(open_set_F)}. open_B: {len(open_set_B)}")

        expansions += 1
        g_values.append(current_state.g)

        # Get the current state from OPEN_D TO CLOSED_D
        f_value, g_value, current_state = OPEN_D.pop()
        # OPENvOPEN.remove_state(current_state, directionF)
        # CLOSED_D.add(current_state)

        # Generate successors
        successors = current_state.successor(args, snake, directionF)
        BF_values.append(len(successors))
        for successor in successors:
            if args.bsd and (successor.head, successor.path_vertices_and_neighbors_bitmap if snake else successor.path_vertices_bitmap) in FNV_D:
                # logger(f"symmetric state removed: {successor.path}")
                continue

            # Check if successor traverses the buffer dimension in cube graphs
            if has_bridge_edge_across_dim(current_state, successor, buffer_dim):
                successor.traversed_buffer_dimension = True
                if not directionF: continue  # do not add backward states that traversed buffer dimension to OPEN_B
            
            generated += 1
            
            # Calculate g, h, f values for successor
            curr_time = time.time()
            h_successor = heuristic(
                successor, goal if directionF else start, heuristic_name, snake
            )
            calc_h_time += time.time() - curr_time
            g_successor = current_path_length + 1
            f_successor = g_successor + h_successor

            # The state symmetric to successor should be inserted to OPEN_D_hat
            if cube and args.symmetrical_generation_in_other_frontier: 
                successor_symmetric = symmetric_state_transform(successor, args.dim_flips_F_B_symmetry, args.dim_swaps_F_B_symmetry)

            # XMM_light + PathMin
            if args.algo == "light" or args.algo == "full":
                OPEN_D.push(successor, min(2 * h_successor, f_value, f_successor))
                if cube and args.symmetrical_generation_in_other_frontier: 
                    OPEN_D_hat.push(successor_symmetric, min(2 * h_successor, f_value, f_successor))
            else: 
                OPEN_D.push(successor, min(f_value, f_successor))
                if cube and args.symmetrical_generation_in_other_frontier: 
                    OPEN_D_hat.push(successor_symmetric, min(f_value, f_successor))
            
            FNV_D.add((successor.head, successor.path_vertices_and_neighbors_bitmap if snake else successor.path_vertices_bitmap))
            OPENvOPEN.insert_state(successor,directionF)
            if cube and args.symmetrical_generation_in_other_frontier: 
                OPENvOPEN.insert_state(successor_symmetric, not directionF)

    # Plotting BF vs g
    # plt.plot(g_values, BF_values,marker='*',linestyle='None', color='red',markersize=8, label='BF');   
    # plt.xlabel("g value")
    # plt.ylabel("BF")
    # plt.savefig("g_vs_BF"+args.log_file_name.replace("results","")+".png")

    # Plotting h
    # plt.plot(expansions_list, h_MIS, label='h_MIS')
    # plt.plot(expansions_list, h_BCC, label='h_BCC')
    # plt.plot(expansions_list, max_f, label='max_f')
    # plt.plot(expansions_list, mis_smaller_flag, label='mis_smaller_flag')
    # plt.legend()
    # plt.xlabel("# expansions")
    # plt.ylabel("h value")
    # plt.savefig("h_vs_expansions.png")
    # print(f"total time for calculating heuristics: {1000*calc_h_time}")
    
    # Statistics logging
    # bidirectional_stats = f"valid meeting checks (g+g<f_max): {valid_meeting_checks_sum_g_under_f_max} out of {valid_meeting_checks}. time: {1000*valid_meeting_check_time:.1f} [ms]. time for heuristic calculations: {1000*calc_h_time:.1f} [ms]. # of states in OPENvOPEN: {OPENvOPEN.counter}."
    # print(f"[Bidirectional Stats] {bidirectional_stats}")
    # with open(args.log_file_name, 'a') as file:
    #     file.write(f"\n[Bidirectional Stats] {bidirectional_stats}\n")

    return best_path, expansions, generated, moved_OPEN_to_AUXOPEN, best_path_meet_point, g_values
