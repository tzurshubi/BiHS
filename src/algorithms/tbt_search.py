import matplotlib.pyplot as plt
import heapq, time
from heuristics.heuristic import heuristic
from models.state import State
from models.openvopen import Openvopen
from models.heapq_state import HeapqState
from utils.utils import *



def tbt_search(graph, start, goal, heuristic_name, snake, args):
    stats = {"expansions": 0, "generated": 0, "symmetric_states_removed": 0, "dominated_states_removed": 0}
    logger = args.logger 
    cube = args.graph_type == "cube"
    d = args.size_of_graphs[0] if cube else None
    buffer_dim = args.cube_buffer_dim if cube else None
    backward_sym_generation = args.backward_sym_generation
    st_states = []
    calc_h_time = 0

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
    initial_state_F = State(graph, [start], [], snake, args) if isinstance(start, int) else State(graph, start, [], snake, args)
    initial_state_B = State(graph, [goal], [], snake, args) if isinstance(goal, int) else State(graph, goal, [], snake, args)

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
    stats["expansions"] = 0
    stats["generated"] = 0
    stats["moved_OPEN_to_AUXOPEN"] = 0

    # Closed sets for forward and backward searches
    CLOSED_F = set()
    CLOSED_B = set()

    while len(OPEN_F) > 0 or len(OPEN_B) > 0:
        # Determine which direction to expand
        directionF = None # True - Forward, False - Backward 
        if cube and backward_sym_generation: 
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

        # Debug:
        if current_state.materialize_path() == [0,1,3,19,18]:
            pass
        if current_state.materialize_path() == [31,29,21,20,22,18]:
            pass

        # Check against OPEN of the other direction, for a valid meeting point
        curr_time = time.time()
        current_st_states, _, _ = OPENvOPEN.find_all_non_overlapping_paths(current_state, directionF, best_path_length, f_value, snake)
        for current_st_state in current_st_states:
            for complementary_st_state in st_states:
                valid_coil, p1, p2 = check_2_st_paths_form_coil(current_st_state, complementary_st_state, d)
                if valid_coil:
                    total_length = current_st_state.g + complementary_st_state.g
                    if total_length > best_path_length:
                        best_path_length = total_length
                        best_path = p1[::-1] + p2[1:]
                        # print(f"p1: {p1}")
                        # print(f"p2: {p2}")
                        # print(f"New best path of length {best_path_length}: {best_path}")
                        if snake:
                            logger(f"Expansion {stats['expansions']}: Found path of length {total_length}: {best_path}. g_F={current_path_length}, g_B={current_st_state.g}, f_max={f_value}, generated={stats['generated']}")
            st_states.append(current_st_state)

        # Termination Condition: check if U is the largest it will ever be
        if best_path_length >= longest_coil_lengths.get(d, float('-inf')):
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
        if stats["expansions"] and stats["expansions"] % 100 == 0:
            logger(f"Expansion {stats['expansions']}: f={f_value}, g={current_state.g}, st_paths={len(st_states)}, best_path_length={best_path_length}, generated={stats['generated']}")
        #     print(f"closed_F: {len(closed_set_F)}. closed_B: {len(closed_set_B)}")
        #     print(f"open_F: {len(open_set_F)}. open_B: {len(open_set_B)}")

        stats["expansions"] += 1
        g_values.append(current_state.g)

        # Get the current state from OPEN_D TO CLOSED_D
        f_value, g_value, current_state = OPEN_D.pop()
        # OPENvOPEN.remove_state(current_state, directionF)
        # CLOSED_D.add(current_state)

        # Generate successors
        successors = current_state.generate_successors(args, snake, directionF)
        BF_values.append(len(successors))
        for successor in successors:
            if args.bsd and (successor.head, successor.path_vertices_and_neighbors_bitmap if snake else successor.path_vertices_bitmap) in FNV_D:
                # logger(f"symmetric state removed: {successor.path}")
                stats["symmetric_states_removed"] += 1
                continue

            # Debug:
            if successor.materialize_path() == [0,1,3,19,18]:
                pass
            if successor.materialize_path() == [31,29,21,20,22,18]:
                pass

            # Check if successor traverses the buffer dimension in cube graphs
            if has_bridge_edge_across_dim(current_state, successor, buffer_dim):
                successor.traversed_buffer_dimension = True
                if not directionF: 
                    OPENvOPEN.insert_state(successor,directionF)
                    continue  # do not add backward states that traversed buffer dimension to OPEN_B

            stats["generated"] += 1

            # Calculate g, h, f values for successor
            curr_time = time.time()
            h_successor = heuristic(
                successor, goal if directionF else start, heuristic_name, snake
            )
            calc_h_time += time.time() - curr_time
            g_successor = current_path_length + 1
            f_successor = g_successor + h_successor

            # The state symmetric to successor should be inserted to OPEN_D_hat
            if cube and backward_sym_generation: 
                successor_symmetric = symmetric_state_transform(successor, args.dim_flips_F_B_symmetry, args.dim_swaps_F_B_symmetry)

            # XMM_light + PathMin
            if args.algo == "light" or args.algo == "full":
                OPEN_D.push(successor, min(2 * h_successor, f_value, f_successor))
                if cube and backward_sym_generation: 
                    OPEN_D_hat.push(successor_symmetric, min(2 * h_successor, f_value, f_successor))
            else: 
                OPEN_D.push(successor, min(f_value, f_successor))
                if cube and backward_sym_generation: 
                    OPEN_D_hat.push(successor_symmetric, min(f_value, f_successor))
            
            FNV_D.add((successor.head, successor.path_vertices_and_neighbors_bitmap if snake else successor.path_vertices_bitmap))
            OPENvOPEN.insert_state(successor,directionF)
            if cube and backward_sym_generation: 
                OPENvOPEN.insert_state(successor_symmetric, not directionF)
    
    # for st_state in st_states:
    #     print(st_state.path[::-1])
    #     if st_state.path[::-1] == [0,1,3,19,18,22,20,21,29,31]:
    #         pass
    #     if st_state.path[::-1] == [0,8,10,14,15,31]:
    #         pass
    return best_path, stats
