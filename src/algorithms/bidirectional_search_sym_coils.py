from os import stat
import matplotlib.pyplot as plt
import heapq, time
from heuristics.heuristic import heuristic
from models.state import State
from models.openvopen import Openvopen
from models.heapq_state import HeapqState
from models.openvopen_prefixSet import Openvopen_prefixSet
from utils.utils import *



def bidirectional_search_sym_coils(graph, start, goal, heuristic_name, snake, args):
    stats = {"expansions": 0, "generated": 0, "symmetric_states_removed": 0, "dominated_states_removed": 0,
             "valid_meeting_checks": 0, "state_vs_state_meeting_checks": 0, "state_vs_prefix_meeting_checks": 0, 
             "num_of_prefix_sets": {'F': 0, 'B': 0}, "prefix_set_mean_size": {'F': 0, 'B': 0},
             "valid_meeting_check_time": 0, "calc_h_time": 0, "moved_OPEN_to_AUXOPEN": 0, "g_values": [], "BF_values": []}
    logger = args.logger 
    cube = args.graph_type == "cube"
    if not cube or not args.sym_coils:
        logger("Error: bidirectional_search_sym_coils is only for cube graphs.")
        raise ValueError("bidirectional_search_sym_coils is only for cube graphs.")
    buffer_dim = args.cube_buffer_dim if cube else None
    c_star = longest_sym_coil_lengths[args.size_of_graphs[0]]
    half_coil_upper_bound = (c_star / 2) - args.cube_first_dims
    g_cutoff_F, g_cutoff_B = half_coil_upper_bound // 2, (half_coil_upper_bound + 1) // 2

    # Options
    alternate = True # False
    lastDirectionF = False

    # Initialize meeting point of the two searches
    best_path_meet_point = None

    # Initialize custom priority queues for forward and backward searches
    OPEN_F = HeapqState()
    OPEN_B = HeapqState()
    OPENvOPEN = Openvopen(graph, start, goal) if args.prefix_set is None else Openvopen_prefixSet(graph, start, goal, args.prefix_set)

    # Initial states
    graph_F, graph_B = graph.copy(), graph.copy()
    graph_F.remove_nodes_from([0] + list(graph.neighbors(0)))
    graph_F.remove_nodes_from([goal] + list(graph.neighbors(goal)))
    graph_B.remove_nodes_from([0] + list(graph.neighbors(0)) + [start] + list(graph.neighbors(start)))
    initial_state_F = State(graph_F, [start], [], snake) if isinstance(start, int) else State(graph_F, start, [], snake)
    initial_state_B = State(graph_B, [goal], [], snake) if isinstance(goal, int) else State(graph_B, goal, [], snake)

    # Initial f_values
    initial_state_F.h = heuristic(initial_state_F, goal, heuristic_name, snake)
    initial_f_value_F = initial_state_F.g + initial_state_F.h
    initial_state_B.h = heuristic(initial_state_B, start, heuristic_name, snake)
    initial_f_value_B = initial_state_B.g + initial_state_B.h

    # Push initial states with priority based on f_value
    OPEN_F.push(initial_state_F, initial_f_value_F)
    OPEN_B.push(initial_state_B, initial_f_value_B)
    FNV_F = {(initial_state_F.head, initial_state_F.path_vertices_and_neighbors_bitmap)}
    FNV_B = {(initial_state_B.head, initial_state_B.path_vertices_and_neighbors_bitmap)}

    # Best path found and its length
    best_path = None        # S in the pseudocode
    best_path_length = -1   # U in the pseudocode

    # Closed sets for forward and backward searches
    CLOSED_F = set()
    CLOSED_B = set()

    while len(OPEN_F) > 0 or len(OPEN_B) > 0:
        # Determine which direction to expand
        directionF = None # True - Forward, False - Backward 
        if cube and args.backward_sym_generation: 
            directionF = True
            if len(OPEN_F) == 0: break
        elif alternate:
            directionF = False if lastDirectionF        and len(OPEN_B) else True
            directionF = True  if not lastDirectionF    and len(OPEN_F) else False
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
        g_cutoff = g_cutoff_F if directionF else g_cutoff_B

        # Get the best state from OPEN_D
        f_value, g_value, current_state = OPEN_D.top()
        current_path_length = current_state.g

        # Check against OPEN of the other direction, for a valid meeting point
        if current_state.g == g_cutoff:
            curr_time = time.time()
            paths = OPENvOPEN.find_all_non_overlapping_paths(current_state, directionF, best_path_length, f_value, snake, stats)
            stats["valid_meeting_check_time"] += time.time() - curr_time
            
            for path in paths:
                half_coil_to_check = args.cube_first_dims_path + path
                is_sym_coil, sym_coil = is_half_of_symmetric_double_coil(half_coil_to_check, args.size_of_graphs[0])
                if is_sym_coil:
                    logger("SYM_COIL_FOUND")
                    logger(f"Expansion {stats["expansions"]}: Found symmetric coil of length {len(sym_coil)-1}: {sym_coil}. generated={stats["generated"]}")
                    # return sym_coil, stats["expansions"], stats["generated"], moved_OPEN_to_AUXOPEN, best_path_meet_point, g_values


        # Termination Condition: check if U is the largest it will ever be
        # if best_path_length >= min(
        #     OPEN_F.top()[0] if len(OPEN_F) > 0 else float("inf"),
        #     OPEN_B.top()[0] if len(OPEN_B) > 0 else float("inf"),
        # ):
        #     logger(f"Upper Bound Terminatation - best path length: {best_path_length}. best path: {best_path}")
        #     break

        # Skip states that traverse the buffer dimension in cube graphs
        if current_state.traversed_buffer_dimension:
            OPEN_D.pop()
            continue
        
        # Symmetric coil pruning: do not expand states with g > half_coil_upper_bound
        if (D=='F' and current_state.g > g_cutoff_F) or (D=='B' and current_state.g > g_cutoff_B): 
            # logger(f"Not expanding state {current_state.path} because state.g = {current_state.g} > half_coil_upper_bound = {half_coil_upper_bound}")
            OPEN_D.pop()
            continue

        # Logging progress
        if stats["expansions"] and stats["expansions"] % 20_000 == 0:
            logger(f"Expansion {stats["expansions"]}: f={f_value}, g={current_state.g}, path={current_state.materialize_path()}, OPEN_F={len(OPEN_F)}, OPEN_B={len(OPEN_B)}, best_path_length={best_path_length}, generated={stats["generated"]}.")
        #     print(f"closed_F: {len(closed_set_F)}. closed_B: {len(closed_set_B)}")
        #     print(f"open_F: {len(open_set_F)}. open_B: {len(open_set_B)}")

        stats["expansions"] += 1

        # Get the current state from OPEN_D TO CLOSED_D
        f_value, g_value, current_state = OPEN_D.pop()

        # Generate successors
        successors = current_state.successor(args, snake, directionF)
        stats["g_values"].append(current_state.g)
        stats["BF_values"].append(len(successors))
        for successor in successors:
            if args.bsd and (successor.head, successor.path_vertices_and_neighbors_bitmap) in FNV_D:
                # logger(f"symmetric state removed: {successor.path}")
                stats["symmetric_states_removed"] += 1
                # logger(f"symmetric states removed: {stats['symmetric_states_removed']}")
                continue

            # Check if successor traverses the buffer dimension in cube graphs
            if buffer_dim is not None and has_bridge_edge_across_dim(current_state, successor, buffer_dim):
                successor.traversed_buffer_dimension = True
                if not directionF: continue  # do not add backward states that traversed buffer dimension to OPEN_B

            stats["generated"] += 1

            # Calculate g, h, f values for successor
            curr_time = time.time()
            h_successor = heuristic(
                successor, goal if directionF else start, heuristic_name, snake
            )
            stats["calc_h_time"] += time.time() - curr_time
            g_successor = current_path_length + 1
            f_successor = g_successor + h_successor

            # The state symmetric to successor should be inserted to OPEN_D_hat
            if cube and args.backward_sym_generation: 
                successor_symmetric = symmetric_state_transform(successor, args.dim_flips_F_B_symmetry, args.dim_swaps_F_B_symmetry)

            # XMM_light + PathMin
            if args.algo == "light" or args.algo == "full":
                OPEN_D.push(successor, min(2 * h_successor, f_value, f_successor))
                if cube and args.backward_sym_generation: 
                    OPEN_D_hat.push(successor_symmetric, min(2 * h_successor, f_value, f_successor))
            else: 
                OPEN_D.push(successor, min(f_value, f_successor))
                if cube and args.backward_sym_generation: 
                    OPEN_D_hat.push(successor_symmetric, min(f_value, f_successor))
            
            FNV_D.add((successor.head, successor.path_vertices_and_neighbors_bitmap))
            if successor.g == g_cutoff: 
                OPENvOPEN.insert_state(successor, directionF)
            if cube and args.backward_sym_generation: 
                OPENvOPEN.insert_state(successor_symmetric, not directionF)
    
    # Statistics logging
    # bidirectional_stats = f"valid meeting checks (g+g<f_max): {valid_meeting_checks_sum_g_under_f_max} out of {valid_meeting_checks}. time: {1000*valid_meeting_check_time:.1f} [ms]. time for heuristic calculations: {1000*calc_h_time:.1f} [ms]. # of states in OPENvOPEN: {OPENvOPEN.counter}."
    # print(f"[Bidirectional Stats] {bidirectional_stats}")
    # with open(args.log_file_name, 'a') as file:
    #     file.write(f"\n[Bidirectional Stats] {bidirectional_stats}\n")
    logger(f"Total number of paths with g == g_cutoff({g_cutoff_F}/{g_cutoff_B}) found: {OPENvOPEN.counter}")
    return best_path, stats, best_path_meet_point
