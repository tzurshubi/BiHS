from os import stat
import matplotlib.pyplot as plt
from collections import deque
import heapq, time
from heuristics.heuristic import heuristic
from models.state import State
from models.openvopen import Openvopen
from models.heapq_state import HeapqState
from models.openvopen_prefixSet import Openvopen_prefixSet
from utils.utils import *

def bidirectional_dfbnb_sym_coil(graph, start, goal, heuristic_name, snake, args):
    stats = {"expansions": 0, "generated": 0, "symmetric_states_removed": 0, "dominated_states_removed": 0,
             "valid_meeting_checks": 0, "state_vs_state_meeting_checks": 0, "state_vs_prefix_meeting_checks": 0, 
             "num_of_prefix_sets": {'F': 0, 'B': 0}, "prefix_set_mean_size": {'F': 0, 'B': 0}, "paths_with_g_cutoff":{'F': 0, 'B': 0},
             "valid_meeting_check_time": 0, "calc_h_time": 0, "moved_OPEN_to_AUXOPEN": 0, "g_values": [], "BF_values": []}
    logger = args.logger 
    cube = args.graph_type == "cube"
    if not cube or not args.sym_coil:
        logger("Error: bidirectional_search_sym_coil is only for cube graphs.")
        raise ValueError("bidirectional_search_sym_coil is only for cube graphs.")
    buffer_dim = args.cube_buffer_dim if cube else None
    c_star = longest_sym_coil_lengths[args.size_of_graphs[0]]
    half_coil_upper_bound = (c_star / 2) - args.cube_first_dims
    g_cutoff_F, g_cutoff_B = half_coil_upper_bound // 2, (half_coil_upper_bound + 1) // 2
    best_path = None

    # Initial states
    graph_F, graph_B = graph.copy(), graph.copy()
    graph_F.remove_nodes_from([0] + list(graph.neighbors(0)))
    graph_F.remove_nodes_from([goal] + list(graph.neighbors(goal)))
    graph_B.remove_nodes_from([0] + list(graph.neighbors(0)) + [start] + list(graph.neighbors(start)))
    initial_state_F = State(graph_F, [start], [], snake) if isinstance(start, int) else State(graph_F, start, [], snake)
    initial_state_B = State(graph_B, [goal], [], snake) if isinstance(goal, int) else State(graph_B, goal, [], snake)

    # Push initial states with priority based on f_value
    stack_F = deque()
    stack_B = deque()
    stack_F.append(initial_state_F)
    stack_B.append(initial_state_B)
    FNV_F = {(initial_state_F.head, initial_state_F.path_vertices_and_neighbors_bitmap)}
    FNV_B = {(initial_state_B.head, initial_state_B.path_vertices_and_neighbors_bitmap)}
    states_g_cutoff_F = []
    states_g_cutoff_B = []

    directionF = True                               # True - Forward, False - Backward
    while len(stack_F) > 0 or len(stack_B) > 0:
        # Determine which direction to expand
        directionF = not directionF                 # Alternate directions
        if len(stack_F) == 0: directionF = False    # if no states in F, must expand B
        if len(stack_B) == 0: directionF = True     # if no states in B, must expand F

        # Set general variables
        D, D_hat = ('F', 'B') if directionF else ('B', 'F')
        stack_D, stack_D_hat = (stack_F, stack_B) if directionF else (stack_B, stack_F)
        FNV_D , FNV_D_hat = (FNV_F, FNV_B) if directionF else (FNV_B, FNV_F)
        states_g_cutoff_D , states_g_cutoff_D_hat = (states_g_cutoff_F, states_g_cutoff_B) if directionF else (states_g_cutoff_B, states_g_cutoff_F)
        g_cutoff = g_cutoff_F if directionF else g_cutoff_B

        # Get the best state from OPEN_D
        current_state = stack_D.pop()

        # Check against OPEN of the other direction, for a valid meeting point
        if current_state.g == g_cutoff:
            stats["paths_with_g_cutoff"][D] += 1
            if directionF: states_g_cutoff_D.append(current_state)
            # curr_time = time.time()
            # paths = OPENvOPEN.find_all_non_overlapping_paths(current_state, directionF, best_path_length, f_value, snake, stats)
            # stats["valid_meeting_check_time"] += time.time() - curr_time
            
            # for path in paths:
            #     half_coil_to_check = args.cube_first_dims_path + path
            #     is_sym_coil, sym_coil = is_half_of_symmetric_double_coil(half_coil_to_check, args.size_of_graphs[0])
            #     if is_sym_coil:
            #         logger("SYM_COIL_FOUND")
            #         logger(f"Expansion {stats["expansions"]}: Found symmetric coil of length {len(sym_coil)-1}: {sym_coil}. generated={stats["generated"]}")
            #         # return sym_coil, stats["expansions"], stats["generated"], moved_OPEN_to_AUXOPEN, best_path_meet_point, g_values

        
        # Symmetric coil pruning: do not expand states with g > half_coil_upper_bound
        if current_state.g > g_cutoff: 
            continue

        # Logging progress
        if stats["expansions"] and stats["expansions"] % 50_000 == 0:
            logger(f"Expansion {stats["expansions"]}: g={current_state.g}, path={current_state.materialize_path()}, stack_F={len(stack_F)}, stack_B={len(stack_B)}, generated={stats["generated"]}.")

        stats["expansions"] += 1

        # Generate successors
        successors = current_state.generate_successors(args, snake, directionF)
        stats["g_values"].append(current_state.g)
        stats["BF_values"].append(len(successors))
        for successor in successors:
            if args.bsd and (successor.head, successor.path_vertices_and_neighbors_bitmap) in FNV_D:
                stats["symmetric_states_removed"] += 1
                # logger(f"symmetric state removed: {successor.path}")
                # logger(f"symmetric states removed: {stats['symmetric_states_removed']}")
                continue

            stats["generated"] += 1

            # Insert successor into the stack and FNV set
            stack_D.append(successor)
            FNV_D.add((successor.head, successor.path_vertices_and_neighbors_bitmap))
            # if successor.g == g_cutoff: 
            #     OPENvOPEN.insert_state(successor, directionF, stats)

    stats['all_paths_with_g_cutoff'] = stats['paths_with_g_cutoff']['F'] + stats['paths_with_g_cutoff']['B']
    logger(f"Number of paths with g_cutoff({g_cutoff_F}/{g_cutoff_B}): {stats['all_paths_with_g_cutoff']} (F:{stats['paths_with_g_cutoff']['F']}, B:{stats['paths_with_g_cutoff']['B']})")
    
    logger("Starting DFBnB state tree search for valid meeting points...")
    F_states_checked = 0
    for state_F in states_g_cutoff_F:
        # DFBnB in the state tree of the backward direction 
        stack = deque()
        stack.append(initial_state_B)
        while len(stack) > 0:
            current_state_B = stack.pop()
            stats["valid_meeting_checks"] += 1
            if state_F.shares_vertex_with(current_state_B, snake): continue
            if current_state_B.g == g_cutoff_B:
                # Check for valid coil
                half_coil_to_check = args.cube_first_dims_path + state_F.materialize_path() + current_state_B.materialize_path()[::-1][1:]
                is_sym_coil, sym_coil = is_half_of_symmetric_double_coil(half_coil_to_check, args.size_of_graphs[0])
                if is_sym_coil:
                    logger("SYM_COIL_FOUND")
                    return sym_coil, stats
                continue
            stack.extend(current_state_B.successors)

        F_states_checked += 1
        if F_states_checked % 10 == 0:
            logger(f"Checked {F_states_checked}/{len(states_g_cutoff_F)} F states against B state tree in DFBnB for valid meeting points...")

    return best_path, stats
