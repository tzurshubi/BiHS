from os import stat
import queue
import matplotlib.pyplot as plt
from collections import deque
import heapq, time
from heuristics.heuristic import heuristic
from models.state import State
from models.openvopen import Openvopen
from models.heapq_state import HeapqState
from models.openvopen_prefixSet import Openvopen_prefixSet
from models.openvopen_illegalVerts import Openvopen_illegalVerts
from utils.utils import *

def bidirectional_gradual_sym_coils(graph, start, goal, heuristic_name, snake, args):
    logger = args.logger 
    cube = args.graph_type == "cube"
    if not cube or not args.sym_coils:
        logger("Error: bidirectional_search_sym_coils is only for cube graphs.")
        raise ValueError("bidirectional_search_sym_coils is only for cube graphs.")
    buffer_dim = args.cube_buffer_dim if cube else None
    c_star = longest_sym_coil_lengths[args.size_of_graphs[0]]
    half_coil_upper_bound = (c_star / 2) - args.cube_first_dims
    g_upper_cutoff_F, g_upper_cutoff_B = half_coil_upper_bound // 2, (half_coil_upper_bound + 1) // 2
    g_lower_cutoff = 2
    best_path = None
    stats = {
        "expansions": 0,
        "generated": {'F': 0, 'B': 0},
        "symmetric_states_removed": 0,
        "dominated_states_removed": 0,
        "valid_meeting_checks": 0,
        "state_vs_state_meeting_checks": 0,
        "state_vs_prefix_meeting_checks": 0,
        "prefix_vs_prefix_meeting_checks": 0,
        "num_of_prefix_sets": {
            'F': {g: 0 for g in range(0, math.ceil(half_coil_upper_bound))},
            'B': {g: 0 for g in range(0, math.ceil(half_coil_upper_bound))}
        },
        "prefix_set_mean_size": {'F': 0, 'B': 0},
        "paths_with_g_upper_cutoff": {'F': 0, 'B': 0},
        "paths_with_g_lower_cutoff": {'F': 0, 'B': 0},
        "valid_meeting_check_time": 0,
        "calc_h_time": 0,
        "moved_OPEN_to_AUXOPEN": 0,
        "g_values": [],
        "BF_values": [],
        "must_checks": 0
    }

    
    # Initial states
    graph_F, graph_B = graph.copy(), graph.copy()
    graph_F.remove_nodes_from([0] + list(graph.neighbors(0)))
    graph_F.remove_nodes_from([goal] + list(graph.neighbors(goal)))
    graph_B.remove_nodes_from([0] + list(graph.neighbors(0)) + [start] + list(graph.neighbors(start)))
    initial_state_F = State(graph_F, [start], [], snake) if isinstance(start, int) else State(graph_F, start, [], snake)
    initial_state_B = State(graph_B, [goal], [], snake) if isinstance(goal, int) else State(graph_B, goal, [], snake)
    # OPENvOPEN = Openvopen(graph, start, goal) if args.prefix_set is None else Openvopen_prefixSet(graph, start, goal, args.prefix_set)
    OPENvOPEN = Openvopen(graph, start, goal) if args.prefix_set is None else Openvopen_illegalVerts(graph, start, goal, args.prefix_set)

    # Push initial states with priority based on f_value
    stack_F = deque()
    stack_B = deque()
    stack_F.append(initial_state_F)
    stack_B.append(initial_state_B)
    FNV_F = {(initial_state_F.head, initial_state_F.path_vertices_and_neighbors_bitmap)}
    FNV_B = {(initial_state_B.head, initial_state_B.path_vertices_and_neighbors_bitmap)}
    states_g_lower_cutoff_F = []
    states_g_lower_cutoff_B = []

    ############################################
    # Main Search Loop
    ############################################

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
        states_g_lower_cutoff_D , states_g_lower_cutoff_D_hat = (states_g_lower_cutoff_F, states_g_lower_cutoff_B) if directionF else (states_g_lower_cutoff_B, states_g_lower_cutoff_F)
        g_upper_cutoff_D, g_upper_cutoff_D_hat = (g_upper_cutoff_F, g_upper_cutoff_B) if directionF else (g_upper_cutoff_B, g_upper_cutoff_F)

        # Pop a state from the stack
        current_state = stack_D.pop()

        # Check if g is equal to g_lower_cutoff
        if current_state.g == g_lower_cutoff:
            stats["paths_with_g_lower_cutoff"][D] += 1
            states_g_lower_cutoff_D.append(current_state)

        # Check if g is equal to g_upper_cutoff
        if current_state.g == g_upper_cutoff_D:
            stats["paths_with_g_upper_cutoff"][D] += 1
            current_state.parent.set_can_reach(current_state.head)
            OPENvOPEN.insert_state(current_state, directionF)

            # Check for symmetric coil
            paths = OPENvOPEN.find_all_non_overlapping_paths(current_state, directionF, None, None, snake, stats)
            for path in paths:
                half_coil_to_check = args.cube_first_dims_path + path
                is_sym_coil, sym_coil = is_half_of_symmetric_double_coil(half_coil_to_check, args.size_of_graphs[0])
                if is_sym_coil:
                    logger(f"SYM_COIL_FOUND! {sym_coil}")
                    best_path = sym_coil

            continue

        # Symmetric coil pruning: do not expand states with g > half_coil_upper_bound
        if current_state.g > g_upper_cutoff_D: raise ValueError("In bidirectional_gradual_sym_coils: current_state.g cannot be larger than g_upper_cutoff")

        # Logging progress
        if stats["expansions"] and stats["expansions"] % 1_000 == 0:
            # logger(f"Expansion {stats["expansions"]}: g={current_state.g}, path={current_state.materialize_path()}, stack_F={len(stack_F)}, stack_B={len(stack_B)}, generated={stats["generated"]}")
            logger(f"Expansion {stats["expansions"]}: g={current_state.g}, path={current_state.materialize_path()}, stack_F={len(stack_F)}, stack_B={len(stack_B)}, generated={stats["generated"]}, memory [MB]: {memory_used_mb():.2f}")

        stats["expansions"] += 1
        stats["num_of_prefix_sets"][D][current_state.g] += 1

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

            stats["generated"][D] += 1

            # Insert successor into the stack and FNV set
            stack_D.append(successor)
            FNV_D.add((successor.head, successor.path_vertices_and_neighbors_bitmap))

    stats['all_paths_with_g_upper_cutoff'] = stats['paths_with_g_upper_cutoff']['F'] + stats['paths_with_g_upper_cutoff']['B']
    stats['all_paths_with_g_lower_cutoff'] = stats['paths_with_g_lower_cutoff']['F'] + stats['paths_with_g_lower_cutoff']['B']
    logger(f"Number of paths with g_upper_cutoff({g_upper_cutoff_F}/{g_upper_cutoff_B}): {stats['all_paths_with_g_upper_cutoff']} (F:{stats['paths_with_g_upper_cutoff']['F']}, B:{stats['paths_with_g_upper_cutoff']['B']})")
    logger(f"Number of paths with g_lower_cutoff({g_lower_cutoff}): {stats['all_paths_with_g_lower_cutoff']} (F:{stats['paths_with_g_lower_cutoff']['F']}, B:{stats['paths_with_g_lower_cutoff']['B']})")
    
    # Print stats
    excluded = {"g_values", "BF_values"}
    filtered_stats = {k: v for k, v in stats.items() if k not in excluded}
    args.logger(f"Stats: {filtered_stats}")

    # logger(f"Starting checks...")
    # sym_coil_found, sym_coil = OPENvOPEN.check_all_valid_paths_if_sym_coil(snake, args, stats)
    # if sym_coil_found:
    #     best_path = sym_coil

    ############################################
    # Checking for valid meeting points
    ############################################

    def check_states(state_F, state_B):
        # Errors
        if state_F.g != state_B.g or g_upper_cutoff_F != g_upper_cutoff_B:
            raise ValueError("In check_states: state_F.g must be equal to state_B.g, and g_upper_cutoff_F must be equal to g_upper_cutoff_B")
        g = state_F.g # == state_B.g
        g_upper_cutoff = g_upper_cutoff_F  # == g_upper_cutoff_B

        # Logs
        stats["valid_meeting_checks"] += 1
        if g == g_upper_cutoff: stats["state_vs_state_meeting_checks"] += 1
        elif g < g_upper_cutoff: stats["prefix_vs_prefix_meeting_checks"] += 1
        else: stats["state_vs_prefix_meeting_checks"] += 1
        if stats["valid_meeting_checks"] % 5_000_000 == 0:
            logger(f"Valid meeting checks so far: {stats['valid_meeting_checks']}, memory [MB]: {memory_used_mb():.2f}, state_vs_state: {stats['state_vs_state_meeting_checks']}, state_vs_prefix: {stats['state_vs_prefix_meeting_checks']}, prefix_vs_prefix: {stats['prefix_vs_prefix_meeting_checks']}")
        
        # Checks
        if g < g_upper_cutoff and not state_F.shares_reachable_vertex_with(state_B):
            return False, None
        if state_F.shares_vertex_with(state_B, snake):
            return False, None
        if g == g_upper_cutoff:
            if state_F.head != state_B.head:
                return False, None
            stats["must_checks"] += 1
            half_coil_to_check = args.cube_first_dims_path + state_F.materialize_path() + state_B.materialize_path()[::-1][1:]
            is_sym_coil, sym_coil = is_half_of_symmetric_double_coil(half_coil_to_check, args.size_of_graphs[0])
            if is_sym_coil:
                logger(f"SYM_COIL_FOUND! {sym_coil}")
            return is_sym_coil, sym_coil
        else:
            for succ_F in state_F.successors:
                for succ_B in state_B.successors:
                    check_states(succ_F, succ_B)
            # Expand the state with the smaller g value
            # if state_F.g < state_B.g:
            #     for succ in state_F.successors: check_states(succ, state_B)
            # else:
            #     for succ in state_B.successors: check_states(state_F, succ)

    # Checks
    # logger(f"Starting checks. memory [MB]: {memory_used_mb():.2f}")
    # check_states(initial_state_F, initial_state_B)

    # largest_g_checked = 0
    # while len(checks_queue) > 0:
    #     state_F, state_B = checks_queue.popleft()

    #     # Logging progress
    #     if state_F.g > largest_g_checked or state_B.g > largest_g_checked:
    #         largest_g_checked = max(state_F.g, state_B.g)
    #         logger(f"Largest g value checked so far: {largest_g_checked}, queue size: {len(checks_queue)}")
    #     stats["valid_meeting_checks"] += 1
    #     if stats["valid_meeting_checks"] % 1_000_000 == 0:
    #         logger(f"Valid meeting checks so far: {stats['valid_meeting_checks']}, queue size: {len(checks_queue)}, memory [MB]: {memory_used_mb():.2f}")
        
    #     # Check if states share a vertex
    #     if state_F.shares_vertex_with(state_B, snake):
    #         # logger(f"SHARED_VERTEX between states with lengths:{state_F.g}, {state_B.g}")
    #         continue
    #     if state_F.g == g_upper_cutoff_F and state_B.g == g_upper_cutoff_B:
    #         half_coil_to_check = args.cube_first_dims_path + state_F.materialize_path() + state_B.materialize_path()[::-1][1:]
    #         is_sym_coil, sym_coil = is_half_of_symmetric_double_coil(half_coil_to_check, args.size_of_graphs[0])
    #         stats["state_vs_state_meeting_checks"] += 1
    #         if is_sym_coil:
    #             logger("SYM_COIL_FOUND")
    #             logger(f"Expansion {stats["expansions"]}: Found symmetric coil of length {len(sym_coil)-1}: {sym_coil}. generated={stats["generated"]}")
    #     else:
    #         if state_F.g < g_upper_cutoff_F and state_B.g < g_upper_cutoff_B: stats["prefix_vs_prefix_meeting_checks"] += 1
    #         else: stats["state_vs_prefix_meeting_checks"] += 1
    #         # Expand the state with the smaller g value
    #         if state_F.g < state_B.g:
    #             checks_queue.extend((succ, state_B) for succ in state_F.successors)
    #         else:
    #             checks_queue.extend((state_F, succ) for succ in state_B.successors)
    
    return best_path, stats
