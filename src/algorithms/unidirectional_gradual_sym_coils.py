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

def unidirectional_gradual_sym_coils(graph, start, goal, heuristic_name, snake, args):
    logger = args.logger 
    cube = args.graph_type == "cube"
    if not cube or not args.sym_coils:
        logger("Error: unidirectional_search_sym_coils is only for cube graphs.")
        raise ValueError("unidirectional_search_sym_coils is only for cube graphs.")
    buffer_dim = args.cube_buffer_dim if cube else None
    c_star = longest_sym_coil_lengths[args.size_of_graphs[0]]
    half_coil_upper_bound = (c_star / 2) - args.cube_first_dims
    g_upper_cutoff = half_coil_upper_bound
    g_lower_cutoff = 2
    best_path = None
    stats = {
        "expansions": 0,
        "generated": 0,
        "symmetric_states_removed": 0,
        "dominated_states_removed": 0,
        "valid_meeting_checks": 0,
        "state_vs_state_meeting_checks": 0,
        "state_vs_prefix_meeting_checks": 0,
        "prefix_vs_prefix_meeting_checks": 0,
        "num_of_prefix_sets": {g: 0 for g in range(0, math.ceil(half_coil_upper_bound))},
        "prefix_set_mean_size": 0,
        "paths_with_g_upper_cutoff": 0,
        "paths_with_g_lower_cutoff": 0,
        "valid_meeting_check_time": 0,
        "calc_h_time": 0,
        "moved_OPEN_to_AUXOPEN": 0,
        "g_values": [],
        "BF_values": [],
        "must_checks": 0
    }
    goal_neighbors_bitmap = 0
    goal_and_neighbors_bitmap = 1 << goal
    for goal_neighbor in graph.neighbors(goal):
        goal_neighbors_bitmap |= 1 << goal_neighbor
        goal_and_neighbors_bitmap |= 1 << goal_neighbor
    
    # Initial states
    initial_state = State(graph, [start], [], snake) if isinstance(start, int) else State(graph, start, [], snake)
    # OPENvOPEN = Openvopen(graph, start, goal) if args.prefix_set is None else Openvopen_prefixSet(graph, start, goal, args.prefix_set)
    # OPENvOPEN = Openvopen(graph, start, goal) if args.prefix_set is None else Openvopen_illegalVerts(graph, start, goal, args.prefix_set)

    # Push initial states to stack
    stack = deque()
    stack.append(initial_state)
    # FNV_F = {(initial_state_F.head, initial_state_F.path_vertices_and_neighbors_bitmap)}
    states_g_lower_cutoff = []

    ############################################
    # Main Search Loop
    ############################################

    known_half_coil = [15, 31, 63, 127, 119, 103, 99, 107, 75, 73, 77, 69, 68, 100, 116, 124, 120, 121, 113, 81]
    while len(stack) > 0:
        # Pop a state from the stack
        current_state = stack.pop()
        # path = current_state.materialize_path()
        # if path==known_half_coil[:len(path)]:
        #     logger(f"Part of known half coil reached: {path}")
        #     pass

        # Symmetric coil pruning: do not expand states with g > half_coil_upper_bound
        if current_state.g > g_upper_cutoff: raise ValueError("In bidirectional_gradual_sym_coils: current_state.g cannot be larger than g_upper_cutoff")

        # Logging progress
        if stats["expansions"] and stats["expansions"] % 100_000 == 0:
            # logger(f"Expansion {stats["expansions"]}: g={current_state.g}, path={current_state.materialize_path()}, stack_F={len(stack_F)}, stack_B={len(stack_B)}, generated={stats["generated"]}") # , memory [MB]: {memory_used_mb():.2f} # \n --- Stats: { {k: v for k, v in stats.items() if k not in {'g_values', 'BF_values'}} }
            logger(f"Expansion {stats["expansions"]}: g={current_state.g}, stack={len(stack)}, generated={stats["generated"]}.")

        stats["expansions"] += 1
        stats["num_of_prefix_sets"][current_state.g] += 1

        # Generate successors
        successors = current_state.generate_successors(args, snake, True)
        # stats["g_values"].append(current_state.g)
        # stats["BF_values"].append(len(successors))
        for successor in successors:
            stats["generated"] += 1
            # If successor is short and reaches the goal or its neighbors, skip it
            if successor.g < g_upper_cutoff - 1 and (1 << successor.head) & goal_and_neighbors_bitmap != 0:
                continue
            # If successor has g equal to g_upper_cutoff - 1 but doesn't reach goal neighbors, skip it
            if successor.g == g_upper_cutoff - 1 and (1 << successor.head) & goal_neighbors_bitmap != 0:
                continue
            # If successor has g equal to g_upper_cutoff but doesn't reach the goal, skip it
            if successor.g == g_upper_cutoff:
                stats["paths_with_g_upper_cutoff"] += 1
                if successor.head != goal:
                    continue
                # Check for symmetric coil
                path = current_state.materialize_path()
                half_coil_to_check = args.cube_first_dims_path + path
                is_sym_coil, sym_coil = is_half_of_symmetric_double_coil(half_coil_to_check, args.size_of_graphs[0])
                if is_sym_coil:
                    logger(f"SYM_COIL_FOUND! {sym_coil}")
                    return sym_coil, stats
                continue

            # if args.bsd and (successor.head, successor.path_vertices_and_neighbors_bitmap) in FNV_D:
            #     stats["symmetric_states_removed"] += 1
            #     # logger(f"symmetric state removed: {successor.path}")
            #     # logger(f"symmetric states removed: {stats['symmetric_states_removed']}")
            #     continue

            # Insert successor into the stack and FNV set
            stack.append(successor)
            # FNV_D.add((successor.head, successor.path_vertices_and_neighbors_bitmap))

    # Print stats
    excluded = {"g_values", "BF_values"}
    filtered_stats = {k: v for k, v in stats.items() if k not in excluded}
    args.logger(f"Stats: {filtered_stats}")

    return best_path, stats
