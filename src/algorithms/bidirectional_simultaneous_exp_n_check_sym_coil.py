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

def bidirectional_simultaneous_exp_n_check_sym_coil(graph, start, goal, heuristic_name, snake, args):
    logger = args.logger 
    cube = args.graph_type == "cube"
    if not cube or not args.sym_coil:
        logger("Error: bidirectional_search_sym_coil is only for cube graphs.")
        raise ValueError("bidirectional_search_sym_coil is only for cube graphs.")
    buffer_dim = args.cube_buffer_dim if cube else None
    c_star = longest_sym_coil_lengths[args.size_of_graphs[0]]
    half_coil_upper_bound = (c_star / 2) - args.cube_first_dims
    g_upper_cutoff_F, g_upper_cutoff_B = half_coil_upper_bound // 2, (half_coil_upper_bound + 1) // 2
    best_path = None
    shortest_path_length_start_goal = nx.shortest_path_length(graph, source=start, target=goal)
    if (shortest_path_length_start_goal % 2) != (half_coil_upper_bound % 2):
        logger(f"start ({start}) and goal ({goal}) are at odd distance {shortest_path_length_start_goal} in Q_{args.size_of_graphs[0]}, so a path of length {half_coil_upper_bound} doesn't exist between them, so no symmetric coil of length {c_star} exists.")
        return None, None
    stats = {
        "expansions": 0,
        "generated": {'F': 0, 'B': 0},
        "symmetric_states_removed": 0,
        "dominated_states_removed": 0,
        "valid_meeting_checks": 0,
        "state_vs_state_meeting_checks": 0,
        "state_vs_prefix_meeting_checks": 0,
        "prefix_vs_prefix_meeting_checks": 0,
        "num_of_states_per_g": {
            'F': {g: 0 for g in range(0, math.ceil(half_coil_upper_bound))},
            'B': {g: 0 for g in range(0, math.ceil(half_coil_upper_bound))}
        },
        "violations_per_g": {g: 0 for g in range(0, math.ceil(half_coil_upper_bound))},
        "prefix_set_mean_size": {'F': 0, 'B': 0},
        "paths_with_g_upper_cutoff": {'F': 0, 'B': 0},
        "paths_with_g_lower_cutoff": {'F': 0, 'B': 0},
        "valid_meeting_check_time": 0,
        "calc_h_time": 0,
        "moved_OPEN_to_AUXOPEN": 0,
        "g_values": [],
        "BF_values": [],
        "must_checks": 0,
        "g_upper_cutoff": {'F': g_upper_cutoff_F, 'B': g_upper_cutoff_B},
    }

    
    # Initial states
    graph_F, graph_B = graph.copy(), graph.copy()
    graph_F.remove_nodes_from([goal] + list(graph.neighbors(goal)))
    graph_B.remove_nodes_from([start] + list(graph.neighbors(start)))
    initial_state_F = State(graph_F, [start], [], snake, args) if isinstance(start, int) else State(graph_F, start, [], snake, args)
    initial_state_B = State(graph_B, [goal], [], snake, args) if isinstance(goal, int) else State(graph_B, goal, [], snake, args)
    
    # we compute a bitmask of the illegal vertices, which we removed from the graph earlier. 
    # This is used in heuristic
    illegal = 0
    for v in range(2**args.size_of_graphs[0]):
        if v not in graph:
            illegal |= (1 << v)
    initial_state_F.illegal = illegal
    initial_state_B.illegal = illegal

    # Push initial states with priority based on f_value
    stack_F = deque()
    stack_B = deque()
    stack_F.append(initial_state_F)
    stack_B.append(initial_state_B)
    # FNV_F = {(initial_state_F.head, initial_state_F.path_vertices_and_neighbors_bitmap)}
    # FNV_B = {(initial_state_B.head, initial_state_B.path_vertices_and_neighbors_bitmap)}
    states_g_lower_cutoff_F = []
    states_g_lower_cutoff_B = []

    ############################################
    # Main Search Loop
    ############################################

    def exp_n_check_states(state_F, state_B):
        # Errors (relevant only if the two frontier advance together, and state_F.g == state_B.g at all times, which is not necessarily the case)
        # if state_F.g != state_B.g or g_upper_cutoff_F != g_upper_cutoff_B:
        #     raise ValueError("In check_states: state_F.g must be equal to state_B.g, and g_upper_cutoff_F must be equal to g_upper_cutoff_B")
        # g = state_F.g # == state_B.g
        # g_upper_cutoff = g_upper_cutoff_F  # == g_upper_cutoff_B

        # Logs
        stats["valid_meeting_checks"] += 1
        if state_F.g == g_upper_cutoff_F and state_B.g == g_upper_cutoff_B: stats["state_vs_state_meeting_checks"] += 1
        elif state_F.g < g_upper_cutoff_F and state_B.g < g_upper_cutoff_B: stats["prefix_vs_prefix_meeting_checks"] += 1
        else: stats["state_vs_prefix_meeting_checks"] += 1
        if stats["valid_meeting_checks"] % 10_000 == 0:
            logger(f"Valid meeting checks so far: {stats['valid_meeting_checks']}, state_vs_state: {stats['state_vs_state_meeting_checks']}, state_vs_prefix: {stats['state_vs_prefix_meeting_checks']}, prefix_vs_prefix: {stats['prefix_vs_prefix_meeting_checks']}")
            # logger(f"Valid meeting checks so far: {stats['valid_meeting_checks']}, memory [MB]: {memory_used_mb():.2f}, state_vs_state: {stats['state_vs_state_meeting_checks']}, state_vs_prefix: {stats['state_vs_prefix_meeting_checks']}, prefix_vs_prefix: {stats['prefix_vs_prefix_meeting_checks']}")
        
        # Checks
        if state_F.violate_constraint(state_B):
            stats["violations_per_g"][state_F.g] += 1
            return False, None

        if state_F.g == g_upper_cutoff_F and state_B.g == g_upper_cutoff_B:
            if state_F.head != state_B.head:
                stats["violations_per_g"][state_F.g] += 1
                return False, None
            stats["must_checks"] += 1
            half_coil_to_check = args.cube_first_dims_path + state_F.materialize_path() + state_B.materialize_path()[::-1][1:]
            is_sym_coil, sym_coil = is_half_of_symmetric_double_coil(half_coil_to_check, args.size_of_graphs[0])
            if is_sym_coil:
                logger(f"SYM_COIL_FOUND! {sym_coil}")
            return is_sym_coil, sym_coil
        else:
            if state_F.head == state_B.head:
                stats["violations_per_g"][state_F.g] += 1
                return False, None
            if graph.has_edge(state_F.head, state_B.head): # ADD THIS WHEN EXPANDING ONE FRONTIER AT A TIME:  and state_F.g != g_upper_cutoff_F - 1 and state_B.g != g_upper_cutoff_B - 1:
                stats["violations_per_g"][state_F.g] += 1
                return False, None
            if state_F.illegal & (1 << state_B.head) or state_B.illegal & (1 << state_F.head):
                stats["violations_per_g"][state_F.g] += 1
                return False, None
            if args.heuristic is not None and args.heuristic != "heuristic0":
                h = heuristic(state_F, state_B, args.heuristic, snake, args)
                if h == 0: # h < abs(g_upper_cutoff_F - state_F.g - (g_upper_cutoff_B - state_B.g)):
                    stats["violations_per_g"][state_F.g] += 1
                    return False, None
            
            # Expand both frontiers together
            stats["expansions"] += 2
            state_F_successors = state_F.generate_successors(args, snake, True)
            state_B_successors = state_B.generate_successors(args, snake, False)
            stats["generated"]['F'] += len(state_F_successors)
            stats["generated"]['B'] += len(state_B_successors)
            stats["num_of_states_per_g"]['F'][state_F.g+1] += len(state_F_successors)
            stats["num_of_states_per_g"]['B'][state_B.g+1] += len(state_B_successors)
            for succ_F in state_F_successors:
                for succ_B in state_B_successors:
                    is_sym_coil, sym_coil = exp_n_check_states(succ_F, succ_B)
                    if is_sym_coil:
                        return True, sym_coil
                    
            # Expand one frontier (shorter one)
            # stats["expansions"] += 1
            # shorter_state = state_F if state_F.g < state_B.g else state_B
            # shorter_state_successors = shorter_state.generate_successors(args, snake, shorter_state is state_F)
            # for succ in shorter_state_successors:
            #     is_sym_coil, sym_coil = exp_n_check_states(succ if shorter_state is state_F else state_F, state_B if shorter_state is state_F else succ)
            #     if is_sym_coil:
            #         return True, sym_coil
            
            return False, None

    best_path_found, best_path = exp_n_check_states(initial_state_F, initial_state_B)
    
    return best_path, stats
