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

def XMD_DFBnB(graph, start, goal, heuristic_name, snake, args):
    v = args.solution_vertices[1]
    if not isinstance(start, int):
        start = start[0]
    if not isinstance(goal, int):
        goal = goal[0]
    logger = args.logger 
    cube = args.graph_type == "cube"
    if not cube or not args.sym_coil:
        logger("Error: XMD_DFBnB is only for cube graphs.")
        raise ValueError("XMD_DFBnB is only for cube graphs.")
    buffer_dim = args.cube_buffer_dim if cube else None
    c_star = longest_sym_coil_lengths[args.size_of_graphs[0]]
    half_coil_upper_bound = (c_star / 2) - args.cube_first_dims
    quarter_coil_upper_bound = half_coil_upper_bound / 2
    g_upper_cutoff = quarter_coil_upper_bound // 2
    best_path = None
    # Initial checks if s-v-t path of appropriate parity exists, which is necessary for a symmetric coil of length c_star to exist
    shortest_path_length_start_goal = nx.shortest_path_length(graph, source=start, target=goal)
    if (shortest_path_length_start_goal % 2) != (half_coil_upper_bound % 2):
        logger(f"start ({start}) and goal ({goal}) are at odd distance {shortest_path_length_start_goal} in Q_{args.size_of_graphs[0]}, so a path of length {half_coil_upper_bound} doesn't exist between them, so no symmetric coil of length {c_star} exists.")
        return None, None
    shortest_path_length_start_v = nx.shortest_path_length(graph, source=start, target=v)
    if (shortest_path_length_start_v % 2) != (quarter_coil_upper_bound % 2):
        logger(f"start ({start}) and v ({v}) are at odd distance {shortest_path_length_start_v} in Q_{args.size_of_graphs[0]}, so a path of length {half_coil_upper_bound} doesn't exist between them, so no symmetric coil of length {c_star} exists.")
        return None, None
    shortest_path_length_v_goal = nx.shortest_path_length(graph, source=v, target=goal)
    if (shortest_path_length_v_goal % 2) != (quarter_coil_upper_bound % 2):
        logger(f"v ({v}) and goal ({goal}) are at odd distance {shortest_path_length_v_goal} in Q_{args.size_of_graphs[0]}, so a path of length {half_coil_upper_bound} doesn't exist between them, so no symmetric coil of length {c_star} exists.")
        return None, None

    stats = {
        "expansions": 0,
        "generated": {'F': 0, 'B': 0, 'VF': 0, 'VB': 0},
        "symmetric_states_removed": 0,
        "dominated_states_removed": 0,
        "valid_meeting_checks": 0,
        "state_vs_state_meeting_checks": 0,
        "state_vs_prefix_meeting_checks": 0,
        "prefix_vs_prefix_meeting_checks": 0,
        "num_of_states_per_g": {
            'F': {g: 0 for g in range(0, math.ceil(half_coil_upper_bound))},
            'B': {g: 0 for g in range(0, math.ceil(half_coil_upper_bound))},
            'VF': {g: 0 for g in range(0, math.ceil(half_coil_upper_bound))},
            'VB': {g: 0 for g in range(0, math.ceil(half_coil_upper_bound))},
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
        "g_upper_cutoff": g_upper_cutoff,
    }

    
    # Initial states
    graph_F, graph_B, graph_VF, graph_VB = graph.copy(), graph.copy(), graph.copy(), graph.copy()
    graph_F.remove_nodes_from([goal] + list(graph.neighbors(goal)) + [v] + list(graph.neighbors(v)))
    graph_B.remove_nodes_from([start] + list(graph.neighbors(start)) + [v] + list(graph.neighbors(v)))
    graph_VF.remove_nodes_from([start] + list(graph.neighbors(start)) + [goal] + list(graph.neighbors(goal)))
    graph_VB.remove_nodes_from([start] + list(graph.neighbors(start)) + [goal] + list(graph.neighbors(goal)))
    initial_state_F = State(graph_F, [start], [], snake, args)
    initial_state_B = State(graph_B, [goal], [], snake, args)
    initial_state_VF = State(graph_VF, [v], [], snake, args)
    initial_state_VB = State(graph_VB, [v], [], snake, args)
    
    # we compute a bitmask of the illegal vertices, which we removed from the graph earlier. 
    # This is used in heuristic
    illegal = 0
    for v in range(2**args.size_of_graphs[0]):
        if v not in graph:
            illegal |= (1 << v)
    initial_state_F.illegal = illegal
    initial_state_B.illegal = illegal
    initial_state_VF.illegal = illegal
    initial_state_VB.illegal = illegal

    ############################################
    # Main Search Loop
    ############################################

    def exp_n_check_states(state_F, state_B, state_VF, state_VB):
        # Errors (relevant only if the two frontier advance together, and state_F.g == state_B.g at all times, which is not necessarily the case)
        g = state_F.g
        if state_B.g != g or state_VF.g != g or state_VB.g != g:
            raise ValueError("In check_states: all states must have the same g value")
        if g > g_upper_cutoff:
            raise ValueError("In check_states: g cannot be greater than g_upper_cutoff")

        # Logs
        stats["valid_meeting_checks"] += 1
        if g == g_upper_cutoff and state_B.g == g_upper_cutoff: stats["state_vs_state_meeting_checks"] += 1
        elif g < g_upper_cutoff and state_B.g < g_upper_cutoff: stats["prefix_vs_prefix_meeting_checks"] += 1
        else: stats["state_vs_prefix_meeting_checks"] += 1
        if stats["valid_meeting_checks"] % 10_000 == 0:
            logger(f"Valid meeting checks so far: {stats['valid_meeting_checks']}, state_vs_state: {stats['state_vs_state_meeting_checks']}, state_vs_prefix: {stats['state_vs_prefix_meeting_checks']}, prefix_vs_prefix: {stats['prefix_vs_prefix_meeting_checks']}")
            # logger(f"Valid meeting checks so far: {stats['valid_meeting_checks']}, memory [MB]: {memory_used_mb():.2f}, state_vs_state: {stats['state_vs_state_meeting_checks']}, state_vs_prefix: {stats['state_vs_prefix_meeting_checks']}, prefix_vs_prefix: {stats['prefix_vs_prefix_meeting_checks']}")
        
        # Checks - NO NEED TO CHECK VIOLATIONS BECAUSE WE CHECK THEM DURING EXPANSION, AND WE ONLY EXPAND VALID STATES, SO ANY MEETING OF TWO VALID STATES MUST BE VALID.
        # if state_F.violate_constraint(state_B) or \
        #     state_F.violate_constraint(state_VF) or \
        #         state_F.violate_constraint(state_VB) or \
        #             state_B.violate_constraint(state_VF) or \
        #                 state_B.violate_constraint(state_VB) or \
        #                     state_VF.violate_constraint(state_VB):
        #     stats["violations_per_g"][g] += 1
        #     return False, None

        if g == g_upper_cutoff:
            if state_F.head != state_VB.head or state_VF.head != state_B.head:
                stats["violations_per_g"][g] += 1
                return False, None
            stats["must_checks"] += 1
            half_coil_to_check = args.cube_first_dims_path + state_F.materialize_path() + state_B.materialize_path()[::-1][1:]
            is_sym_coil, sym_coil = is_half_of_symmetric_double_coil(half_coil_to_check, args.size_of_graphs[0])
            if is_sym_coil:
                logger(f"SYM_COIL_FOUND! {sym_coil}")
            return is_sym_coil, sym_coil
        else:
            if state_F.head == state_B.head or state_F.head == state_VF.head or state_F.head == state_VB.head or \
                state_B.head == state_VF.head or state_B.head == state_VB.head or \
                     (state_VF.head == state_VB.head and g > 0):
                stats["violations_per_g"][g] += 1
                return False, None
            if graph.has_edge(state_F.head, state_B.head) or graph.has_edge(state_F.head, state_VF.head) or \
                graph.has_edge(state_F.head, state_VB.head) or graph.has_edge(state_B.head, state_VF.head) or \
                    graph.has_edge(state_B.head, state_VB.head) or graph.has_edge(state_VF.head, state_VB.head): # ADD THIS WHEN EXPANDING ONE FRONTIER AT A TIME:  and state_F.g != g_upper_cutoff_F - 1 and state_B.g != g_upper_cutoff_B - 1:
                stats["violations_per_g"][g] += 1
                return False, None
            if state_F.illegal & (1 << state_B.head) or state_F.illegal & (1 << state_VF.head) or \
                state_F.illegal & (1 << state_VB.head) or state_B.illegal & (1 << state_VF.head) or \
                    state_B.illegal & (1 << state_VB.head) or state_VF.illegal & (1 << state_VB.head):
                stats["violations_per_g"][g] += 1
                return False, None
            if args.heuristic is not None and args.heuristic != "heuristic0":
                h = heuristic(state_F, state_B, args.heuristic, snake, args)
                if h == 0: # h < abs(g_upper_cutoff_F - state_F.g - (g_upper_cutoff_B - state_B.g)):
                    stats["violations_per_g"][g] += 1
                    return False, None
            
            # Expand all 4 frontiers together
            stats["expansions"] += 4
            state_F_successors = state_F.generate_successors(args, snake, True)
            state_B_successors = state_B.generate_successors(args, snake, False)
            state_VF_successors = state_VF.generate_successors(args, snake, True)
            state_VB_successors = state_VB.generate_successors(args, snake, False)
            stats["generated"]['F'] += len(state_F_successors)
            stats["generated"]['B'] += len(state_B_successors)
            stats["generated"]['VF'] += len(state_VF_successors)
            stats["generated"]['VB'] += len(state_VB_successors)
            stats["num_of_states_per_g"]['F'][g+1] += len(state_F_successors)
            stats["num_of_states_per_g"]['B'][g+1] += len(state_B_successors)
            stats["num_of_states_per_g"]['VF'][g+1] += len(state_VF_successors)
            stats["num_of_states_per_g"]['VB'][g+1] += len(state_VB_successors)
            for succ_F in state_F_successors:
                for succ_B in state_B_successors:
                    for succ_VF in state_VF_successors:
                        for succ_VB in state_VB_successors:
                            is_sym_coil, sym_coil = exp_n_check_states(succ_F, succ_B, succ_VF, succ_VB)
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

    best_path_found, best_path = exp_n_check_states(initial_state_F, initial_state_B, initial_state_VF, initial_state_VB)
    
    return best_path, stats