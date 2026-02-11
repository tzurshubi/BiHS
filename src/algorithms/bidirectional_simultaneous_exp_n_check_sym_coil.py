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
    graph_F.remove_nodes_from([goal] + list(graph.neighbors(goal)))
    graph_B.remove_nodes_from([start] + list(graph.neighbors(start)))
    initial_state_F = State(graph_F, [start], [], snake, args) if isinstance(start, int) else State(graph_F, start, [], snake, args)
    initial_state_B = State(graph_B, [goal], [], snake, args) if isinstance(goal, int) else State(graph_B, goal, [], snake, args)
    # OPENvOPEN = Openvopen(graph, start, goal) if args.prefix_set is None else Openvopen_prefixSet(graph, start, goal, args.prefix_set)
    # OPENvOPEN = Openvopen(graph, start, goal) if args.prefix_set is None else Openvopen_illegalVerts(graph, start, goal, args.prefix_set)

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
        if stats["valid_meeting_checks"] % 1_000_000 == 0:
            logger(f"Valid meeting checks so far: {stats['valid_meeting_checks']}, state_vs_state: {stats['state_vs_state_meeting_checks']}, state_vs_prefix: {stats['state_vs_prefix_meeting_checks']}, prefix_vs_prefix: {stats['prefix_vs_prefix_meeting_checks']}")
            # logger(f"Valid meeting checks so far: {stats['valid_meeting_checks']}, memory [MB]: {memory_used_mb():.2f}, state_vs_state: {stats['state_vs_state_meeting_checks']}, state_vs_prefix: {stats['state_vs_prefix_meeting_checks']}, prefix_vs_prefix: {stats['prefix_vs_prefix_meeting_checks']}")
        
        # Checks
        if state_F.violate_constraint(state_B):
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
            stats["expansions"] += 2
            state_F_successors = state_F.generate_successors(args, snake, True)
            state_B_successors = state_B.generate_successors(args, snake, False)
            stats["generated"]['F'] += len(state_F_successors)
            stats["generated"]['B'] += len(state_B_successors)
            for succ_F in state_F_successors:
                for succ_B in state_B_successors:
                    exp_n_check_states(succ_F, succ_B)


    exp_n_check_states(initial_state_F, initial_state_B)
    
    return best_path, stats
