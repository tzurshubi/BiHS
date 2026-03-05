from heuristics.heuristic import heuristic
from models.state import State
from utils.utils import *

def BiXDFS_CIB_qp(graph, start, goal, heuristic_name, snake, args):
    qp = args.solution_vertices[1]
    logger = args.logger 
    cube = args.graph_type == "cube"
    if not cube or not args.sym_coil:
        logger("Error: BiXDFS_CIB_qp is only for cube graphs.")
        raise ValueError("BiXDFS_CIB_qp is only for cube graphs.")
    buffer_dim = args.cube_buffer_dim if cube else None
    c_star = longest_sym_coil_lengths[args.size_of_graphs[0]]
    half_coil_upper_bound = (c_star / 2) - args.cube_first_dims
    g_upper_cutoff = half_coil_upper_bound // 2
    best_path = None
    shortest_path_length_start_goal = nx.shortest_path_length(graph, source=start, target=goal)
    if (shortest_path_length_start_goal % 2) != (half_coil_upper_bound % 2):
        logger(f"start ({start}) and goal ({goal}) are at odd distance {shortest_path_length_start_goal} in Q_{args.size_of_graphs[0]}, so a path of length {half_coil_upper_bound} doesn't exist between them, so no symmetric coil of length {c_star} exists.")
        return None, None
    violation_reasons = {
        "wrong_distance_from_qp": 0,
        "dont_meet_at_qp": 0,
        "meet_early": 0,
        "meet_adjacent": 0,
        "illegal_vertex": 0,
        "heuristic": 0,
        "no_successors": 0,
    }
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
            'F': {g: 0 for g in range(0, math.ceil(g_upper_cutoff) + 1)},
            'B': {g: 0 for g in range(0, math.ceil(g_upper_cutoff) + 1)},
        },
        "violations": {reason: {g: 0 for g in range(0, math.ceil(g_upper_cutoff) + 1)} for reason in violation_reasons.keys()},
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
    graph_F, graph_B = graph.copy(), graph.copy()
    graph_F.remove_nodes_from([goal] + list(graph.neighbors(goal)))
    graph_B.remove_nodes_from([start] + list(graph.neighbors(start)))
    initial_state_F = State(graph_F, [start], [], snake, args) if isinstance(start, int) else State(graph_F, start, [], snake, args)
    initial_state_B = State(graph_B, [goal], [], snake, args) if isinstance(goal, int) else State(graph_B, goal, [], snake, args)
    
    # we compute a bitmask of the illegal vertices, which we removed from the graph earlier. 
    # This is used in heuristic
    illegal = 0
    for i in range(2**args.size_of_graphs[0]):
        if i not in graph:
            illegal |= (1 << i)
    initial_state_F.illegal = illegal
    initial_state_B.illegal = illegal

    # FNV_F = {(initial_state_F.head, initial_state_F.path_vertices_and_neighbors)}
    # FNV_B = {(initial_state_B.head, initial_state_B.path_vertices_and_neighbors)}

    # graph for heuristic calculations
    h_graph = graph.copy()
    h_graph.remove_nodes_from([v for v in h_graph.nodes if ((initial_state_F.illegal >> v) & 1) or ((initial_state_B.illegal >> v) & 1)])

    ############################################
    # Main Search Loop
    ############################################

    def exp_n_check_states(state_F, state_B, h_graph):
        # Errors (relevant only if the two frontier advance together, and state_F.g == state_B.g at all times, which is not necessarily the case)
        g = state_F.g
        if state_B.g != g:
            logger("In check_states: all states must have the same g value", error=True)
        if g > g_upper_cutoff:
            logger("In check_states: g cannot be greater than g_upper_cutoff", error=True)

        # Logs
        # print(f"state_F: {state_F.materialize_path()}, state_B: {state_B.materialize_path()}")
        # if stats["valid_meeting_checks"] % 50 == 0:
        #     c=1
        stats["valid_meeting_checks"] += 1 
        if g == g_upper_cutoff: stats["state_vs_state_meeting_checks"] += 1  
        elif g < g_upper_cutoff: stats["prefix_vs_prefix_meeting_checks"] += 1
        else: stats["state_vs_prefix_meeting_checks"] += 1
        if stats["valid_meeting_checks"] % 250_000 == 0:
            logger(f"Exp: {stats['expansions']}, validity checks: {stats['valid_meeting_checks']}, s-s: {stats['state_vs_state_meeting_checks']}, s-p: {stats['state_vs_prefix_meeting_checks']}, p-p: {stats['prefix_vs_prefix_meeting_checks']}. heuristic violations so far: {sum(stats['violations']['heuristic'].values())}")
            # logger(f"Exp: {stats['expansions']}, Mem: {memory_used_mb():.2f} [MB], valid meet checks: {stats['valid_meeting_checks']}, s-s: {stats['state_vs_state_meeting_checks']}, s-p: {stats['state_vs_prefix_meeting_checks']}, p-p: {stats['prefix_vs_prefix_meeting_checks']}. heuristic violations so far: {sum(stats['violations']['heuristic'].values())}")
        
        # Checks - NO NEED TO CHECK VIOLATIONS BECAUSE WE CHECK THEM DURING EXPANSION, AND WE ONLY EXPAND VALID STATES, SO ANY MEETING OF TWO VALID STATES MUST BE VALID.
        # if state_F.violate_constraint(state_B):
        #     stats["violations"][g] += 1
        #     return False, None

        # qp checks
        state_F_head_diff_bits_with_qp = (state_F.head ^ qp).bit_count()
        state_B_head_diff_bits_with_qp = (state_B.head ^ qp).bit_count()
        edges_left_until_qp = g_upper_cutoff - g
        if edges_left_until_qp >= 4:
            if state_F_head_diff_bits_with_qp > edges_left_until_qp or state_B_head_diff_bits_with_qp > edges_left_until_qp:
                # violation: a state's head is too far from qp
                stats["violations"]["wrong_distance_from_qp"][g] += 1
                return False, None
        else:
            if state_F_head_diff_bits_with_qp != edges_left_until_qp or state_B_head_diff_bits_with_qp != edges_left_until_qp:
                # violation: a state's head is not at the right distance from qp
                stats["violations"]["wrong_distance_from_qp"][g] += 1
                return False, None
        if edges_left_until_qp > 1 and (state_F_head_diff_bits_with_qp <= 1 or state_B_head_diff_bits_with_qp <= 1):
            # violation: a state's head is too close to qp
            stats["violations"]["wrong_distance_from_qp"][g] += 1
            return False, None

        if g == g_upper_cutoff:
            if state_F.head != state_B.head or state_F.head != qp or state_B.head != qp:
                # violation: the two states end at different vertices
                stats["violations"]["dont_meet_at_qp"][g] += 1
                return False, None
            stats["must_checks"] += 1
            half_coil_to_check = args.cube_first_dims_path + state_F.materialize_path() + state_B.materialize_path()[::-1][1:]
            is_sym_coil, sym_coil = is_half_of_symmetric_double_coil(half_coil_to_check, args.size_of_graphs[0])
            if is_sym_coil:
                logger(f"SYM_COIL_FOUND! {sym_coil}")
            return is_sym_coil, sym_coil
        else:
            if state_F.head == state_B.head:
                # violation: the two states meet early
                stats["violations"]["meet_early"][g] += 1
                return False, None
            if graph.has_edge(state_F.head, state_B.head): # ADD THIS WHEN EXPANDING ONE FRONTIER AT A TIME:  and state_F.g != g_upper_cutoff - 1 and state_B.g != g_upper_cutoff - 1:
                # violation: the two states's heads are adjacent
                stats["violations"]["meet_adjacent"][g] += 1
                return False, None
            if state_F.illegal & (1 << state_B.head) or state_B.illegal & (1 << state_F.head):
                # violation: one state's head is at an illegal vertex for the other state
                stats["violations"]["illegal_vertex"][g] += 1
                return False, None
            if args.heuristic is not None and args.heuristic != "heuristic0":
                h_graph_updated = h_graph.copy()
                nodes_to_remove = [v for v in h_graph_updated.nodes if \
                    ((state_F.illegal >> v) & 1) or ((state_B.illegal >> v) & 1)]
                # print(f"Removing {len(nodes_to_remove)} nodes from h_graph: {nodes_to_remove}.")
                h_graph_updated.remove_nodes_from(nodes_to_remove)
                h = heuristic(state_F, state_B, args.heuristic, snake, args, h_graph_updated)
                # print(f"h: {h}") # for debug
                if h < g_upper_cutoff - state_F.g + g_upper_cutoff - state_B.g:
                    # violation: heuristic
                    stats["violations"]["heuristic"][g] += 1
                    return False, None
            
            # Expand both frontiers together
            stats["expansions"] += 2
            state_F_successors = state_F.generate_successors(args, snake, True)
            state_B_successors = state_B.generate_successors(args, snake, False)
            # print(f"state_F_successors: {[succ.materialize_path() for succ in state_F_successors]}") # for debug
            # print(f"state_B_successors: {[succ.materialize_path() for succ in state_B_successors]}") # for debug
            stats["generated"]['F'] += len(state_F_successors)
            stats["generated"]['B'] += len(state_B_successors)
            stats["num_of_states_per_g"]['F'][g+1] += len(state_F_successors)
            stats["num_of_states_per_g"]['B'][state_B.g+1] += len(state_B_successors)
            if len(state_F_successors) == 0 or len(state_B_successors) == 0:
                # violation: one frontier cannot advance
                stats["violations"]["no_successors"][g] += 1
                return False, None
            for succ_F in state_F_successors:
                for succ_B in state_B_successors:
                    is_sym_coil, sym_coil = exp_n_check_states(succ_F, succ_B, h_graph)
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

    best_path_found, best_path = exp_n_check_states(initial_state_F, initial_state_B, h_graph)
    
    return best_path, stats
