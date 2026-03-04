from heuristics.heuristic import heuristic
from models.state import State
from utils.utils import *
import math
import networkx as nx

def XDFS_CIB_qp(graph, start, goal, heuristic_name, snake, args):
    # In this context, goal is mp. qp is the quarter point.
    mp = args.solution_vertices[0] if hasattr(args, 'solution_vertices') and args.solution_vertices else goal
    qp = args.solution_vertices[1]
    logger = args.logger 
    cube = args.graph_type == "cube"
    
    if not cube or not args.sym_coil:
        logger("Error: XDFS_CIB_qp is only for cube graphs.")
        raise ValueError("XDFS_CIB_qp is only for cube graphs.")
        
    c_star = longest_sym_coil_lengths[args.size_of_graphs[0]]
    max_g = (c_star // 2) - args.cube_first_dims  # This is the total depth to reach mp
    g_upper_cutoff = max_g // 2                   # This is the exact depth qp must be reached
    
    shortest_path_length_start_goal = nx.shortest_path_length(graph, source=start, target=goal)
    if (shortest_path_length_start_goal % 2) != (max_g % 2):
        logger(f"start ({start}) and goal ({goal}) are at odd distance {shortest_path_length_start_goal} in Q_{args.size_of_graphs[0]}, so a path of length {max_g} doesn't exist between them, so no symmetric coil of length {c_star} exists.")
        return None, None
        
    violation_reasons = {
        "wrong_distance_from_qp": 0,
        "dont_meet_at_qp": 0,
        "wrong_distance_from_mp": 0,
        "dont_meet_at_mp": 0,
        "heuristic": 0,
        "no_successors": 0,
    }
    
    stats = {
        "expansions": 0,
        "generated": 0,
        "valid_meeting_checks": 0,
        "num_of_states_per_g": {g: 0 for g in range(0, max_g + 1)},
        "violations": {reason: {g: 0 for g in range(0, max_g + 1)} for reason in violation_reasons.keys()},
        "calc_h_time": 0,
        "must_checks": 0,
        "max_g": max_g,
        "g_upper_cutoff": g_upper_cutoff,
    }

    # Initial state
    graph_init = graph.copy()
    # We do NOT remove goal/mp from the graph because the unidirectional search must step on it at max_g.
    initial_state = State(graph_init, [start], [], snake, args) if isinstance(start, int) else State(graph_init, start, [], snake, args)
    
    # Compute a bitmask of the globally illegal vertices
    illegal = 0
    for i in range(2**args.size_of_graphs[0]):
        if i not in graph:
            illegal |= (1 << i)
    initial_state.illegal |= illegal

    # Graph for heuristic calculations
    h_graph = graph.copy()
    h_graph.remove_nodes_from([v for v in h_graph.nodes if ((initial_state.illegal >> v) & 1)])

    ############################################
    # Main Search Loop
    ############################################

    def exp_n_check_states(state, h_graph):
        g = state.g
        if g > max_g:
            logger("In check_states: g cannot be greater than max_g", error=True)

        stats["valid_meeting_checks"] += 1
        if stats["valid_meeting_checks"] % 250_000 == 0:
            logger(f"Exp: {stats['expansions']}, validity checks: {stats['valid_meeting_checks']}. heuristic violations so far: {sum(stats['violations']['heuristic'].values())}")
        
        # ----------------------------------------
        # 1. qp Checks (First half of the search)
        # ----------------------------------------
        if g < g_upper_cutoff:
            diff_qp = (state.head ^ qp).bit_count()
            edges_left_until_qp = g_upper_cutoff - g
            
            if edges_left_until_qp >= 4:
                if diff_qp > edges_left_until_qp:
                    stats["violations"]["wrong_distance_from_qp"][g] += 1
                    return False, None
            else:
                if diff_qp != edges_left_until_qp:
                    stats["violations"]["wrong_distance_from_qp"][g] += 1
                    return False, None
            if edges_left_until_qp > 1 and diff_qp <= 1:
                stats["violations"]["wrong_distance_from_qp"][g] += 1
                return False, None

        elif g == g_upper_cutoff:
            if state.head != qp:
                stats["violations"]["dont_meet_at_qp"][g] += 1
                return False, None

        # ----------------------------------------
        # 2. mp Checks (Second half of the search)
        # ----------------------------------------
        if g_upper_cutoff <= g < max_g:
            diff_mp = (state.head ^ mp).bit_count()
            edges_left_until_mp = max_g - g
            
            if edges_left_until_mp >= 4:
                if diff_mp > edges_left_until_mp:
                    stats["violations"]["wrong_distance_from_mp"][g] += 1
                    return False, None
            else:
                if diff_mp != edges_left_until_mp:
                    stats["violations"]["wrong_distance_from_mp"][g] += 1
                    return False, None
            if edges_left_until_mp > 1 and diff_mp <= 1:
                stats["violations"]["wrong_distance_from_mp"][g] += 1
                return False, None

        # ----------------------------------------
        # 3. Success / Final Meet Check
        # ----------------------------------------
        if g == max_g:
            if state.head != mp:
                stats["violations"]["dont_meet_at_mp"][g] += 1
                return False, None
                
            stats["must_checks"] += 1
            # For unidirectional, the state's path is the complete half-coil!
            half_coil_to_check = args.cube_first_dims_path + state.materialize_path()
            is_sym_coil, sym_coil = is_half_of_symmetric_double_coil(half_coil_to_check, args.size_of_graphs[0])
            if is_sym_coil:
                logger(f"SYM_COIL_FOUND! {sym_coil}")
            return is_sym_coil, sym_coil

        # ----------------------------------------
        # 4. Heuristic Evaluation
        # ----------------------------------------
        if args.heuristic is not None and args.heuristic != "heuristic0":
            h_graph_updated = h_graph.copy()
            nodes_to_remove = [v for v in h_graph_updated.nodes if ((state.illegal >> v) & 1)]
            h_graph_updated.remove_nodes_from(nodes_to_remove)
            
            # Note: You may need to adapt your heuristic function signature in heuristic.py 
            # to accept an integer 'mp' instead of a 'state_B' object for unidirectional mode.
            h = heuristic(state, mp, args.heuristic, snake, args, h_graph_updated)
            
            if h < max_g - state.g:
                stats["violations"]["heuristic"][g] += 1
                return False, None
        
        # ----------------------------------------
        # 5. Expansion
        # ----------------------------------------
        stats["expansions"] += 1
        successors = state.generate_successors(args, snake, True)
        
        stats["generated"] += len(successors)
        stats["num_of_states_per_g"][g+1] += len(successors)
        
        if len(successors) == 0:
            stats["violations"]["no_successors"][g] += 1
            return False, None
            
        for succ in successors:
            is_sym_coil, sym_coil = exp_n_check_states(succ, h_graph)
            if is_sym_coil:
                return True, sym_coil
                
        return False, None

    best_path_found, best_path = exp_n_check_states(initial_state, h_graph)
    
    return best_path, stats