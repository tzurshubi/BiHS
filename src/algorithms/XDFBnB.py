from os import stat
import math
import queue
from collections import defaultdict, deque
import heapq, time
from heuristics.heuristic import heuristic
from models.state import State
from utils.utils import *

def XDFBnB(graph, start, goal, heuristic_name, snake, args):
    logger = args.logger 
    N = max(graph.nodes)
    V = len(graph.nodes)
    g_h_buckets = [[0 for _ in range(V + 1)] for _ in range(V + 1)]

    violation_reasons = {
        "heuristic": 0,
        "no_successors": 0,
    }
    
    stats = {
        "expansions": 0,
        "generated": 0,
        "valid_meeting_checks": 0,
        "num_of_states_per_g": {g: 0 for g in range(0, N + 1)},
        "violations": {reason: {g: 0 for g in range(0, N + 1)} for reason in violation_reasons.keys()},
        "calc_h_time": 0,
        "symmetric_states_removed": 0,
    }
    
    # Initial state
    initial_state = State(graph, [start], [], snake, args) if isinstance(start, int) else State(graph, start, [], snake, args)

    if args.bsd:
        state_key = (initial_state.head, initial_state.path_vertices_and_neighbors if snake else initial_state.path_vertices)
        FNV = {state_key: initial_state.g}

    # Track the global best path across the entire DFS tree for true B&B pruning
    global_longest_path = []

    ############################################
    # Main Search Loop
    ############################################

    def exp_n_check_states(state, h_graph):
        nonlocal global_longest_path
        
        # Log
        # print(f"Expansion {stats['expansions']}: {state.materialize_path()}")

        stats["valid_meeting_checks"] += 1
        if stats["valid_meeting_checks"] % 200_000 == 0:
            logger(f"Valid states checked so far: {stats['valid_meeting_checks']}, Expansions: {stats['expansions']}, Global best: {len(global_longest_path)}")

        # Base Case: Reached the goal
        if state.head == goal:
            if state.g > len(global_longest_path) - 1:  # Found a better path than current global best
                global_longest_path = state.materialize_path()
                logger(f"Expansion {stats['expansions']}: New longest path found with length {len(global_longest_path) - 1}: {global_longest_path}")
                logger(f"g_h_buckets: {matrix_to_sparse_string(g_h_buckets)}")
            return global_longest_path, stats
        elif graph.has_edge(state.head, goal) and state.g + 1 > len(global_longest_path) - 1:
            # Found a better path via direct edge to goal
            global_longest_path = state.materialize_path() + [goal]
            logger(f"Expansion {stats['expansions']}: New longest path found with length {len(global_longest_path) - 1}: {global_longest_path}")
            logger(f"g_h_buckets: {matrix_to_sparse_string(g_h_buckets)}")
        
        # Prepare for expansion
        h_graph_for_succ = h_graph.copy()
        h_graph_for_succ.remove_nodes_from([state.head])
        
        successors = state.generate_successors(args, snake, True)

        if not successors:
            stats["violations"]["no_successors"][state.g] += 1
            return [], stats
        
        stats["expansions"] += 1
        stats["generated"] += len(successors)
        stats["num_of_states_per_g"][state.g + 1] += len(successors)

        if heuristic_name:
            successors_with_h = [(heuristic(succ, goal, heuristic_name, snake, args, h_graph_for_succ.copy() if snake else h_graph_for_succ), succ) for succ in successors]
            successors_with_h.sort(key=lambda item: item[0], reverse=True)
        else:
            successors_with_h = [(V, succ) for succ in successors]
        # logger(f"Expansion {stats['expansions']}: {state.materialize_path()} Generated {len(successors)} successors: {[(h,succ.materialize_path()) for h, succ in successors_with_h]}")
                
        for h_val, succ in successors_with_h:
            succ.h = h_val
            g_h_buckets[succ.g][succ.h] += 1
            if args.bsd:
                state_key = (succ.head, succ.path_vertices_and_neighbors if snake else succ.path_vertices)
                if state_key in FNV and FNV[state_key] >= succ.g:
                    stats["symmetric_states_removed"] += 1
                    continue
            
            # DFBnB Pruning
            if succ.g + h_val <= len(global_longest_path) - 1: 
                stats["violations"]["heuristic"][state.g] += 1
                break 
                
            # Update BSD tracker with the new longest arrival to this footprint
            if args.bsd: FNV[state_key] = succ.g
                
            exp_n_check_states(succ, h_graph_for_succ)
                
        return global_longest_path, stats
        
    h_graph = graph.copy()
    
    # Start the recursive search
    exp_n_check_states(initial_state, h_graph)
    logger(f"Search completed. g_h_buckets: {matrix_to_sparse_string(g_h_buckets)}")
    return global_longest_path, stats