from os import stat
import math
import queue
from collections import defaultdict, deque
import heapq, time
from heuristics.heuristic import heuristic
from models.state import State
from utils.utils import *

def XIDA(graph, start, goal, heuristic_name, snake, args):
    logger = args.logger 
    N = max(graph.nodes)
    V = len(graph.nodes)

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

    global_longest_path = []

    # --- IDA* Initialization ---
    initial_h = V
    if heuristic_name:
        initial_h = heuristic(initial_state, goal, heuristic_name, snake, args, graph.copy() if snake else graph)
    
    threshold = initial_state.g + initial_h
    next_threshold = -1
    target_found = False

    def evaluate_state(state):
        """ Evaluates intermediate lookahead states against the goal. """
        nonlocal global_longest_path, target_found
        
        if target_found: return False, False # Short-circuit if optimal is already proven
        
        stats["valid_meeting_checks"] += 1
        if stats["valid_meeting_checks"] % 200_000 == 0:
            logger(f"Valid states checked so far: {stats['valid_meeting_checks']}, Expansions: {stats['expansions']}, Global best: {len(global_longest_path)}")

        # Base Case: Reached the exact goal
        if state.head == goal:
            if state.g > len(global_longest_path) - 1:  
                global_longest_path = state.materialize_path()
                if args.graph_type == "cube": logger(f"Expansion {stats['expansions']}: New longest path found with length {len(global_longest_path) - 1}: {global_longest_path}")
            
            if len(global_longest_path) - 1 >= threshold:
                target_found = True
            return True, False # Reached the goal, stop exploring this specific branch

        # Reached adjacent to goal
        elif graph.has_edge(state.head, goal) and state.g + 1 > len(global_longest_path) - 1:
            if is_vertex_in_bitmap(goal, state.illegal): 
                return False, False 
            global_longest_path = state.materialize_path() + [goal]
            if args.graph_type == "cube": logger(f"Expansion {stats['expansions']}: New longest path found with length {len(global_longest_path) - 1}: {global_longest_path}")
            
            if len(global_longest_path) - 1 >= threshold:
                target_found = True
                
            if snake:
                return True, False 
            return True, True 
        
        return True, True 

    def get_lookahead_successors(cur_state, cur_h_graph, remaining):
        """Recursively advances the frontier down to depth k."""
        if remaining == 0:
            h_val = V
            if heuristic_name:
                h_val = heuristic(cur_state, goal, heuristic_name, snake, args, cur_h_graph.copy() if snake else cur_h_graph)
            return [(h_val, cur_state, cur_h_graph)]
            
        succs = cur_state.generate_successors(args, snake, True)
        
        stats["generated"] += len(succs)
        if len(succs) > 0: stats["num_of_states_per_g"][cur_state.g + 1] += len(succs)

        next_h_graph = cur_h_graph.copy()
        if cur_state.head in next_h_graph: next_h_graph.remove_node(cur_state.head)

        all_leaves = []
        for succ in succs:
            is_valid, should_continue = evaluate_state(succ)
            
            if not is_valid: 
                continue 
                
            if not should_continue:
                all_leaves.append((0, succ, next_h_graph))
                continue

            if remaining == 1:
                h_val = V
                if heuristic_name:
                    h_val = heuristic(succ, goal, heuristic_name, snake, args, next_h_graph.copy() if snake else next_h_graph)
                all_leaves.append((h_val, succ, next_h_graph))
            else:
                all_leaves.extend(get_lookahead_successors(succ, next_h_graph, remaining - 1))
                
        return all_leaves

    ############################################
    # Main Search Loop
    ############################################

    def exp_n_check_states(state, h_graph):
        nonlocal global_longest_path, next_threshold, target_found
        
        if target_found: return
        
        stats["expansions"] += 1
        
        leaves = get_lookahead_successors(state, h_graph, args.lookahead)
        
        if not leaves:
            stats["violations"]["no_successors"][state.g] += 1
            return
            
        leaves.sort(key=lambda item: item[0], reverse=True)
                
        for h_val, leaf, leaf_h_graph in leaves:
            if target_found: break
            
            leaf.h = h_val
            f_val = leaf.g + h_val
            
            if args.bsd:
                state_key = (leaf.head, leaf.path_vertices_and_neighbors if snake else leaf.path_vertices)
                if state_key in FNV and FNV[state_key] >= leaf.g:
                    stats["symmetric_states_removed"] += 1
                    continue
            
            # --- IDA* Pruning ---
            # Prune if the branch's f_val falls below the required threshold
            if f_val < threshold: 
                stats["violations"]["heuristic"][state.g] += 1
                # Track the highest failing f_val to become the next iteration's threshold
                if f_val > next_threshold:
                    next_threshold = f_val
                break # Since leaves are sorted descending, all subsequent leaves will also fail
                
            if args.bsd: FNV[state_key] = leaf.g
                
            exp_n_check_states(leaf, leaf_h_graph)

    # --- IDA* Iterative Loop ---
    while threshold >= 0 and not target_found:
        # logger(f"--- IDA* Iteration Starting With Threshold {next_threshold} ---")
            
        next_threshold = -1
        
        # In IDA*, the closed list MUST be wiped clean at the start of every iteration
        if args.bsd:
            state_key = (initial_state.head, initial_state.path_vertices_and_neighbors if snake else initial_state.path_vertices)
            FNV = {state_key: initial_state.g}
            
        h_graph = graph.copy()
        
        is_valid, should_continue = evaluate_state(initial_state)
        if is_valid and should_continue:
            exp_n_check_states(initial_state, h_graph)
            
        # If we successfully found a path matching the upper bound, halt the search
        if len(global_longest_path) - 1 >= threshold:
            target_found = True
            break
            
        # Lower the threshold for the next depth-first iteration
        threshold = next_threshold
        
    return global_longest_path, stats