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

    def evaluate_state(state):
        """ Evaluates intermediate lookahead states against the goal. """
        nonlocal global_longest_path
        
        stats["valid_meeting_checks"] += 1
        if stats["valid_meeting_checks"] % 200_000 == 0:
            logger(f"Valid states checked so far: {stats['valid_meeting_checks']}, Expansions: {stats['expansions']}, Global best: {len(global_longest_path)}")

        # Base Case: Reached the exact goal
        if state.head == goal:
            if state.g > len(global_longest_path) - 1:  # Found a better path than current global best
                global_longest_path = state.materialize_path()
                if args.graph_type == "cube": logger(f"Expansion {stats['expansions']}: New longest path found with length {len(global_longest_path) - 1}: {global_longest_path}")
            return True, False # Reached the goal, stop exploring this specific branch

        # Reached adjacent to goal
        elif graph.has_edge(state.head, goal) and state.g + 1 > len(global_longest_path) - 1:
            if is_vertex_in_bitmap(goal, state.illegal): 
                return False, False 
            global_longest_path = state.materialize_path() + [goal]
            if args.graph_type == "cube": logger(f"Expansion {stats['expansions']}: New longest path found with length {len(global_longest_path) - 1}: {global_longest_path}")
            if snake:
                return True, False # For snake, treat adjacent as a valid solution, but do not continue expanding
            return True, True # Found better path via adjacent edge, but continue expanding as per original logic
        
        return True, True # Valid intermediate state, continue expanding


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

        # Graph copy hoisted outside the loop for performance
        next_h_graph = cur_h_graph.copy()
        if cur_state.head in next_h_graph: next_h_graph.remove_node(cur_state.head)

        all_leaves = []
        for succ in succs:
            is_valid, should_continue = evaluate_state(succ)
            
            if not is_valid: 
                continue 
                
            if not should_continue:
                # The state hit the exact goal. Treat it as a completed leaf.
                all_leaves.append((0, succ, next_h_graph))
                continue

            if remaining == 1:
                # Fast-path for the final lookahead layer
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
        nonlocal global_longest_path
        
        stats["expansions"] += 1
        
        # Retrieve lookahead leaves (or immediate successors if args.lookahead == 1)
        leaves = get_lookahead_successors(state, h_graph, args.lookahead)
        
        if not leaves:
            stats["violations"]["no_successors"][state.g] += 1
            return
            
        leaves.sort(key=lambda item: item[0], reverse=True)
                
        for h_val, leaf, leaf_h_graph in leaves:
            leaf.h = h_val
            
            if args.bsd:
                state_key = (leaf.head, leaf.path_vertices_and_neighbors if snake else leaf.path_vertices)
                if state_key in FNV and FNV[state_key] >= leaf.g:
                    stats["symmetric_states_removed"] += 1
                    continue
            
            # DFBnB Pruning
            if leaf.g + h_val <= len(global_longest_path) - 1: 
                stats["violations"]["heuristic"][state.g] += 1
                break 
                
            # Update BSD tracker with the new longest arrival to this footprint
            if args.bsd: FNV[state_key] = leaf.g
                
            exp_n_check_states(leaf, leaf_h_graph)
                
    h_graph = graph.copy()
    
    # Initialize the search by checking the starting position
    is_valid, should_continue = evaluate_state(initial_state)
    if is_valid and should_continue:
        exp_n_check_states(initial_state, h_graph)
        
    return global_longest_path, stats