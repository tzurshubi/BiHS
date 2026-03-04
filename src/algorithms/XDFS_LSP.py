from os import stat
import math
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

def XDFS_LSP(graph, start, goal, heuristic_name, snake, args):
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
    }
    
    # Initial state
    initial_state = State(graph, [start], [], snake, args) if isinstance(start, int) else State(graph, start, [], snake, args)

    # Track the global best path across the entire DFS tree for true B&B pruning
    global_longest_path = []

    ############################################
    # Main Search Loop
    ############################################

    def exp_n_check_states(state, h_graph):
        nonlocal global_longest_path
        
        # Log
        stats["valid_meeting_checks"] += 1
        if stats["valid_meeting_checks"] % 200_000 == 0:
            logger(f"Valid states checked so far: {stats['valid_meeting_checks']}, Expansions: {stats['expansions']}")

        # Base Case: Reached the goal
        if state.head == goal:
            current_path = state.materialize_path()
            if len(current_path) > len(global_longest_path):
                global_longest_path = current_path
            return current_path, stats
        
        # Prepare for expansion
        h_graph_for_succ = h_graph.copy()
        h_graph_for_succ.remove_nodes_from([state.head])
        
        successors = state.generate_successors(args, snake, True)
        
        stats["expansions"] += 1
        stats["generated"] += len(successors)
        stats["num_of_states_per_g"][state.g + 1] += len(successors)

        if not successors:
            stats["violations"]["no_successors"][state.g] += 1
            return [], stats

        # Calculate heuristic and sort (descending order to find long paths first)
        # Note: Your heuristic function must be able to handle an integer 'goal' instead of a state_B object
        if heuristic_name:
            successors_with_h = [(heuristic(succ, goal, heuristic_name, snake, args, h_graph_for_succ), succ) for succ in successors]
            successors_with_h.sort(key=lambda item: item[0], reverse=True)
        else:
            successors_with_h = [(V, succ) for succ in successors]
                
        for h_val, succ in successors_with_h:
            # DFBnB Pruning:
            # succ.g is the number of edges taken so far. h_val is the max possible remaining edges.
            # If the best possible path through this child cannot beat our global best, prune it!
            best_edges_found_so_far = max(len(global_longest_path) - 1, 0)
            
            if succ.g + h_val <= best_edges_found_so_far: 
                stats["violations"]["heuristic"][state.g] += 1
                break # Because it's sorted descending, all subsequent children will also fail this check
                
            exp_n_check_states(succ, h_graph_for_succ)
                
        return global_longest_path, stats
        
    h_graph = graph.copy()
    
    # Start the recursive search
    exp_n_check_states(initial_state, h_graph)
    
    # In unidirectional, the meet point is simply the goal node
    return global_longest_path, stats #, goal