from os import stat
import queue
import math
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

def BiXDFS_LSP(graph, start, goal, heuristic_name, snake, args):
    logger = args.logger 
    N = max(graph.nodes)
    V = len(graph.nodes)

    violation_reasons = {
        "intersection": 0,
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
            'F': {g: 0 for g in range(0, math.ceil(N/2) + 1)},
            'B': {g: 0 for g in range(0, math.ceil(N/2) + 1)}
        },
        "violations": {reason: {g: 0 for g in range(0, math.ceil(N/2) + 1)} for reason in violation_reasons.keys()},
        "prefix_set_mean_size": {'F': 0, 'B': 0},
        "paths_with_g_upper_cutoff": {'F': 0, 'B': 0},
        "paths_with_g_lower_cutoff": {'F': 0, 'B': 0},
        "valid_meeting_check_time": 0,
        "calc_h_time": 0,
        "moved_OPEN_to_AUXOPEN": 0,
        "g_values": [],
        "BF_values": [],
        "must_checks": 0,
    }

    # Initial states
    initial_state_F = State(graph, [start], [], snake, args) if isinstance(start, int) else State(graph, start, [], snake, args)
    initial_state_B = State(graph, [goal], [], snake, args) if isinstance(goal, int) else State(graph, goal, [], snake, args)

    # Global bound
    global_longest_path = []
    global_meet_point = None

    ############################################
    # Main Search Loop
    ############################################

    def exp_n_check_states(state_F, state_B, h_graph):
        nonlocal global_longest_path, global_meet_point
        
        # Log
        # print(f"state_F: {state_F.materialize_path()}, state_B: {state_B.materialize_path()}")
        stats["valid_meeting_checks"] += 1
        if state_F.head == state_B.head: stats["state_vs_state_meeting_checks"] += 1
        else: stats["prefix_vs_prefix_meeting_checks"] += 1
        if stats["valid_meeting_checks"] % 200_000 == 0:
            logger(f"Valid meeting checks so far: {stats['valid_meeting_checks']}, state_vs_state: {stats['state_vs_state_meeting_checks']}, state_vs_prefix: {stats['state_vs_prefix_meeting_checks']}, prefix_vs_prefix: {stats['prefix_vs_prefix_meeting_checks']}")
        
        # Check Meeting
        if state_F.head == state_B.head:
            current_path = state_F.materialize_path() + state_B.materialize_path()[::-1][1:]
            # Update global bound if we found a better path
            if len(current_path) > len(global_longest_path):
                global_longest_path = current_path
                global_meet_point = state_F.head
            return current_path, state_F.head
            
        elif is_vertex_in_bitmap(state_F.head, state_B.illegal) or is_vertex_in_bitmap(state_B.head, state_F.illegal):
            stats["violations"]["intersection"][state_F.g] += 1
            return [], None
        
        # Expand
        if state_F.g <= state_B.g:
            h_graph_for_succ_F = h_graph.copy()
            h_graph_for_succ_F.remove_nodes_from([state_F.head])
            state_F_successors = state_F.generate_successors(args, snake, True)
            
            if heuristic_name:
                successors_with_h = [(heuristic(succ_F, state_B, heuristic_name, snake, args, h_graph_for_succ_F), succ_F) for succ_F in state_F_successors]
                successors_with_h.sort(key=lambda item: item[0], reverse=True)
            else:
                successors_with_h = [(V, succ_F) for succ_F in state_F_successors]
            
            stats["expansions"] += 1
            stats["generated"]['F'] += len(state_F_successors)
            stats["num_of_states_per_g"]['F'][state_F.g+1] += len(state_F_successors)
            
            for h_val, succ_F in successors_with_h:
                # Compare against global bound
                # print(f"{succ_F.g} + {h_val} + {state_B.g} <= {len(global_longest_path) - 1}")
                if succ_F.g + h_val + state_B.g <= len(global_longest_path) - 1: 
                    stats["violations"]["heuristic"][state_F.g] += 1
                    break # Prune this and all subsequent sorted successors
                    
                exp_n_check_states(succ_F, state_B, h_graph_for_succ_F)
            
        else:
            h_graph_for_succ_B = h_graph.copy()
            h_graph_for_succ_B.remove_nodes_from([state_B.head])
            state_B_successors = state_B.generate_successors(args, snake, False)
            
            if heuristic_name:
                successors_with_h = [(heuristic(state_F, succ_B, heuristic_name, snake, args, h_graph_for_succ_B), succ_B) for succ_B in state_B_successors]
                successors_with_h.sort(key=lambda item: item[0], reverse=True)
            else:
                successors_with_h = [(V, succ_B) for succ_B in state_B_successors]
            
            stats["expansions"] += 1
            stats["generated"]['B'] += len(state_B_successors)
            stats["num_of_states_per_g"]['B'][state_B.g+1] += len(state_B_successors)
            
            for h_val, succ_B in successors_with_h:
                # Compare against global bound
                # print(f"{state_F.g} + {h_val} + {succ_B.g} <= {len(globaly_longest_path) - 1}")
                if state_F.g + h_val + succ_B.g <= len(global_longest_path) - 1: 
                    stats["violations"]["heuristic"][state_B.g] += 1
                    break # Prune this and all subsequent sorted successors
                    
                exp_n_check_states(state_F, succ_B, h_graph_for_succ_B)
        
    h_graph = graph.copy()
    exp_n_check_states(initial_state_F, initial_state_B, h_graph)
    
    # Return the global best found after the whole DFS tree is resolved
    return global_longest_path, stats, global_meet_point