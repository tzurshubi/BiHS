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

def BiXDFBnB_F2E(graph, start, goal, heuristic_name, snake, args):
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

    if args.bsd:
        double_state_key = (initial_state_F.head, initial_state_F.path_vertices_and_neighbors if snake else initial_state_F.path_vertices, initial_state_B.head, initial_state_B.path_vertices_and_neighbors if snake else initial_state_B.path_vertices)
        FNV = {double_state_key: initial_state_F.g + initial_state_B.g}

    # Global bound
    global_longest_path = []
    global_meet_point = None

    ############################################
    # Main Search Loop
    ############################################

    def exp_n_check_states(state_F, state_B, h_graph_F, h_graph_B):
        nonlocal global_longest_path, global_meet_point
        
        # Log
        # print(f"Exp: {stats['expansions']}. state_F: {state_F.materialize_path()}, state_B: {state_B.materialize_path()}")
        stats["valid_meeting_checks"] += 1
        if state_F.head == state_B.head: stats["state_vs_state_meeting_checks"] += 1
        else: stats["prefix_vs_prefix_meeting_checks"] += 1
        if stats["valid_meeting_checks"] % 200_000 == 0:
            logger(f"Valid meeting checks so far: {stats['valid_meeting_checks']}, state_vs_state: {stats['state_vs_state_meeting_checks']}, state_vs_prefix: {stats['state_vs_prefix_meeting_checks']}, prefix_vs_prefix: {stats['prefix_vs_prefix_meeting_checks']}")
        
        # Check Meeting
        if state_F.head == state_B.head:
            # We found a meeting, check if it's better than global bound
            if state_F.g + state_B.g > len(global_longest_path) - 1:
                global_longest_path = state_F.materialize_path() + state_B.materialize_path()[::-1][1:]
                global_meet_point = state_F.head
                # args.logger(f"New longest path found with length {len(global_longest_path) - 1} at expansion {stats['expansions']}")
            return global_longest_path, state_F.head
        elif is_vertex_in_bitmap(state_F.head, state_B.illegal) or is_vertex_in_bitmap(state_B.head, state_F.illegal):
            # paths are intersecting at an illegal vertex, prune
             stats["violations"]["intersection"][state_F.g] += 1
             return [], None
        elif graph.has_edge(state_F.head, state_B.head):
            # We found a meeting via adjacent heads, check if it's better than global bound
            if state_F.g + 1 + state_B.g > len(global_longest_path) - 1:
                global_longest_path = state_F.materialize_path() + [state_B.head] + state_B.materialize_path()[::-1][1:]
                global_meet_point = state_B.head
                args.logger(f"New longest path found with length {len(global_longest_path) - 1} at expansion {stats['expansions']}")
            if snake: return global_longest_path, global_meet_point
        
        
        # Expand - both frontiers together
        state_F_successors = state_F.generate_successors(args, snake, True)
        state_B_successors = state_B.generate_successors(args, snake, False)
        h_graph_for_succ_F = h_graph_F.copy()
        h_graph_for_succ_F.remove_nodes_from([state_F.head])
        h_graph_for_succ_B = h_graph_B.copy()
        h_graph_for_succ_B.remove_nodes_from([state_B.head])
        successors_with_h = []
        # Calculate F2E heuristic for all successors
        
        for succ_F in state_F_successors:
            # if succ_F.materialize_path() == [0,6,12,13,7,1,2,8]:
            #     print("Found the path in F succs!")
            if heuristic_name: succ_F.h = heuristic(succ_F, goal, heuristic_name, snake, args, h_graph_for_succ_F.copy() if snake else h_graph_for_succ_F)
            else: succ_F.h = V
        for succ_B in state_B_successors:
            if heuristic_name: succ_B.h = heuristic(succ_B, start, heuristic_name, snake, args, h_graph_for_succ_B.copy() if snake else h_graph_for_succ_B)
            else: succ_B.h = V

        for succ_F in state_F_successors:
            for succ_B in state_B_successors:
                h_val = min(succ_F.h ,succ_B.h)
                successors_with_h.append((h_val, succ_F, succ_B))
        successors_with_h.sort(key=lambda item: item[0], reverse=True)

        stats["expansions"] += 1
        stats["generated"]['F'] += len(state_F_successors)
        stats["generated"]['B'] += len(state_B_successors)
        stats["num_of_states_per_g"]['F'][state_F.g+1] += len(state_F_successors)
        stats["num_of_states_per_g"]['B'][state_B.g+1] += len(state_B_successors)

        for h_val, succ_F, succ_B in successors_with_h:
            if args.bsd:
                double_state_key = (succ_F.head, succ_F.path_vertices_and_neighbors if snake else succ_F.path_vertices, succ_B.head, succ_B.path_vertices_and_neighbors if snake else succ_B.path_vertices)
                if double_state_key in FNV and FNV[double_state_key] >= succ_F.g + succ_B.g:
                    stats["symmetric_states_removed"] += 1
                    continue
            
            # DFBnB Pruning
            if succ_F.g + h_val <= len(global_longest_path) - 1: # succ_F.g = succ_B.g
                stats["violations"]["heuristic"][state_F.g] += 1
                break # Prune this and all subsequent sorted successors

            if args.bsd: FNV[double_state_key] = succ_F.g + succ_B.g

            # pass the h_graph for the next expansions, with the current heads removed
            exp_n_check_states(succ_F, succ_B, h_graph_for_succ_F, h_graph_for_succ_B)
                
    h_graph = graph.copy()
    exp_n_check_states(initial_state_F, initial_state_B, h_graph, h_graph)
    
    # Return the global best found after the whole DFS tree is resolved
    return global_longest_path, stats, global_meet_point