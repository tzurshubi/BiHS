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

def BiXDFBnB(graph, start, goal, heuristic_name, snake, args):
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
            'F': {g: 0 for g in range(0, N + 1)},
            'B': {g: 0 for g in range(0, N + 1)}
        },
        "violations": {reason: {g: 0 for g in range(0, N + 1)} for reason in violation_reasons.keys()},
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

    initial_state_F = State(graph, [start], [], snake, args) if isinstance(start, int) else State(graph, start, [], snake, args)
    initial_state_B = State(graph, [goal], [], snake, args) if isinstance(goal, int) else State(graph, goal, [], snake, args)

    if args.bsd:
        double_state_key = (initial_state_F.head, initial_state_F.path_vertices_and_neighbors if snake else initial_state_F.path_vertices, initial_state_B.head, initial_state_B.path_vertices_and_neighbors if snake else initial_state_B.path_vertices)
        FNV = {double_state_key: initial_state_F.g + initial_state_B.g}

    global_longest_path = []
    global_meet_point = None


    def evaluate_pair(f, b):
        """ Evaluates intermediate lookahead states to stop chord violations and capture meets immediately. """
        nonlocal global_longest_path, global_meet_point
        
        stats["valid_meeting_checks"] += 1
        if f.head == b.head: stats["state_vs_state_meeting_checks"] += 1
        else: stats["prefix_vs_prefix_meeting_checks"] += 1
        
        if stats["expansions"] % 100_000 == 0:
            logger(f"Expansions: {stats['expansions']}. Checks - state_vs_state: {stats['state_vs_state_meeting_checks']}, state_vs_prefix: {stats['state_vs_prefix_meeting_checks']}, prefix_vs_prefix: {stats['prefix_vs_prefix_meeting_checks']}")

        # 1. Check Exact Meet FIRST
        if f.head == b.head:
            if f.g + b.g > len(global_longest_path) - 1:
                global_longest_path = f.materialize_path() + b.materialize_path()[::-1][1:]
                global_meet_point = f.head
                if args.graph_type == "cube": logger(f"Expansion {stats['expansions']}: New longest path found with length {len(global_longest_path) - 1}: {global_longest_path}")
            return True, False # Valid, but met, do not expand

        # 2. Check Overlap / Chord Violations AND Snake Adjacent Meets
        is_f_in_b = is_vertex_in_bitmap(f.head, b.illegal)
        is_b_in_f = is_vertex_in_bitmap(b.head, f.illegal)
        
        if is_f_in_b or is_b_in_f:
            if graph.has_edge(f.head, b.head):
                if snake:
                    # Snake logic: They are adjacent, so they WILL trigger the illegal bitmap.
                    # We must manually verify that there are no chords to the opposite tails.
                    f_path = f.materialize_path()
                    b_path = b.materialize_path()
                    valid_snake_meet = True
                    for v in b_path:
                        if v != b.head and graph.has_edge(f.head, v):
                            valid_snake_meet = False
                            break
                    if valid_snake_meet:
                        for v in f_path:
                            if v != f.head and graph.has_edge(b.head, v):
                                valid_snake_meet = False
                                break
                    
                    if valid_snake_meet:
                        if f.g + 1 + b.g > len(global_longest_path) - 1:
                            global_longest_path = f_path + [b.head] + b_path[::-1][1:]
                            global_meet_point = b.head
                            if args.graph_type == "cube": logger(f"Expansion {stats['expansions']}: New longest path found with length {len(global_longest_path) - 1}: {global_longest_path}")
                        return True, False # Valid snake meet, stop expanding
                    else:
                        stats["violations"]["intersection"][f.g] += 1
                        return False, False # Invalid adjacent meet (has chord to tail)
                else:
                    # LSP logic: If it's adjacent but STILL triggered the illegal bitmap, it hit the tail.
                    stats["violations"]["intersection"][f.g] += 1
                    return False, False
            else:
                # No edge between heads; it is a pure intersection/chord.
                stats["violations"]["intersection"][f.g] += 1
                return False, False

        # 3. Check Adjacent Meet for LSP (snake=False)
        elif graph.has_edge(f.head, b.head):
            if f.g + 1 + b.g > len(global_longest_path) - 1:
                global_longest_path = f.materialize_path() + [b.head] + b.materialize_path()[::-1][1:]
                global_meet_point = b.head
                if args.graph_type == "cube": logger(f"Expansion {stats['expansions']}: New longest path found with length {len(global_longest_path) - 1}: {global_longest_path}")
            return True, True # Valid LSP meet, continue expanding

        return True, True # Valid, continue expanding

    def get_lookahead_successors(cur_F, cur_B, cur_h_graph, remaining):
        """Recursively advances frontiers while strictly enforcing mutual validity."""
        if remaining == 0:
            h_val = V
            if heuristic_name:
                h_val = heuristic(cur_F, cur_B, heuristic_name, snake, args, cur_h_graph.copy() if snake else cur_h_graph)
            return [(h_val, cur_F, cur_B, cur_h_graph)]
        
        if remaining >= 2:
            succs_F = cur_F.generate_successors(args, snake, True)
            succs_B = cur_B.generate_successors(args, snake, False)
            
            stats["generated"]['F'] += len(succs_F)
            stats["generated"]['B'] += len(succs_B)
            if len(succs_F) > 0: stats["num_of_states_per_g"]['F'][cur_F.g+1] += len(succs_F)
            if len(succs_B) > 0: stats["num_of_states_per_g"]['B'][cur_B.g+1] += len(succs_B)

            all_leaves = []
            for f in succs_F:
                for b in succs_B:
                    # Enforce intermediate validity before recursing deeper
                    is_valid, should_continue = evaluate_pair(f, b)
                    if not is_valid or not should_continue:
                        continue

                    # FIX: Remove the OLD heads that are now consumed as tail/body
                    next_h_graph = cur_h_graph.copy()
                    if cur_F.head in next_h_graph: next_h_graph.remove_node(cur_F.head)
                    if cur_B.head in next_h_graph: next_h_graph.remove_node(cur_B.head)

                    all_leaves.extend(get_lookahead_successors(f, b, next_h_graph, remaining - 2))
            return all_leaves

        if remaining == 1:
            # 1-step logic ensures checking F against cur_B, and B against cur_F
            succs_F = cur_F.generate_successors(args, snake, True)
            F_leaves = []
            for f in succs_F:
                is_valid, should_continue = evaluate_pair(f, cur_B)
                if not is_valid or not should_continue: continue
                
                # FIX: Remove the OLD forward head
                next_h_graph_F = cur_h_graph.copy()
                if cur_F.head in next_h_graph_F: next_h_graph_F.remove_node(cur_F.head)
                
                h_val = V
                if heuristic_name:
                    h_val = heuristic(f, cur_B, heuristic_name, snake, args, next_h_graph_F.copy() if snake else next_h_graph_F)
                F_leaves.append((h_val, f, cur_B, next_h_graph_F))
            
            F_leaves.sort(key=lambda item: item[0], reverse=True)
            max_h_F = F_leaves[0][0] if F_leaves else -1
            avg_h_F = sum(item[0] for item in F_leaves) / len(F_leaves) if F_leaves else -1

            succs_B = cur_B.generate_successors(args, snake, False)
            B_leaves = []
            for b in succs_B:
                is_valid, should_continue = evaluate_pair(cur_F, b)
                if not is_valid or not should_continue: continue
                
                # FIX: Remove the OLD backward head
                next_h_graph_B = cur_h_graph.copy()
                if cur_B.head in next_h_graph_B: next_h_graph_B.remove_node(cur_B.head)
                
                h_val = V
                if heuristic_name:
                    h_val = heuristic(cur_F, b, heuristic_name, snake, args, next_h_graph_B.copy() if snake else next_h_graph_B)
                B_leaves.append((h_val, cur_F, b, next_h_graph_B))
            
            B_leaves.sort(key=lambda item: item[0], reverse=True)
            max_h_B = B_leaves[0][0] if B_leaves else -1
            avg_h_B = sum(item[0] for item in B_leaves) / len(B_leaves) if B_leaves else -1

            expand_F = True
            if max_h_B > max_h_F or (max_h_B == max_h_F and avg_h_B > avg_h_F):
                expand_F = False

            if expand_F:
                stats["generated"]['F'] += len(succs_F)
                if len(succs_F) > 0: stats["num_of_states_per_g"]['F'][cur_F.g+1] += len(succs_F)
                return F_leaves
            else:
                stats["generated"]['B'] += len(succs_B)
                if len(succs_B) > 0: stats["num_of_states_per_g"]['B'][cur_B.g+1] += len(succs_B)
                return B_leaves

    def exp_n_check_states(state_F, state_B, h_graph):
        nonlocal global_longest_path, global_meet_point
        
        # Expand macro-state
        stats["expansions"] += 1
        
        # Retrieve thoroughly validated leaves
        leaves = get_lookahead_successors(state_F, state_B, h_graph, args.lookahead)
        leaves.sort(key=lambda item: item[0], reverse=True)

        for h_val, leaf_F, leaf_B, leaf_h_graph in leaves:
            if args.bsd:
                double_state_key = (leaf_F.head, leaf_F.path_vertices_and_neighbors if snake else leaf_F.path_vertices, leaf_B.head, leaf_B.path_vertices_and_neighbors if snake else leaf_B.path_vertices)
                if double_state_key in FNV and FNV[double_state_key] >= leaf_F.g + leaf_B.g:
                    stats["symmetric_states_removed"] += 1
                    continue
            
            # DFBnB Pruning
            if leaf_F.g + h_val + leaf_B.g <= len(global_longest_path) - 1: 
                stats["violations"]["heuristic"][state_F.g] += 1
                break 

            if args.bsd: FNV[double_state_key] = leaf_F.g + leaf_B.g

            exp_n_check_states(leaf_F, leaf_B, leaf_h_graph)

                
    h_graph = graph.copy()
    
    # Initialize the search by checking the starting positions
    is_valid, should_continue = evaluate_pair(initial_state_F, initial_state_B)
    if is_valid and should_continue:
        exp_n_check_states(initial_state_F, initial_state_B, h_graph)
    
    return global_longest_path, stats, global_meet_point