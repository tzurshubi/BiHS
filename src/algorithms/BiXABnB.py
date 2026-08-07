import heapq, time, itertools
from heuristics.heuristic import heuristic
from models.state import State
from utils.utils import *
import matplotlib.pyplot as plt
from collections import defaultdict

def BiXABnB(graph, start, goal, heuristic_name, snake, args):
    """
    Hybrid A* -> DFBnB Algorithm.
    Runs A* until len(open_set) >= memory_limit, then switches to DFBnB.
    """
    stats = args.stats
    logger = args.logger
    memory_limit = args.memory_limit
    N = max(graph.nodes)
    V = len(graph.nodes)

    initial_state_F = State(graph, [start], [], snake, args) if isinstance(start, int) else State(graph, start, [], snake, args)
    initial_state_B = State(graph, [goal], [], snake, args) if isinstance(goal, int) else State(graph, goal, [], snake, args)
    
    stats['num_of_states_per_g']['F'][0] += 1
    stats['num_of_states_per_g']['B'][0] += 1

    global_longest_path = []
    global_meet_point = None

    if args.bsd:
        state_key = (initial_state_F.head, initial_state_F.path_vertices_and_neighbors if snake else initial_state_F.path_vertices,
                     initial_state_B.head, initial_state_B.path_vertices_and_neighbors if snake else initial_state_B.path_vertices)
        FNV = {state_key: initial_state_F.g + initial_state_B.g}

    def evaluate_pair(f, b):
        nonlocal global_longest_path, global_meet_point
        
        stats["valid_meeting_checks"] += 1
        if f.head == b.head: stats["state_vs_state_meeting_checks"] += 1
        else: stats["prefix_vs_prefix_meeting_checks"] += 1
        
        # if stats["expansions"] % 100_000 == 0 and stats["expansions"] > 0:
        #     logger(f"Expansions: {stats['expansions']}. Checks - state_vs_state: {stats['state_vs_state_meeting_checks']}, prefix: {stats['prefix_vs_prefix_meeting_checks']}")

        if f.head == b.head:
            if f.g + b.g > len(global_longest_path) - 1:
                global_longest_path = f.materialize_path() + b.materialize_path()[::-1][1:]
                global_meet_point = f.head
                if args.graph_type == "cube": logger(f"Expansion {stats['expansions']}: New longest path found with length {len(global_longest_path) - 1}: {global_longest_path}")
            return True, False 

        is_f_in_b = is_vertex_in_bitmap(f.head, b.illegal)
        is_b_in_f = is_vertex_in_bitmap(b.head, f.illegal)
        
        if is_f_in_b or is_b_in_f:
            stats["violations"]["intersection"][f.g] += 1
            return False, False

        elif graph.has_edge(f.head, b.head):
            if f.g + 1 + b.g > len(global_longest_path) - 1:
                global_longest_path = f.materialize_path() + [b.head] + b.materialize_path()[::-1][1:]
                global_meet_point = b.head
                if args.graph_type == "cube": logger(f"Expansion {stats['expansions']}: New longest path found with length {len(global_longest_path) - 1}: {global_longest_path}")
            if not snake:
                return True, True 
            else:
                stats["violations"]["intersection"][f.g] += 1
                return False, False 

        return True, True 

    def get_lookahead_successors(cur_F, cur_B, cur_h_graph, remaining, expand_F_turn=True):
        # Lookahead logic remains identical to your provided implementations.
        if remaining == -2:
            succs_F = cur_F.generate_successors(args, snake, True)
            succs_B = cur_B.generate_successors(args, snake, False)
            expand_F = len(succs_F) <= len(succs_B)

            if expand_F:
                stats["generated"]['F'] += len(succs_F)
                if len(succs_F) > 0: stats["num_of_states_per_g"]['F'][cur_F.g+1] += len(succs_F)
                next_h_graph = cur_h_graph.copy()
                if cur_F.head in next_h_graph: next_h_graph.remove_node(cur_F.head)
                leaves = []
                for f in succs_F:
                    is_valid, should_continue = evaluate_pair(f, cur_B) 
                    if not is_valid or not should_continue: continue
                    h_val = V
                    if heuristic_name: h_val = heuristic(f, cur_B, heuristic_name, snake, args, next_h_graph.copy() if snake else next_h_graph)
                    leaves.append((h_val, f, cur_B, next_h_graph))
                return leaves
            else:
                stats["generated"]['B'] += len(succs_B)
                if len(succs_B) > 0: stats["num_of_states_per_g"]['B'][cur_B.g+1] += len(succs_B)
                next_h_graph = cur_h_graph.copy()
                if cur_B.head in next_h_graph: next_h_graph.remove_node(cur_B.head)
                leaves = []
                for b in succs_B:
                    is_valid, should_continue = evaluate_pair(cur_F, b) 
                    if not is_valid or not should_continue: continue
                    h_val = V
                    if heuristic_name: h_val = heuristic(cur_F, b, heuristic_name, snake, args, next_h_graph.copy() if snake else next_h_graph)
                    leaves.append((h_val, cur_F, b, next_h_graph))
                return leaves

        if remaining == -1:
            if expand_F_turn:
                succs_F = cur_F.generate_successors(args, snake, True)
                stats["generated"]['F'] += len(succs_F)
                if len(succs_F) > 0: stats["num_of_states_per_g"]['F'][cur_F.g+1] += len(succs_F)
                next_h_graph = cur_h_graph.copy()
                if cur_F.head in next_h_graph: next_h_graph.remove_node(cur_F.head)
                leaves = []
                for f in succs_F:
                    is_valid, should_continue = evaluate_pair(f, cur_B) 
                    if not is_valid or not should_continue: continue
                    h_val = V
                    if heuristic_name: h_val = heuristic(f, cur_B, heuristic_name, snake, args, next_h_graph.copy() if snake else next_h_graph)
                    leaves.append((h_val, f, cur_B, next_h_graph))
                return leaves
            else:
                succs_B = cur_B.generate_successors(args, snake, False)
                stats["generated"]['B'] += len(succs_B)
                if len(succs_B) > 0: stats["num_of_states_per_g"]['B'][cur_B.g+1] += len(succs_B)
                next_h_graph = cur_h_graph.copy()
                if cur_B.head in next_h_graph: next_h_graph.remove_node(cur_B.head)
                leaves = []
                for b in succs_B:
                    is_valid, should_continue = evaluate_pair(cur_F, b) 
                    if not is_valid or not should_continue: continue
                    h_val = V
                    if heuristic_name: h_val = heuristic(cur_F, b, heuristic_name, snake, args, next_h_graph.copy() if snake else next_h_graph)
                    leaves.append((h_val, cur_F, b, next_h_graph))
                return leaves

        if remaining == 0:
            h_val = V
            if heuristic_name: h_val = heuristic(cur_F, cur_B, heuristic_name, snake, args, cur_h_graph.copy() if snake else cur_h_graph)
            return [(h_val, cur_F, cur_B, cur_h_graph)]
        
        if remaining >= 2:
            succs_F = cur_F.generate_successors(args, snake, True)
            succs_B = cur_B.generate_successors(args, snake, False)
            stats["generated"]['F'] += len(succs_F)
            stats["generated"]['B'] += len(succs_B)
            if len(succs_F) > 0: stats["num_of_states_per_g"]['F'][cur_F.g+1] += len(succs_F)
            if len(succs_B) > 0: stats["num_of_states_per_g"]['B'][cur_B.g+1] += len(succs_B)

            next_h_graph = cur_h_graph.copy()
            if cur_F.head in next_h_graph: next_h_graph.remove_node(cur_F.head)
            if cur_B.head in next_h_graph: next_h_graph.remove_node(cur_B.head)

            all_leaves = []
            for f in succs_F:
                for b in succs_B:
                    is_valid, should_continue = evaluate_pair(f, b)
                    if not is_valid or not should_continue: continue
                    if remaining == 2:
                        h_val = V
                        if heuristic_name: h_val = heuristic(f, b, heuristic_name, snake, args, next_h_graph.copy() if snake else next_h_graph)
                        all_leaves.append((h_val, f, b, next_h_graph))
                    else:
                        all_leaves.extend(get_lookahead_successors(f, b, next_h_graph, remaining - 2))
            return all_leaves

        if remaining == 1:
            succs_F = cur_F.generate_successors(args, snake, True)
            next_h_graph_F = cur_h_graph.copy()
            if cur_F.head in next_h_graph_F: next_h_graph_F.remove_node(cur_F.head)
            F_leaves = []
            for f in succs_F:
                is_valid, should_continue = evaluate_pair(f, cur_B)
                if not is_valid or not should_continue: continue
                h_val = V
                if heuristic_name: h_val = heuristic(f, cur_B, heuristic_name, snake, args, next_h_graph_F.copy() if snake else next_h_graph_F)
                F_leaves.append((h_val, f, cur_B, next_h_graph_F))
            
            F_leaves.sort(key=lambda item: item[0], reverse=True)
            max_h_F = F_leaves[0][0] if F_leaves else -1
            avg_h_F = sum(item[0] for item in F_leaves) / len(F_leaves) if F_leaves else -1

            succs_B = cur_B.generate_successors(args, snake, False)
            next_h_graph_B = cur_h_graph.copy()
            if cur_B.head in next_h_graph_B: next_h_graph_B.remove_node(cur_B.head)
            B_leaves = []
            for b in succs_B:
                is_valid, should_continue = evaluate_pair(cur_F, b)
                if not is_valid or not should_continue: continue
                h_val = V
                if heuristic_name: h_val = heuristic(cur_F, b, heuristic_name, snake, args, next_h_graph_B.copy() if snake else next_h_graph_B)
                B_leaves.append((h_val, cur_F, b, next_h_graph_B))
            
            B_leaves.sort(key=lambda item: item[0], reverse=True)
            max_h_B = B_leaves[0][0] if B_leaves else -1
            avg_h_B = sum(item[0] for item in B_leaves) / len(B_leaves) if B_leaves else -1

            expand_F = True
            if max_h_B > max_h_F or (max_h_B == max_h_F and avg_h_B > avg_h_F): expand_F = False

            if expand_F:
                stats["generated"]['F'] += len(succs_F)
                if len(succs_F) > 0: stats["num_of_states_per_g"]['F'][cur_F.g+1] += len(succs_F)
                return F_leaves
            else:
                stats["generated"]['B'] += len(succs_B)
                if len(succs_B) > 0: stats["num_of_states_per_g"]['B'][cur_B.g+1] += len(succs_B)
                return B_leaves

    # --- Phase 1: Best-First Search (A*) ---
    open_set = []
    tie_breaker = itertools.count()

    initial_h_value = V
    if heuristic_name:
        initial_h_value = heuristic(initial_state_F, initial_state_B, heuristic_name, snake, args, graph.copy())
    
    initial_f_value = initial_state_F.g + initial_state_B.g + initial_h_value
    heapq.heappush(open_set, (-initial_f_value, -(initial_state_F.g + initial_state_B.g), next(tie_breaker), (initial_state_F, initial_state_B, graph.copy(), True)))

    is_valid, should_continue = evaluate_pair(initial_state_F, initial_state_B)
    if not is_valid or not should_continue:
        return global_longest_path, stats, global_meet_point

    # Stop A* phase if queue exceeds memory limit
    while len(open_set) > 0 and len(open_set) < memory_limit:
        neg_priority, neg_g, _, (current_F, current_B, h_graph, expand_F_turn) = heapq.heappop(open_set)
        f_value = -neg_priority
        g_value = -neg_g

        stats["expansions"] += 1
        if stats["expansions"] % 50_000 == 0:
            logger(f"Expansions: {stats['expansions']}. F states: {stats['generated']['F']}, B states: {stats['generated']['B']}. Checks: {stats['valid_meeting_checks']})")
        if stats["expansions"] % 5000 == 0:
            logger(f"OPEN list length: {len(open_set)}")


        if global_longest_path and f_value <= len(global_longest_path) - 1:
            break

        leaves = get_lookahead_successors(current_F, current_B, h_graph, args.lookahead, expand_F_turn)

        for h_val, leaf_F, leaf_B, leaf_h_graph in leaves:
            current_f_value = leaf_F.g + leaf_B.g + h_val

            if global_longest_path and current_f_value <= len(global_longest_path) - 1:
                stats["violations"]["heuristic"][leaf_F.g] += 1
                continue

            if args.bsd:
                double_state_key = (leaf_F.head, leaf_F.path_vertices_and_neighbors if snake else leaf_F.path_vertices, 
                                    leaf_B.head, leaf_B.path_vertices_and_neighbors if snake else leaf_B.path_vertices)
                if double_state_key in FNV and FNV[double_state_key] >= leaf_F.g + leaf_B.g:
                    stats["symmetric_states_removed"] += 1
                    continue
                FNV[double_state_key] = leaf_F.g + leaf_B.g

            next_turn = not expand_F_turn if args.lookahead in [-1, -2] else True
            priority = min(current_f_value, f_value)

            heapq.heappush(open_set, (-priority, -(leaf_F.g + leaf_B.g), next(tie_breaker), (leaf_F, leaf_B, leaf_h_graph, next_turn)))


    # --- Phase 2: Depth-First Branch and Bound (DFBnB) ---
    def exp_n_check_states(state_F, state_B, h_graph, expand_F_turn=True):
        nonlocal global_longest_path, global_meet_point
        stats["expansions"] += 1
        
        leaves = get_lookahead_successors(state_F, state_B, h_graph, args.lookahead, expand_F_turn)
        leaves.sort(key=lambda item: item[0], reverse=True)

        for h_val, leaf_F, leaf_B, leaf_h_graph in leaves:
            if args.bsd:
                double_state_key = (leaf_F.head, leaf_F.path_vertices_and_neighbors if snake else leaf_F.path_vertices, leaf_B.head, leaf_B.path_vertices_and_neighbors if snake else leaf_B.path_vertices)
                if double_state_key in FNV and FNV[double_state_key] >= leaf_F.g + leaf_B.g:
                    stats["symmetric_states_removed"] += 1
                    continue
            
            if leaf_F.g + h_val + leaf_B.g <= len(global_longest_path) - 1: 
                stats["violations"]["heuristic"][state_F.g] += 1
                break 

            if args.bsd: FNV[double_state_key] = leaf_F.g + leaf_B.g
            next_turn = not expand_F_turn if args.lookahead == -1 else True
            exp_n_check_states(leaf_F, leaf_B, leaf_h_graph, next_turn)

    if len(open_set) >= memory_limit:
        logger(f"Memory limit of {memory_limit} reached. {stats['expansions']} expansions. OPEN list length: {len(open_set)}. Switching to DFBnB.")

    # Sort the remaining frontier to expand the best nodes first. 
    # Because items are stored as (-priority, ...), sorting normally puts the most promising nodes at index 0.
    open_set.sort(key=lambda x: (x[0], x[1]))

    for item in open_set:
        neg_priority, neg_g, _, (current_F, current_B, h_graph, expand_F_turn) = item
        
        # Prune against the best path found during either A* or previous DFBnB branches
        if -neg_priority <= len(global_longest_path) - 1:
            continue
            
        exp_n_check_states(current_F, current_B, h_graph, expand_F_turn)

    return global_longest_path, stats, global_meet_point