import matplotlib.pyplot as plt
import heapq, time
from heuristics.heuristic import heuristic
from models.state import State
from models.openvopen import Openvopen
from models.heapq_state import HeapqState
from utils.utils import *
import math


def coil_cycle_vertices_from_states(s1: State, s2: State) -> list[int]:
    # cycle = path1 + reverse(path2[1:-1]) (no repeated endpoints)
    return s1.path + list(reversed(s2.path[1:-1]))


def coil_len_from_states(s1: State, s2: State) -> int:
    # For a cycle, #edges == #vertices
    return len(coil_cycle_vertices_from_states(s1, s2))


def states_form_valid_coil(s1: State, s2: State, d: int, snake: bool = True) -> bool:
    """
    Returns True iff two start->goal *State* objects define two s->t paths whose union forms
    an induced cycle (coil) in Q_d.

    Uses State fields for fast rejection (bitmaps) and only then checks induced-cycle.
    Assumes:
      - s1.path and s2.path are stored (you re-enabled path).
      - graph is Q_d (int vertices 0..2^d-1).
    """
    if s1 is None or s2 is None:
        return False
    if not s1.path or not s2.path:
        return False

    # Both must be start->goal
    if s1.path[0] != s2.path[0] or s1.path[-1] != s2.path[-1]:
        return False

    start = s1.path[0]
    goal = s1.path[-1]

    # Quick disjointness (internal vertices must be disjoint for a simple cycle)
    # Use bitmaps for speed.
    # path_vertices_bitmap is "visited excluding head". It includes start and internal nodes.
    # Internal-only bitmap = path_vertices_bitmap with start bit cleared.
    start_bit = 1 << start
    internal1 = s1.path_vertices_bitmap & ~start_bit
    internal2 = s2.path_vertices_bitmap & ~start_bit

    if internal1 & internal2:
        return False

    # Stronger snake-style quick rejection:
    # If you want the resulting cycle to be a snake/coil, internal nodes of one path
    # should not touch (as neighbors) the other path's body.
    # Your pvan is "body vertices + their neighbors, excluding head".
    # This is a *sufficient* quick reject (can have false negatives only if your pvan
    # definition differs), but generally matches your snake constraints.
    if snake:
        if (s1.path_vertices_and_neighbors_bitmap & (s2.path_vertices_bitmap & ~start_bit)) != 0:
            return False
        if (s2.path_vertices_and_neighbors_bitmap & (s1.path_vertices_bitmap & ~start_bit)) != 0:
            return False

    # Build the cycle vertex order: go along s1 start->goal, then go back along s2 goal->start
    # without repeating endpoints.
    p1 = s1.path
    p2 = s2.path
    cycle = p1 + list(reversed(p2[1:-1]))

    # Must be a cycle with unique vertices
    if len(cycle) < 4:
        return False
    if len(set(cycle)) != len(cycle):
        return False

    # Check all consecutive edges exist in Q_d (diff is power of two)
    m = len(cycle)
    for i in range(m):
        u = cycle[i]
        v = cycle[(i + 1) % m]
        diff = u ^ v
        if diff == 0 or (diff & (diff - 1)) != 0:
            return False

    # Induced (no chords): in Q_d, chord exists iff two nonconsecutive vertices differ in 1 bit.
    pos = {v: i for i, v in enumerate(cycle)}
    for u in cycle:
        iu = pos[u]
        for bit in range(d):
            v = u ^ (1 << bit)
            if v in pos:
                iv = pos[v]
                if (iu - iv) % m in (1, m - 1):
                    continue
                return False

    return True


def tbt_search(graph, d, buffer_dim, heuristic_name, snake, args):
    start = 0
    goal = 2**d - 1
    valid_st_paths = []
    best_coil = None
    best_coil_len = -1
    best_coil_pair = None  # (idx_old, idx_new) optional

    
    calc_h_time = 0
    valid_meeting_check_time = 0
    valid_meeting_checks = 0
    valid_meeting_checks_sum_g_under_f_max = 0
    g_values = []
    BF_values = []

    # Options
    alternate = False # False
    lastDirectionF = False

    # Initialize meeting point of the two searches
    best_path_meet_point = None

    # Initialize custom priority queues for forward and backward searches
    OPEN_F = HeapqState()
    OPEN_B = HeapqState()
    OPENvOPEN = Openvopen(graph, start, goal)

    # Initial states
    initial_state_F = State(graph, [start], [], snake) if isinstance(start, int) else State(graph, start, [], snake)
    initial_state_B = State(graph, [goal], [], snake) if isinstance(goal, int) else State(graph, goal, [], snake)

    # Initial f_values
    initial_state_F.h = heuristic(initial_state_F, goal, heuristic_name, snake)
    initial_f_value_F = initial_state_F.g + initial_state_F.h
    initial_state_B.h = heuristic(initial_state_B, start, heuristic_name, snake)
    initial_f_value_B = initial_state_B.g + initial_state_B.h

    # Push initial states with priority based on f_value
    OPEN_F.push(initial_state_F, initial_f_value_F)
    OPEN_B.push(initial_state_B, initial_f_value_B)
    OPENvOPEN.insert_state(initial_state_F, True)
    OPENvOPEN.insert_state(initial_state_B, False)
    FNV_F = {(initial_state_F.head, initial_state_F.path_vertices_and_neighbors_bitmap if snake else initial_state_F.path_vertices_bitmap)}
    FNV_B = {(initial_state_B.head, initial_state_B.path_vertices_and_neighbors_bitmap if snake else initial_state_B.path_vertices_bitmap)}

    # Best path found and its length
    best_path = None        # S in the pseudocode
    best_path_length = -1   # U in the pseudocode

    # Expansion counter, generated counter
    expansions = 0
    generated = 0
    moved_OPEN_to_AUXOPEN = 0

    # Closed sets for forward and backward searches
    CLOSED_F = set()
    CLOSED_B = set()

    while len(OPEN_F) > 0 or len(OPEN_B) > 0:
        # Determine which direction to expand
        directionF = None # True - Forward, False - Backward 
        if alternate:
            directionF = False if lastDirectionF else True
            lastDirectionF = not lastDirectionF
        else:
            if len(OPEN_F) > 0 and (
                len(OPEN_B) == 0 or OPEN_F.top()[0] >= OPEN_B.top()[0]
            ):
                directionF = True
            else:
                directionF = False

        # Set general variables
        D, D_hat = ('F', 'B') if directionF else ('B', 'F')
        OPEN_D, OPEN_D_hat = (OPEN_F, OPEN_B) if directionF else (OPEN_B, OPEN_F)
        CLOSED_D, CLOSED_D_hat = (CLOSED_F, CLOSED_B) if directionF else (CLOSED_B, CLOSED_F)
        FNV_D , FNV_D_hat = (FNV_F, FNV_B) if directionF else (FNV_B, FNV_F)

        # Get the best state from OPEN_D
        f_value, g_value, current_state = OPEN_D.top()
        current_path_length = len(current_state.path) - 1
        
        if expansions % 10000 == 0:
            print(
                f"Expansion #{expansions}: state {current_state.path}, f={f_value}, len={len(current_state.path)}"
            )
            with open(args.log_file_name, 'a') as file:
                file.write(f"\nExpansion #{expansions}: state {current_state.path}, f={f_value}, len={len(current_state.path)}")

            # print(f"closed_F: {len(closed_set_F)}. closed_B: {len(closed_set_B)}")
            # print(f"open_F: {len(open_set_F)}. open_B: {len(open_set_B)}")

        # Check against OPEN of the other direction, for a valid meeting point
        curr_time = time.time()
        state, _, _, _, num_checks, num_checks_sum_g_under_f_max = OPENvOPEN.find_longest_non_overlapping_state(current_state, directionF, best_path_length, f_value, snake)
        valid_meeting_check_time += time.time() - curr_time
        valid_meeting_checks += num_checks
        valid_meeting_checks_sum_g_under_f_max += num_checks_sum_g_under_f_max

        if state:
            total_length = current_path_length + len(state.path) - 1
            candidate_state = current_state + state

            # ---- NEW: compare against all previous candidate s->t states ----
            for i, prev in enumerate(valid_st_paths):
                if states_form_valid_coil(prev, candidate_state, d, snake=snake):
                    L = coil_len_from_states(prev, candidate_state)
                    if L > best_coil_len:
                        best_coil_len = L
                        best_coil = coil_cycle_vertices_from_states(prev, candidate_state)
                        best_coil_pair = (i, len(valid_st_paths))  # prev index, new index (after append)

            # append only after comparing (so we don't compare to itself)
            valid_st_paths.append(candidate_state)
            if total_length > best_path_length:
                best_path_length = total_length
                best_path = current_state.path[:-1] + state.path[::-1]
                best_path_meet_point = current_state.head
                if snake and total_length >= f_value-3:
                    print(f"[{time2str(args.start_time,time.time())} expansion {expansions}, {time_ms(args.start_time,time.time())}] Found path of length {total_length}: {best_path}. g_F={current_path_length}, g_B={len(state.path) - 1}, f_max={f_value}, generated={generated}")
                    with open(args.log_file_name, 'a') as file:
                        file.write(f"[{time2str(args.start_time,time.time())} expansion {expansions}] Found path of length {total_length}. {best_path}. g_F={current_path_length}, g_B={len(state.path) - 1}, f_max={f_value}\n")

        # Termination Condition: check if U is the largest it will ever be
        # if best_path_length >= min(
        #     OPEN_F.top()[0] if len(OPEN_F) > 0 else float("inf"),
        #     OPEN_B.top()[0] if len(OPEN_B) > 0 else float("inf"),
        # ):
        #     # print(f"Terminating with best path of length {best_path_length}")
        #     break

        
        # XMM_full. if g > f_max/2 don't expant it, but keep it in OPENvOPEN for checking collision of search from the other side
        # if C* = 20, in the F direction we won't expand S with g > 9, in the B direction we won't expand S with g > 9.5 
        # if C* = 19, in the F direction we won't expand S with g > 8.5, in the B direction we won't expand S with g > 9 
        if args.algo == "cutoff" or args.algo == "full":
            if (D=='F' and current_state.g > f_value/2 - 1) or (D=='B' and current_state.g > (f_value - 1)/2): 
                OPEN_D.pop()
                moved_OPEN_to_AUXOPEN += 1
                # print(f"Not expanding state {current_state.path} because state.g = {current_state.g}")
                continue

        expansions += 1
        g_values.append(current_state.g)

        # Get the current state from OPEN_D TO CLOSED_D
        f_value, g_value, current_state = OPEN_D.pop()
        # OPENvOPEN.remove_state(current_state, directionF)
        CLOSED_D.add(current_state)

        if current_state.traversed_buffer_dimension:
            continue

        # Generate successors
        successors = current_state.successor(args, snake, directionF, ignore_max_dim_crossed=True)
        BF_values.append(len(successors))
        for successor in successors:
            if args.bsd and (successor.head, successor.path_vertices_and_neighbors_bitmap if snake else successor.path_vertices_bitmap) in FNV_D:
                # print(f"symmetric state removed: {successor.path}")
                continue

            if has_bridge_edge_across_dim(current_state, successor, buffer_dim):
                successor.traversed_buffer_dimension = True
            if (not directionF) and has_bridge_edge_across_dim(successor, current_state, buffer_dim):
                # print(f"Skipping expansion of {successor.path} from OPEN_B due to TBT bridge edge across dim {buffer_dim}.")
                continue

            generated += 1
            
            curr_time = time.time()
            h_successor = heuristic(
                successor, goal if directionF else start, heuristic_name, snake
            )
            calc_h_time += time.time() - curr_time

            # # For Plotting h
            # h_BCC.append(h_value)
            # h_mis = heuristic(successor, goal if direction == "F" else start, "mis_heuristic")
            # h_MIS.append(h_mis+0.1)
            # mis_smaller_flag.append(-1 if h_value<h_mis else 0 if h_value==h_mis else 1)
            # max_f.append(f_value)
            # expansions_list.append(expansions)

            g_successor = current_path_length + 1
            f_successor = g_successor + h_successor

            # XMM_light + PathMin
            if args.algo == "light" or args.algo == "full":
                OPEN_D.push(successor, min(2 * h_successor, f_value, f_successor))
            else: OPEN_D.push(successor, min(f_value, f_successor))
            
            FNV_D.add((successor.head, successor.path_vertices_and_neighbors_bitmap if snake else successor.path_vertices_bitmap))
            OPENvOPEN.insert_state(successor,directionF)


    return best_path, expansions, generated, moved_OPEN_to_AUXOPEN, best_path_meet_point, g_values
