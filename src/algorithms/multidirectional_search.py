import matplotlib.pyplot as plt
import heapq, time
from heuristics.heuristic import heuristic
from models.state import State
from models.openvopen import Openvopen
from models.multi_openvopen import MultiOpenvopen
from models.heapq_state import HeapqState
from utils.utils import *


def initialize_segments_and_frontiers(graph, start, goal, solution_vertices, heuristic_name, snake, args):
    frontiers = {}
    segments = {}

    # Initialize base segments
    segments[0] = {"start": start, "goal": solution_vertices[0], "name": f"s-v{solution_vertices[0]}", id: 0}
    segments[len(solution_vertices)] = {"start": solution_vertices[-1], "goal": goal, "name": f"v{solution_vertices[-1]}-t", id: len(solution_vertices)}
    
    # Initialize base frontiers
    frontiers["s_F"] = {
        "start": start,
        "goal": solution_vertices[0],
        "segment": segments[0],
        "direction": "F"
    }
    frontiers[f"v{solution_vertices[0]}_B"] = {
        "start": solution_vertices[0],
        "goal": start,
        "segment": segments[0],
        "direction": "B"
    }
    frontiers[f"v{solution_vertices[-1]}_F"] = {
        "start": solution_vertices[-1],
        "goal": goal,
        "segment": segments[len(solution_vertices)],
        "direction": "F"
    }
    frontiers["t_B"] = {
        "start": goal,
        "goal": solution_vertices[-1],
        "segment": segments[len(solution_vertices)],
        "direction": "B"
    }

    # Intermediate vertices
    for i, sv in enumerate(solution_vertices):
        if i == 0:
            continue  # already added
        segments[i] = {
            "start": solution_vertices[i-1],
            "goal": sv,
            "name": f"v{solution_vertices[i-1]}-v{sv}",
            id: i
        }
        frontiers[f"v{sv}_B"] = {
            "start": sv,
            "goal": start if i == 0 else solution_vertices[i-1],
            "segment": segments[f"v{solution_vertices[i-1]}-v{sv}"] if i > 0 else segments[f"s-v{sv}"],
            "direction": "B"
        }
        frontiers[f"v{sv}_F"] = {
            "start": sv,
            "goal": goal if i == len(solution_vertices) - 1 else solution_vertices[i + 1],
            "segment": segments[f"v{sv}-v{solution_vertices[i+1]}"]
            if i < len(solution_vertices) - 1 else segments[f"v{sv}-t"],
        "direction": "F"
        }

    # Connect segments to their frontiers
    for segment_key, segment in segments.items():
        # Best path found and its length
        segment["best_path"] = None        # S in the pseudocode
        segment["best_path_length"] = -1   # U in the pseudocode
        segment["best_path_meet_point"] = None
        segment["expansions"] = 0
        segment["generated"] = 0
        segment["OPENvOPEN"] = Openvopen(graph, segment["start"], segment["goal"])
        segment["frontier_F"] = (
            frontiers.get(f"v{segment['start']}_F")
            if segment["start"] in solution_vertices
            else frontiers.get("s_F")
        )
        segment["frontier_B"] = (
            frontiers.get(f"v{segment['goal']}_B")
            if segment["goal"] in solution_vertices
            else frontiers.get("t_B")
        )

    # Initialize OPEN/CLOSED for all frontiers
    for frontier in frontiers.values():
        frontier["OPEN"] = HeapqState()
        frontier["CLOSED"] = set()

    # Initialize all frontiers with their first states
    for frontier_key, frontier in frontiers.items():
        initial_state = (
            State(graph, [frontier["start"]], [], snake)
            if isinstance(frontier["start"], int)
            else State(graph, frontier["start"], [], snake)
        )
        initial_state.h = heuristic(initial_state, frontier["goal"], heuristic_name, snake)
        initial_f_value = initial_state.g + initial_state.h

        # Push initial state
        frontier["OPEN"].push(initial_state, initial_f_value)
        frontier["segment"]["OPENvOPEN"].insert_state(initial_state, frontier_key.endswith("_F"))
        frontier["FNV"] = {
            (
                initial_state.head,
                initial_state.path_vertices_and_neighbors_bitmap if snake else initial_state.path_vertices_bitmap
            )
        }
    OPENs_empty = len(frontiers) * [False]
    return segments, frontiers, OPENs_empty


def multidirectional_search(graph, start, goal, solution_vertices, heuristic_name, snake, args):
    k = len(solution_vertices)
    calc_h_time = 0
    frontiers = {}
    segments = {}

    # Options
    alternate = False
    lastDirectionF = False

    # Initialize meeting point of the two searches
    best_path_meet_point = None

    # Initialize frontiers & segments
    segments, frontiers, OPENs_empty = initialize_segments_and_frontiers(
        graph, start, goal, solution_vertices, heuristic_name, snake, args
    )
    
    # Best path found and its length
    multi_best_path = None        # S in the pseudocode
    multi_best_path_length = -1   # U in the pseudocode
    multi_best_path_meet_point = None

    # Expansion counter, generated counter
    multi_expansions = 0
    multi_generated = 0
    multi_moved_OPEN_to_AUXOPEN = 0

    multi_openvopen = MultiOpenvopen(graph, snake, max(graph.nodes) + 1, segments, start, goal, solution_vertices)
    stop_multi_search = False

    while any(len(f["OPEN"]) > 0 for f in frontiers.values()): # While the union of all OPEN lists is not empty
        for segment_id, segment in segments.items():
            # print(f"--- Segment {segment["name"]} ---")
            OPENvOPEN, OPEN_F, OPEN_B, CLOSED_F, CLOSED_B, FNV_F, FNV_B, best_path, best_path_length, best_path_meet_point, expansions, generated = segment["OPENvOPEN"], segment["frontier_F"]["OPEN"], segment["frontier_B"]["OPEN"], segment["frontier_F"]["CLOSED"], segment["frontier_B"]["CLOSED"], segment["frontier_F"]["FNV"], segment["frontier_B"]["FNV"], segment["best_path"], segment["best_path_length"], segment["best_path_meet_point"], segment["expansions"], segment["generated"]
            while len(OPEN_F) > 0 or len(OPEN_B) > 0:
                # Determine which direction to expand
                directionF = None # True - Forward, False - Backward 
                if alternate:
                    directionF = False if lastDirectionF else True
                    lastDirectionF = not lastDirectionF
                else:
                    if len(OPEN_F) > 0 and (len(OPEN_B) == 0 or OPEN_F.top()[0] >= OPEN_B.top()[0]):
                        directionF = True
                    else:
                        directionF = False

                # Set general variables
                D, D_hat = ('F', 'B') if directionF else ('B', 'F')
                OPEN_D, OPEN_D_hat = (OPEN_F, OPEN_B) if directionF else (OPEN_B, OPEN_F)
                CLOSED_D, CLOSED_D_hat = (CLOSED_F, CLOSED_B) if directionF else (CLOSED_B, CLOSED_F)
                FNV_D , FNV_D_hat = (FNV_F, FNV_B) if directionF else (FNV_B, FNV_F)
                seg_start, seg_goal = (segment["start"], segment["goal"]) if directionF else (segment["goal"], segment["start"])
                additional_frontier = None if (directionF and segment_id==0) or (not directionF and segment_id==len(solution_vertices)) else (segments[segment_id-1]["frontier_B"] if directionF else segments[segment_id+1]["frontier_F"])

                # Get the best state from OPEN_D
                f_value, g_value, current_state = OPEN_D.top()
                # print(f"Current state to expand ({D}): {current_state.path}, f={f_value}, g={current_state.g}, h={f_value - current_state.g}. Best path length: {best_path_length}. OPEN_F: {len(OPEN_F)}. OPEN_B: {len(OPEN_B)}.")

                # Handle the case where the current_state ends with one of {s,v1,..,vk,t} - we should not expand it
                while current_state.head in solution_vertices+[start,goal] and current_state.g > 0:
                    # If this state is a goal for this segment, add it to OPENvOPEN
                    if current_state.head == seg_goal:
                        # print(f"Adding state {current_state.path} to OPENvOPEN of segment {segment['name']} because it reached the goal of the segment.")
                        multi_openvopen.add_paths_to_segment(segment_id, [current_state])
                        st_path, st_path_len = multi_openvopen.update_su_vt_paths(segment_id, [current_state])
                        if st_path_len > multi_best_path_length:
                            multi_best_path_length = st_path_len
                            multi_best_path = st_path
                            best_path_meet_point = current_state.head
                    
                    # Remove from OPEN_D, Add to CLOSED_D
                    OPEN_D.pop()
                    if additional_frontier: additional_frontier["OPEN"].remove(current_state)
                    CLOSED_D.add(current_state)
                    if additional_frontier: additional_frontier["CLOSED"].add(current_state)

                    if OPEN_D.is_empty():
                        break
                    f_value, g_value, current_state = OPEN_D.top()
                current_path_length = len(current_state.path) - 1
                if OPEN_D.is_empty():
                    break

                # Logging
                if args.log:
                    if expansions % 1000 == 0 and expansions > 0:
                        print(f"Expansion #{expansions}: state {current_state.path}, f={f_value}, len={len(current_state.path)}")
                    curr_time = time.time()

                # Check against OPEN of the other direction, for a valid meeting point.
                seg_paths, num_checks, num_checks_sum_g_under_f_max = segment["OPENvOPEN"].find_all_non_overlapping_paths(current_state,directionF, best_path_length, f_value, segment_id, snake)
                multi_openvopen.add_paths_to_segment(segment_id, seg_paths)
                st_path, st_path_len = multi_openvopen.update_su_vt_paths(segment_id, seg_paths)
                if st_path_len > multi_best_path_length:
                    multi_best_path_length = st_path_len
                    multi_best_path = st_path
                    best_path_meet_point = current_state.head
                    print(f"!!! [{time2str(args.start_time,time.time())} expansion {multi_expansions}, {time_ms(args.start_time,time.time())}] Found path of length {multi_best_path_length}: {multi_best_path.path}. g_F={current_path_length}, g_B={st_path_len - current_path_length}, f_max={f_value}, generated={multi_generated}")
                    # with open(args.log_file_name, 'a') as file:
                    #     file.write(f"[{time2str(args.start_time,time.time())} expansion {multi_expansions}] Found path of length {multi_best_path_length}. {multi_best_path}. g_F={current_path_length}, g_B={st_path_len - current_path_length}, f_max={f_value}\n")


                # Termination Condition: check if U is the largest it will ever be
                if best_path_length >= min(
                    OPEN_F.top()[0] if len(OPEN_F) > 0 else float("inf"),
                    OPEN_B.top()[0] if len(OPEN_B) > 0 else float("inf"),
                ):
                    # print(f"Terminating with best path of length {best_path_length}")
                    OPEN_D.pop()
                    break

                
                # XMM_full. if g > f_max/2 don't expant it, but keep it in OPENvOPEN for checking collision of search from the other side
                if args.algo == "cutoff" or args.algo == "full":
                    if (D=='F' and current_state.g > f_value/2 - 1) or (D=='B' and current_state.g > (f_value - 1)/2): 
                        OPEN_D.pop()
                        # moved_OPEN_to_AUXOPEN += 1
                        # print(f"Not expanding state {current_state.path} because state.g = {current_state.g}")
                        continue

                multi_expansions += 1
                if multi_expansions > 100000: break

                # Get the current state from OPEN_D TO CLOSED_D
                f_value, g_value, current_state = OPEN_D.pop()
                if additional_frontier: additional_frontier["OPEN"].remove(current_state)
                # OPENvOPEN.remove_state(current_state, directionF)
                CLOSED_D.add(current_state)
                if additional_frontier: additional_frontier["CLOSED"].add(current_state)

                # Generate successors
                successors = current_state.successor(args, snake, directionF)
                # print(f"Generated {len(successors)} successors: {[s.path for s in successors]}")
                for successor in successors:
                    # Handle symmetric states removal
                    if args.bsd and (successor.head, successor.path_vertices_and_neighbors_bitmap if snake else successor.path_vertices_bitmap) in FNV_D:
                        # print(f"symmetric state removed: {successor.path}")
                        continue

                    generated += 1
                    multi_generated += 1
                    h_successor = heuristic(successor, seg_goal, heuristic_name, snake)
                    if h_successor >= 0:
                        g_successor = current_path_length + 1
                        f_successor = g_successor + h_successor

                        # XMM_light + PathMin
                        if args.algo == "light" or args.algo == "full":
                            OPEN_D.push(successor, min(2 * h_successor, f_value, f_successor))
                        else: OPEN_D.push(successor, min(f_value, f_successor))
                        FNV_D.add((successor.head, successor.path_vertices_and_neighbors_bitmap if snake else successor.path_vertices_bitmap))
                        OPENvOPEN.insert_state(successor,directionF)
                    
                    # If this is not Forward from start or Backward from goal, we need to add this state to another OPEN list
                    if additional_frontier:
                        additional_segment = additional_frontier["segment"]
                        h_successor = heuristic(successor, additional_frontier["goal"] if directionF else additional_frontier["start"], heuristic_name, snake)
                        if h_successor >= 0:
                            f_successor = g_successor + h_successor
                            
                            # XMM_light + PathMin
                            if args.algo == "light" or args.algo == "full":
                                additional_frontier["OPEN"].push(successor, min(2 * h_successor, f_value, f_successor))
                            else: additional_frontier["OPEN"].push(successor, min(f_value, f_successor))
                            additional_frontier["FNV"].add((successor.head, successor.path_vertices_and_neighbors_bitmap if snake else successor.path_vertices_bitmap))
                            additional_segment["OPENvOPEN"].insert_state(successor,not directionF)
                        
                # Update the segment and frontier structures
                segment.update({
                    "OPENvOPEN": OPENvOPEN,
                    "best_path": best_path,
                    "best_path_length": best_path_length,
                    "best_path_meet_point": best_path_meet_point,
                    "expansions": expansions,
                    "generated": generated,
                })
                segment["frontier_F"].update({
                    "OPEN": OPEN_F,
                    "CLOSED": CLOSED_F,
                    "FNV": FNV_F,
                })
                segment["frontier_B"].update({
                    "OPEN": OPEN_B,
                    "CLOSED": CLOSED_B,
                    "FNV": FNV_B,
                })

                break  # expand one state at a time, per segment

    return multi_best_path.path, multi_expansions, multi_generated, multi_best_path.meet_points
