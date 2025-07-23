import heapq,time
from heuristics.heuristic import heuristic
from models.state import State
from models.heapq_state import HeapqState
from utils.utils import *
import matplotlib.pyplot as plt
from collections import defaultdict


def unidirectional_search(graph, start, goal, heuristic_name, snake, args):
    # For Plotting
    g_degree_pairs = []  # Store (g, degree) for each expanded state

    # Initialize custom priority queue
    open_set = HeapqState()

    # Initial state
    initial_state = State(graph, [start], snake) if isinstance(start, int) else State(graph, start, snake)

    # Initial f_value
    initial_h_value = heuristic(initial_state, goal, heuristic_name, snake)
    initial_f_value = initial_state.g + initial_h_value

    # Push initial state with priority based on f_value
    open_set.push(initial_state, initial_f_value)

    # Basic symmetry detection - a dictionary with the key (head,nodes)
    FNV = {(initial_state.head,initial_state.path_vertices_bitmap)}

    # The best path found
    best_path = None
    best_path_length = -1

    # Expansion counter, generated counter
    expansions = 0
    generated = 0

    while len(open_set) > 0:
        # Pop the state with the highest priority (g(N) + h(N))
        f_value, g_value, current_state = open_set.pop()
        current_path_length = len(current_state.path) - 1

        # Increment the expansion counter
        expansions += 1
        # if expansions % 10000 == 0:
        #     # print(f"Expansion #{expansions}: state {current_state.path}, f={f_value}, len={len(current_state.path)}")
        #     with open(args.log_file_name, 'a') as file:
        #         file.write(f"\nExpansion #{expansions}: state {current_state.path}, f={f_value}, len={len(current_state.path)}")


        # Check if the current state is the goal state
        if current_state.head == goal:
            if current_path_length > best_path_length:
                best_path = current_state.path
                best_path_length = current_path_length
                # if snake:
                #     print(f"[{time2str(args.start_time,time.time())} expansion {expansions}, {time_ms(args.start_time,time.time())}] Found path of length {best_path_length}. {best_path}. generated: {generated}")
                #     with open(args.log_file_name, 'a') as file:
                #         file.write(f"[{time2str(args.start_time,time.time())} expansion {expansions}] Found path of length {best_path_length}. {best_path}\n")

            continue

        # Finish if the f_value is smaller than the best path length found so far
        if f_value <= best_path_length:
            # print(f"apparently we won't find a path longer than {best_path_length}")
            break

        # Generate successors
        successors = current_state.successor(args, snake, True)

        # For Plotting
        g_degree_pairs.append((current_state.g, len(successors)))

        for successor in successors:
            if args.bsd and (successor.head, successor.path_vertices_bitmap) in FNV:
                # print(f"symmetric state removed: {successor.path}")
                continue

            generated += 1
            
            # Check if successor reached the goal
            if successor.head == goal:
                h_successor = 0
                if current_path_length > best_path_length:
                    best_path = successor.path
                    best_path_length = current_path_length
                    if f_value <= best_path_length:
                        break
            else:
                # Calculate the heuristic value
                h_successor = heuristic(successor, goal, heuristic_name, snake)
            # Calculate the g_successor
            g_successor = successor.g
            # Calculate the f_value
            f_successor = g_successor + h_successor
            # Push the successor to the priority queue with the priority as - (g(N) + h(N))
            open_set.push(successor, min(f_successor, f_value))
            FNV.add((successor.head,successor.path_vertices_bitmap))

    # For Plotting
    # g_values = [pair[0] for pair in g_degree_pairs]
    # degrees = [pair[1] for pair in g_degree_pairs]
    # g_bins = defaultdict(list)
    # for g, degree in g_degree_pairs:
    #     g_bins[g].append(degree)
    # sorted_g = sorted(g_bins.keys())
    # avg_degrees = [np.mean(g_bins[g]) for g in sorted_g]
    # std_devs = [np.std(g_bins[g]) for g in sorted_g]
    # plt.figure(figsize=(10, 6))
    # plt.errorbar(sorted_g, avg_degrees, yerr=std_devs, fmt='o-', capsize=5)
    # plt.xlabel("g value (path cost from start)")
    # plt.ylabel("Average degree ± std (number of successors)")
    # plt.title("Average Degree vs. g value with Standard Deviation")
    # plt.grid(True)
    # plt.scatter(g_values, degrees, color='red', marker='*', alpha=0.6, label='Raw data')
    # plt.savefig("avg_std_BF_vs_g_"+args.log_file_name.replace("results_","")+".png")

    return best_path, expansions, generated
