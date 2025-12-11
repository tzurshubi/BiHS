import os
import pandas as pd
import argparse
# import networkx as nx
import json
import random
import traceback
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import time
import math
import tracemalloc
from models.graph import *
from algorithms.unidirectional_search import *
from algorithms.bidirectional_search import *
import pickle
from models.open import Open
from heuristics.heuristic import heuristic
from models.state import State
from models.heapq_state import HeapqState


DEFAULT_DIMENSION = 4
DEFAULT_DIRECTION = "B" # "B"
DEFAULT_HEURISTIC = "bcc_heuristic"  # "heuristic0" / "reachable_heuristic" / "bcc_heuristic" / "mis_heuristic"


base_dir = "/"
current_directory = os.getcwd()
if current_directory.startswith("/cs_storage/"):
    base_dir = "/BiHS/"

# Function to parse command-line arguments
def parse_args():
    """
    Parse command-line arguments to handle only two inputs:
    1. Dimension of the graph.
    2. Direction (e.g., 'F' or 'B').
    """
    parser = argparse.ArgumentParser(description="Run graph search experiments.")
    parser.add_argument("--dimension", type=int, default=DEFAULT_DIMENSION, help="Dimension of the hypercube.")
    parser.add_argument("--direction", type=str, choices=["F", "B"], default=DEFAULT_DIRECTION, help="Search direction: 'F' for forward, 'B' for backward.")
    parser.add_argument("--heuristic", type=str, default=DEFAULT_HEURISTIC, help="Heuristic to use: bcc_heuristic, reachable_heuristic, etc.")

    return parser.parse_args()

# Main script
if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    # Assign values from arguments or defaults
    dimension = args.dimension
    direction = args.direction
    heuristic_name = args.heuristic

    # Print the variables with their names and values
    print("--------------------------")
    print("dimension:", dimension)
    print("direction:", direction)    
    print("heuristic:", heuristic_name)


    name_of_graph=f"{dimension}d_cube"
    (start , goal)= ([0,32,48],1) if direction=="F" else ([1],0)
    name_of_graph=f"cubes/"+name_of_graph
    print(name_of_graph)

    G = load_graph_from_file(current_directory+base_dir+"data/graphs/" + name_of_graph.replace(" ", "_") + ".json")
    if G.has_edge(0, 1):
        G.remove_edge(0, 1)
    open_set = HeapqState()
    open2save = Open(max(G.nodes)+1)
    open_file_name = f"open{direction}"


    # Initial state
    initial_state = State(G, start, None, True)

    # Initial f_value
    initial_f_value = initial_state.g + heuristic(initial_state, goal, heuristic_name, True)

    # Push initial state with priority based on f_value
    open_set.push(initial_state, initial_f_value)
    open2save.insert_state(initial_state)

    # The best path found
    best_path = None
    best_path_length = -1

    # Expansion counter
    expansions = 0

    while len(open_set) > 0:
        # Pop the state with the highest priority (g(N) + h(N))
        _, _, current_state, f_value = open_set.pop()
        current_path_length = len(current_state.path) - 1

        # Increment the expansion counter
        expansions += 1
        if expansions % 100000 == 1 or expansions==50001 or expansions==10001:
            block_start_time = time.time()
            with open(open_file_name, 'wb') as f:
                pickle.dump(open2save, f)
            block_end_time = time.time()
            with open(f"expansions{direction}.txt", 'w') as file:
                file.write(f"after {expansions-1} expansion, saving OPEN took {block_end_time - block_start_time:.4f} seconds\n")
            # print(f"Expansion #{expansions}: state {current_state.path}, f={f_value}, len={len(current_state.path)}")
            

        # Check if the current state is the goal state
        if current_state.head == goal:
            # Update the best path if the current one is longer
            if current_path_length > best_path_length:
                best_path = current_state.path
                best_path_length = current_path_length
                # print(
                #     f"This state has head as the goal, and is the longest found, with length {current_path_length}"
                # )
            continue

        # Finish if the f_value is smaller than the best path length found so far
        if f_value <= best_path_length:
            # print(f"apparently we won't find a path longer than {best_path_length}")
            break

        # Generate successors
        successors = current_state.successor(True)
        for successor in successors:
            # Calculate the heuristic value
            h_value = heuristic(successor, goal, heuristic_name, True)
            # Calculate the g_value
            g_value = current_path_length + 1
            # Calculate the f_value
            f_value = g_value + h_value
            # Push the successor to the priority queue with the priority as - (g(N) + h(N))
            open_set.push(successor, f_value)
            open2save.insert_state(successor)
    print('done')
    with open(open_file_name, 'wb') as f:
        pickle.dump(open2save, f)

