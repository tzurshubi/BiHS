import os
# import pandas as pd
# import networkx as nx
import argparse
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
from algorithms.multidirectional_search import *
from algorithms.multidirectional_search1 import *
from algorithms.tbt_search import *
from algorithms.bidirectional_search_sym_coil import *
from algorithms.unidirectional_search_sym_coil import *
from algorithms.unidirectional_gradual_sym_coil import *
from algorithms.bidirectional_dfbnb_sym_coil import *
from algorithms.bidirectional_gradual_sym_coil import *
from utils.utils import *
# from sage.graphs.connectivity import TriconnectivitySPQR
# from sage.graphs.graph import Graph


# Define default input values
# --date 4_8_24 --number_of_graphs 1 --graph_type grid --size_of_graphs 6 6 --run_uni
DEFAULT_LOG = True                      # True # False
DEFAULT_DATE = "cubes"                  # "SM_Grids" / "cubes" / "mazes" / "Check_Sparse_Grids"
DEFAULT_NUMBER_OF_GRAPHS = 1            # 10
DEFAULT_GRAPH_TYPE = "cube"             # "grid" / "cube" / "manual" / "maze"
DEFAULT_SIZE_OF_GRAPHS = [7,7]          # dimension of cube
DEFAULT_PER_OF_BLOCKS = 16              # 4 / 8 / 12 / 16
DEFAULT_HEURISTIC = "heuristic0"        # "bcc_heuristic" / "mis_heuristic" / "heuristic0" / "reachable_heuristic" / "bct_is_heuristic" /
DEFAULT_SNAKE = True                    # True # False
DEFAULT_RUN_UNI = True                 # True # False
DEFAULT_RUN_BI = False                   # True # False
DEFAULT_RUN_MULTI = False               # True # False
DEFAULT_SOLUTION_VERTICES = [64]        # [] # for multidirectional search on cubes # 60 is good mean for 7d cube symcoil
DEFAULT_ALGO = "basic"                  # "basic" # "light" # "cutoff" # "full"
DEFAULT_BSD = True                      # True # False
DEFAULT_CUBE_FIRST_DIMENSIONS = 4       # 3 # 4 # 5 # 6 # 7
DEFAULT_CUBE_BUFFER_DIMENSION = None    # None # 3 # 4 # 5 # 6 # 7
DEFAULT_BACKWARD_SYM_GENERATION = False # True # False
DEFAULT_SYM_COIL = True                # True # False
DEFAULT_PREFIX_SET = 4               # None # 2 # 3 # 4 # comparing sets of states with same prefix of length k-3

base_dir = "/"
current_directory = os.getcwd()
if current_directory.startswith("/cs_storage/") or current_directory.startswith("/mnt/"):
    base_dir = "/BiHS/"
if current_directory.startswith("/home/") and "tzur-shubi" not in current_directory:
    base_dir = "/BiHS/"


# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run graph search experiments.")
    parser.add_argument("--log", type=str, default=DEFAULT_LOG, help="Date for naming files.")
    parser.add_argument("--date", type=str, default=DEFAULT_DATE, help="Date for naming files.")
    parser.add_argument("--number_of_graphs", type=int, default=DEFAULT_NUMBER_OF_GRAPHS, help="Number of graphs to process.")
    parser.add_argument("--graph_type", type=str, default=DEFAULT_GRAPH_TYPE, help="Type of graph: grid, cube, manual, maze.")
    parser.add_argument("--size_of_graphs", nargs=2, type=int, default=DEFAULT_SIZE_OF_GRAPHS, help="Size of graphs (e.g., 8 8).")
    parser.add_argument("--per_blocked", type=int, default=DEFAULT_PER_OF_BLOCKS, help="Number of graphs to process.")
    parser.add_argument("--heuristic", type=str, default=DEFAULT_HEURISTIC, help="Heuristic to use: bcc_heuristic, reachable_heuristic, etc.")
    parser.add_argument("--snake", action="store_true", default=DEFAULT_SNAKE, help="Enable snake mode.")
    parser.add_argument("--run_uni", action="store_true", default=DEFAULT_RUN_UNI, help="Enable snake mode.")
    parser.add_argument("--run_bi", action="store_true", default=DEFAULT_RUN_BI, help="Enable snake mode.")
    parser.add_argument("--run_multi", action="store_true", default=DEFAULT_RUN_MULTI, help="Enable snake mode.")
    parser.add_argument("--solution_vertices", nargs='+', type=int, default=DEFAULT_SOLUTION_VERTICES, help="Solution vertices for multidirectional search.")
    parser.add_argument("--algo", type=str, default=DEFAULT_ALGO, help="Algo to use: basic, light, full")
    parser.add_argument("--bsd", type=str, default=DEFAULT_BSD, help="Basic Symmetry Detection")
    parser.add_argument("--cube_first_dims", type=int, default=DEFAULT_CUBE_FIRST_DIMENSIONS, help="Number of initial dimensions crossed.")
    parser.add_argument("--cube_buffer_dim", type=int, default=DEFAULT_CUBE_BUFFER_DIMENSION, help="Buffer dimension for cube graphs.")
    parser.add_argument("--backward_sym_generation", type=str, default=DEFAULT_BACKWARD_SYM_GENERATION, help="Symmetrical generation in other frontier.")
    parser.add_argument("--sym_coil", type=str, default=DEFAULT_SYM_COIL, help="Find symmetrical coil.")
    parser.add_argument("--prefix_set", type=int, default=DEFAULT_PREFIX_SET, help="Comparing sets of states with same prefix of length k-3.")
    return parser.parse_args()


# This function is only a copy! the original is in create_graph.py
def save_table_as_png(
    rows, cols, black_cells, filename="grid.png", path=None, points=None
):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(cols, rows))

    # Turn off axis
    ax.axis("off")

    # Create a table of all white cells
    grid = np.full((rows, cols), 255)

    # Make specified cells black if black_cells is not empty
    if black_cells:
        for cell in black_cells:
            grid[cell // cols, cell % cols] = 0

    # Plot the grid
    ax.imshow(grid, cmap="gray", aspect="auto", vmin=0, vmax=255)

    # Add borders to the table and cells
    for i in range(rows):
        for j in range(cols):
            # Create a rectangle patch for each cell
            rect = patches.Rectangle(
                (j - 0.5, i - 0.5),
                1,
                1,
                linewidth=1,
                edgecolor="black",
                facecolor="none",
            )
            ax.add_patch(rect)

    # Add borders around the entire table
    outer_rect = patches.Rectangle(
        (-0.5, -0.5), cols, rows, linewidth=2, edgecolor="black", facecolor="none"
    )
    ax.add_patch(outer_rect)

    # Add text to each cell
    for i in range(rows):
        for j in range(cols):
            cell_index = i * cols + j
            color = "white" if grid[i, j] == 0 else "black"
            ax.text(
                j,
                i,
                str(cell_index),
                va="center",
                ha="center",
                color=color,
                fontsize=10,
            )

    # Add the path if provided
    if path:
        path_coords = [(cell % cols, cell // cols) for cell in path]
        path_x, path_y = zip(*path_coords)
        ax.plot(path_x, path_y, color="red", linewidth=2)

    # Add points if provided
    if points and points[0]:
        for point in points:
            point_x, point_y = point % cols, point // cols
            ax.plot(point_x, point_y, "bo", markersize=20)  # Blue dot

    # Save the figure
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

# This function is only a copy! the original is in create_graph.py
def display_graph_with_path_and_points(
    graph, title="Graph", filename=None, path=None, points=None
):
    """
    Display a graph with optional highlighted path and points.
    
    Args:
        graph (networkx.Graph): The graph to display.
        title (str): Title of the graph.
        filename (str, optional): If provided, saves the graph as a PNG to the given filename.
        path (list, optional): List of vertices to highlight as a path. Their edges will be blue.
        points (list, optional): List of vertices to highlight in red.
    """
    cube = "cube" in filename.lower()  # Simple check to see if the graph is a cube
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(graph)

    # Default node colors
    # node_colors = [
    #     "orange" if node in (points or []) else "skyblue" for node in graph.nodes()
    # ]
    # node_colors = [
    #     "skyblue" if node in (points or []) else "skyblue" for node in graph.nodes()
    # ]
    node_colors = [
        "green" if node in (0, 1, 3, 7) else "orange" if node in (points or []) else "skyblue"
        for node in graph.nodes()
    ]



    # Build set of path edges
    path_edges = set(zip(path or [], (path or [])[1:]))
    path_edges |= {(v, u) for (u, v) in path_edges}  # undirected

    # Special cube edges to make green
    special_cube_edges = {(0, 1), (1, 3), (3, 7)}
    special_cube_edges |= {(b, a) for (a, b) in special_cube_edges}  # both directions

    edge_colors = []
    edge_widths = []
    for u, v in graph.edges():
        if cube and (u, v) in special_cube_edges:
            color, width = "green", 4  # "green" , 2
        elif (u, v) in path_edges:
            color, width = "red", 4  # 2
        else:
            color, width = "gray", 1
        edge_colors.append(color)
        edge_widths.append(width)

    # Draw the graph
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=500,
        node_color=node_colors,
        font_size=10,
        font_weight="bold",
        edge_color=edge_colors,
        width=edge_widths,
    )
    plt.title(title)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def search(
    name_of_graph,
    start,
    goal,
    search_type,
    heuristic,
    snake,
    args
):
    # Load the graph
    G = load_graph_from_file(current_directory+base_dir+"data/graphs/" + name_of_graph.replace(" ", "_") + ".json")
    args.graph_image_path = current_directory+base_dir+"data/graphs/" + name_of_graph.replace(" ", "_") + "_solved.png"
    
    # Remove nodes and edges from the graph
    G_original = G.copy()
    if args.graph_type=="cube" and cube_first_dims and args.sym_coil and search_type=="unidirectional":
        if G.has_edge(0, 1): G.remove_edge(0, 1)

        # ----------------------------
        # Remove N(S) except keepers
        # ----------------------------
        verts = [(2**k - 1) for k in range(0, cube_first_dims)]
        verts_set = set(verts)
        keepers = {2**cube_first_dims - 1}

        neighbors = set()
        for v in verts:
            neighbors |= set(G.neighbors(v))

        G.remove_nodes_from((neighbors | verts_set) - keepers)

        # ----------------------------
        # Remove T path (t -> traverse dims 0..d-1 once, in order) and N(T) except keepers
        # ----------------------------
        t = int(args.solution_vertices[0])
        d = int(cube_first_dims)
        keepers = {t}

        # Path vertices: v0=t, v1=t^2^0, v2=t^2^0^2^1, ..., v_d=t^(2^0^...^2^(d-1))
        # T_path vertices: v1=t^2^0, v2=t^2^0^2^1, ..., v_d=t^(2^0^...^2^(d-1))
        T_path = []
        v = t
        vertex_symmetric_to_start = t
        for i in range(d):
            v ^= (1 << i)
            T_path.append(v)
            vertex_symmetric_to_start = v

        T_set = set(T_path)

        # N(T): union of neighbors of all vertices in the path (in the current G)
        N_T = set()
        for x in T_set:
            if G.has_node(x):
                N_T |= set(G.neighbors(x))

        # Remove T ∪ N(T)
        G.remove_nodes_from((T_set | N_T) - keepers)

        # ----------------------------
        # If buffer dimension is defined: keep only vertices with BD bit = 1
        # ----------------------------
        if args.cube_buffer_dim is not None:
            bd = int(args.cube_buffer_dim)
            mask = 1 << bd

            # collect first, then remove (safe while iterating)
            to_remove = [v for v in G.nodes() if (int(v) & mask) == 0]
            G.remove_nodes_from(to_remove)


        # The list of dimension-swap pairs used to mirror the first `cube_first_dims` dimensions of a hypercube.
        args.dim_swaps_F_B_symmetry = [] # [(i, cube_first_dims - 1 - i) for i in range(cube_first_dims // 2)]
        args.dim_flips_F_B_symmetry = list(range(args.size_of_graphs[0]))
        args.cube_first_dims_path = verts
        args.vertex_symmetric_to_start = vertex_symmetric_to_start
    if args.graph_type=="cube" and cube_first_dims and not args.sym_coil:
        if G.has_edge(0, 1): G.remove_edge(0, 1)
        # G.remove_nodes_from((set(G.neighbors(1)) | set(G.neighbors(3))) - {0, 7})
        # G.remove_nodes_from((set(G.neighbors(1)) | set(G.neighbors(3)) | set(G.neighbors(7))) - {0, 15})
        #G.remove_nodes_from((set(G.neighbors(1)) | set(G.neighbors(3)) | set(G.neighbors(7)) | set(G.neighbors(15))) - {0, 31})

        # Compute the vertices: 1, 3, 7, 15, ... (2^k - 1)
        verts = [(2**k - 1) for k in range(1, cube_first_dims)]

        # Compute the end vertex to keep: 0 and last vertex
        keepers = {0, 2**cube_first_dims - 1}

        # Union of all neighbors of these vertices
        neighbors = set()
        for v in verts:
            neighbors |= set(G.neighbors(v))

        # Remove all except keepers
        G.remove_nodes_from(neighbors - keepers)

        # The list of dimension-swap pairs used to mirror the first `cube_first_dims` dimensions of a hypercube.
        args.dim_swaps_F_B_symmetry = [(i, cube_first_dims - 1 - i) for i in range(cube_first_dims // 2)]
        args.dim_flips_F_B_symmetry = list(range(cube_first_dims))
    if args.graph_type=="cube" and cube_first_dims and args.sym_coil and search_type=="bidirectional":
        if G.has_edge(0, 1): G.remove_edge(0, 1)

        # Compute the vertices: 1, 3, 7, 15, ... (2^k - 1)
        verts = [(2**k - 1) for k in range(1, cube_first_dims)]

        # Compute the end vertex to keep: 0 and last vertex
        keepers = {0, 2**cube_first_dims - 1}
        keepers_list = list(keepers)

        # Union of all neighbors of these vertices
        neighbors = set()
        for v in verts:
            neighbors |= set(G.neighbors(v))

        # Remove all except keepers
        G.remove_nodes_from(neighbors - keepers)

        # The list of dimension-swap pairs used to mirror the first `cube_first_dims` dimensions of a hypercube.
        args.dim_swaps_F_B_symmetry = [] # [(i, cube_first_dims - 1 - i) for i in range(cube_first_dims // 2)]
        args.dim_flips_F_B_symmetry = list(range(args.size_of_graphs[0]))
        args.cube_first_dims_path = [keepers_list[0]]+verts
    
    blocks = []
    logs = {}

    for node in range(args.size_of_graphs[0] * args.size_of_graphs[1]):
        if node not in G:
            blocks.append(node)
    if args.graph_type=="grid":
        if isinstance(goal,int):
            while goal not in G:
                goal -= 1
        if isinstance(start,int):
            while start not in G:
                start += 1

    meet_point = None
    meet_points = None
    tracemalloc.start()
    start_time = time.time()
    args.start_time = start_time
    args.logger.set_t0(start_time)
    args.start = start
    args.goal = goal

    # Run heuristic search to find LSP in the graph
    if search_type == "unidirectional":
        # print(f"\nUnidirectional search on graph '{name_of_graph}' from {start} to {goal} with heuristic '{heuristic}' {'in SNAKE mode' if snake else ''}")
        if not args.sym_coil:
            path, stats = unidirectional_search(G, start, goal, heuristic, snake, args)
        else: # if args.sym_coil:
            # path, stats = unidirectional_search_sym_coil(G, start, goal, heuristic, snake, args)
            path, stats = unidirectional_gradual_sym_coil(G, start, goal, heuristic, snake, args)
    elif search_type == "bidirectional":
        # print(f"\nBidirectional search on graph '{name_of_graph}' from {start} to {goal} with heuristic '{heuristic}' {'in SNAKE mode' if snake else ''}")
        if not args.sym_coil:
            path, stats, meet_point = bidirectional_search(G, start, goal, heuristic, snake, args)
        else: # if args.sym_coil:
            # path, stats, meet_point = bidirectional_search_sym_coil(G, start, goal, heuristic, snake, args)
            # path, stats = bidirectional_dfbnb_sym_coil(G, start, goal, heuristic, snake, args)
            path, stats = bidirectional_gradual_sym_coil(G, start, goal, heuristic, snake, args)

    elif search_type == "multidirectional":
        # print(f"\nMultidirectional search on graph '{name_of_graph}' from {start} to {goal} with heuristic '{heuristic}' {'in SNAKE mode' if snake else ''}")
        # path, expansions, generated, meet_points = multidirectional_search1(G, start, goal, args.solution_vertices, heuristic, snake, args)
        # print("\n\nRunning TBT search as multidirectional!")
        # time.sleep(1)
        path, stats = tbt_search(G, start, goal, heuristic, snake, args)
        

    if path and not isinstance(path,list):
        path_state = path
        path = path_state.materialize_path()
    if not path: args.logger("No path found.")

    # Collect logs
    end_time = time.time()
    logs["time[ms]"] = math.floor(1000 * (end_time - start_time))
    logs.update(stats)


    excluded = {"g_values", "BF_values"}
    filtered_logs = {k: v for k, v in logs.items() if k not in excluded}
    args.logger(f"LOGS: {filtered_logs}")


    # Save the graph as PNG with the path if found
    meet_points = meet_points if meet_points else [meet_point]
    if not meet_points is None and not meet_points[0] is None:
        if args.graph_type=="grid":
            save_table_as_png(
                args.size_of_graphs[0],
                args.size_of_graphs[1],
                blocks,
                args.graph_image_path,
                path,
                meet_points,
            )
        elif args.graph_type=="cube":
            display_graph_with_path_and_points(G_original,"",args.graph_image_path,path,[meet_point])
        meet_point_index = path.index(meet_points[0])
        logs["g_F"] = len(path[:meet_point_index])
        logs["g_B"] = len(path[meet_point_index + 1 :])

    # print(path)
    # print(
    #     f"Longest Simple Path from {start} to {goal}: {path} with length={len(path)-1}"
    # )
    # print("Number of expansions:", expansions)
    return logs, path, meet_point


# Main script
if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    # Assign values from arguments or defaults
    log = args.log
    date = args.date
    number_of_graphs = args.number_of_graphs
    graph_type = args.graph_type
    size_of_graphs = args.size_of_graphs
    per_blocked = args.per_blocked
    heuristic = args.heuristic
    snake = args.snake
    run_uni = args.run_uni
    run_bi = args.run_bi
    run_multi = args.run_multi
    solution_vertices = args.solution_vertices
    algo = args.algo
    bsd = args.bsd
    cube_first_dims = args.cube_first_dims
    cube_buffer_dim = args.cube_buffer_dim
    backward_sym_generation = args.backward_sym_generation
    sym_coil = args.sym_coil
    prefix_set = args.prefix_set

    if log:
        log_file_name = "logs"
        if graph_type=="cube":
            log_file_name = f"results_{size_of_graphs[0]}d_cube_{heuristic}{"_snake" if snake else ""}{"_uni" if run_uni else ""}{"_bi" if run_bi else ""}{"_multi" if run_multi else ""}"
        else:
            log_file_name = f"results_{size_of_graphs[0]}x{size_of_graphs[1]}_{graph_type}_{per_blocked}per_blocked_{heuristic}{"_snake" if snake else ""}{"_uni" if run_uni else ""}{"_bi" if run_bi else ""}{"_multi" if run_multi else ""}"
        if cube_buffer_dim is not None:
            log_file_name += f"_buffDim{cube_buffer_dim}"
        if backward_sym_generation:
            log_file_name += f"_symGen"
        if sym_coil:
            log_file_name += f"_symcoil"
        if args.bsd:
            log_file_name += "_BSD"
        if args.cube_first_dims is not None:
            log_file_name += f"_{cube_first_dims}DDS"
        if prefix_set is not None:
            log_file_name += f"_prefix{prefix_set}"
        if solution_vertices is not None and len(solution_vertices)>0:
            log_file_name += "_solVert"+"_".join([str(v) for v in solution_vertices])
        args.log_file_name = log_file_name
        args.logger = make_logger(open(log_file_name, "w"))
    
        args.logger("--------------------------")
        args.logger(f"date: {date}")
        args.logger(f"number_of_graphs: {number_of_graphs}")
        args.logger(f"graph_type: {graph_type}")
        args.logger(f"size_of_graphs: {size_of_graphs}")
        args.logger(f"heuristic: {heuristic}")
        args.logger(f"snake: {snake}")
        args.logger(f"run_uni: {run_uni}")
        args.logger(f"run_bi: {run_bi}")
        args.logger(f"run_multi: {run_multi}")
        args.logger(f"solution_vertices: {solution_vertices}")
        args.logger(f"algo: {algo}")
        args.logger(f"bsd: {bsd}")
        args.logger(f"cube_first_dims: {cube_first_dims}")
        args.logger(f"cube_buffer_dim: {cube_buffer_dim}")
        args.logger(f"backward_sym_generation: {backward_sym_generation}")
        args.logger(f"sym_coil: {sym_coil}")
        args.logger(f"prefix_set: {prefix_set}")


    # Initialize an empty DataFrame to store the results
    columns = [
        "# blocks",
        "Search type",
        "# expansions",
        "Time [ms]",
        "Memory [kB]",
        "[g_F,g_B]",
        "Grid with Solution",
    ]
    # results_df = pd.DataFrame(columns=columns)
    results = []

    avgs={"uni_st": {"expansions":[], "time":[]}, "uni_ts":{"expansions":[], "time":[]}, "bi":{"expansions":[], "time":[]}, "multi":{"expansions":[], "time":[]}}
    for i in list(range(0, number_of_graphs)):
    # for i in range(number_of_graphs, number_of_graphs+1):
        # try:
        # Inputs
        if graph_type=="grid":
            if per_blocked:
                name_of_graph = f"{size_of_graphs[0]}x{size_of_graphs[1]}_grid_with_random_blocks_{per_blocked}per_{i}"
            else: name_of_graph = f"{size_of_graphs[0]}x{size_of_graphs[1]}_grid_with_random_blocks_{i}" # f"paper_graph_{i}" # f"{size_of_graphs[0]}x{size_of_graphs[1]}_grid_with_random_blocks_{i}"
            # log_file_name = "results_"+name_of_graph[:-2]+"_"+heuristic[:3]
            start = 0  # 0 # "s"
            goal = size_of_graphs[0] * size_of_graphs[1] - 1  # size_of_graphs[0] * size_of_graphs[1] - 1  # "t"
        elif graph_type=="maze":
            name_of_graph = f"{size_of_graphs[0]}x{size_of_graphs[1]}_maze_with_blocks_and_random_removals_{i}" # f"paper_graph_{i}" # f"{size_of_graphs[0]}x{size_of_graphs[1]}_grid_with_random_blocks_{i}"
            start = 0  # 0 # "s"
            goal = size_of_graphs[0] * size_of_graphs[1] - 1  # size_of_graphs[0] * size_of_graphs[1] - 1  # "t"
            # log_file_name = "results_"+name_of_graph[:-2]+"_"+heuristic[:3]
        elif graph_type=="cube":
            # if i<3: continue
            name_of_graph=f"{size_of_graphs[0]}d_cube" # hypercube
            start = 2**cube_first_dims-1 if cube_first_dims is not None else 2**size_of_graphs[0]-1 # 7 # 15 # 31
            goal = 0
            if args.sym_coil:
                start = 2**cube_first_dims-1 if cube_first_dims is not None else 0 # 7 # 15 # 31
                goal = args.solution_vertices[0]
        elif graph_type=="manual":
            name_of_graph = f"paper_graph_{i}"
            start = "s"  # 0 # "s"
            goal = "t"  # size_of_graphs[0] * size_of_graphs[1] - 1  # "t"

        name_of_graph=f"{date}/"+name_of_graph
        args.logger("\n---------- "+name_of_graph+" ----------")
           
        
        # unidirectional
        if run_uni:
            # unidirectional s-t
            logs, path, _ = search(
                name_of_graph, start, goal, "unidirectional", heuristic, snake, args
            )
            avgs["uni_st"]["expansions"].append(logs['expansions'])
            avgs["uni_st"]["time"].append(logs['time[ms]'])
            if path: args.logger(f"! Unidirectional s-t. expansions: {logs['expansions']:,}, time: {logs['time[ms]']:,} [ms], path length: {len(path)-1:,} [edges], generated: {logs['generated']}")
            else:    args.logger(f"! Unidirectional s-t. expansions: {logs['expansions']:,}, time: {logs['time[ms]']:,} [ms], generated: {logs['generated']}")
            results.append(
                {
                    "# blocks": i,
                    "Search type": "unidirectional s-t",
                    "# expansions": logs["expansions"],
                    "Time [ms]": logs["time[ms]"],
                    "[g_F,g_B]": "[N/A,N/A]",
                    "Grid with Solution": "file_path_here",  # Update with actual file path if needed
                }
            )

            # unidirectional t-s
            if graph_type!="cube":
                logs, path, _ = search(
                    name_of_graph, goal, start, "unidirectional", heuristic, snake, args
                )
                avgs["uni_ts"]["expansions"].append(logs['expansions'])
                avgs["uni_ts"]["time"].append(logs['time[ms]'])
                args.logger(f"! Unidirectional t-s. expansions: {logs['expansions']:,}, time: {logs['time[ms]']:,} [ms], path length: {len(path)-1:,} [edges], generated: {logs['generated']}")
                results.append(
                    {
                        "# blocks": i,
                        "Search type": "unidirectional t-s",
                        "# expansions": logs["expansions"],
                        "Time [ms]": logs["time[ms]"],
                        "[g_F,g_B]": "[N/A,N/A]",
                        "Grid with Solution": "file_path_here",  # Update with actual file path if needed
                    }
                )

        # bidirectional
        if run_bi:
            logs, path, meet_point = search(
                name_of_graph, start, goal, "bidirectional", heuristic, snake, args
            )
            avgs["bi"]["expansions"].append(logs['expansions'])
            avgs["bi"]["time"].append(logs['time[ms]'])
            # print(f"expanded states with g over C*/2 ({(len(path)-1)/2}): {len([s for s in logs["g_values"] if s > (len(path)-1)/2])}")
            if not args.sym_coil: 
                args.logger(f"! Bidirectional. expansions: {logs['expansions']:,}, time: {logs['time[ms]']:,} [ms], path length: {len(path)-1:,} [edges], g_F: {logs['g_F']:,}, g_B: {logs['g_B']:,}, generated: {logs['generated']}, MM: {abs(logs['g_F']-logs['g_B'])}, MMPER: {100*abs(logs['g_F']-logs['g_B'])/len(path)-1}%") # , memory: {logs['memory[kB]']:,} [kB] , moved_OPEN_to_AUXOPEN:{logs['moved_OPEN_to_AUXOPEN']}
                results.append(
                {
                    "# blocks": i,
                    "Search type": "bidirectional",
                    "# expansions": logs["expansions"],
                    "Time [ms]": logs["time[ms]"],
                    "[g_F,g_B]": f"[{logs['g_F']},{logs['g_B']}]",
                    "Grid with Solution": "file_path_here",  # Update with actual file path if needed
                }
            )
            else:
                if path:    args.logger(f"! Bidirectional Symmetrical Coil. expansions: {logs['expansions']:,}, time: {logs['time[ms]']:,} [ms], path length: {len(path)-1:,} [edges], generated: {logs['generated']}") #  , moved_OPEN_to_AUXOPEN:{logs['moved_OPEN_to_AUXOPEN']}
                else:       args.logger(f"! Bidirectional Symmetrical Coil. No path found. expansions: {logs['expansions']:,}, time: {logs['time[ms]']:,} [ms], generated: {logs['generated']}") #  , moved_OPEN_to_AUXOPEN:{logs['moved_OPEN_to_AUXOPEN']}
                results.append(
                {
                    "# blocks": i,
                    "Search type": "bidirectional symmetrical coil",
                    "# expansions": logs["expansions"],
                    "Time [ms]": logs["time[ms]"],
                    "Grid with Solution": "file_path_here",  # Update with actual file path if needed
                }
            )

        # multidirectional
        if run_multi:
            if graph_type=="cube":
                args.solution_vertices = [2**args.size_of_graphs[0]-1]
            elif meet_point: args.solution_vertices = [meet_point]
            logs, path, meet_point = search(
                name_of_graph, start, goal, "multidirectional", heuristic, snake,args
            )
            args.logger(f"! Multidirectional. expansions: {logs['expansions']:,}, time: {logs['time[ms]']:,} [ms], path length: {len(path)-1:,} [edges], generated: {logs['generated']}") # , memory: {logs['memory[kB]']:,} [kB] , moved_OPEN_to_AUXOPEN:{logs['moved_OPEN_to_AUXOPEN']}
            results.append(
                {
                    "# blocks": i,
                    "Search type": "multidirectional",          
                    "# expansions": logs["expansions"],
                    "Time [ms]": logs["time[ms]"],
                    # "[g_F,g_B]": f"[{logs['g_F']},{logs['g_B']}]",
                    "Grid with Solution": "file_path_here",  # Update with actual file path if needed
                }
            )

            avgs["multi"]["expansions"].append(logs['expansions'])
            avgs["multi"]["time"].append(logs['time[ms]'])

        # if graph_type == "cube":
        #     os.system('cls' if os.name == 'nt' else 'clear')
        #     print(f"Path: {path}")
        #     cube_dimension = size_of_graphs[0]
        #     # longest_coil = [path[0], 1, 3] + path[-1:0:-1]
        #     longest_coil = [path[-1], 1, 3] + path[0:-1]
        #     path_length = len(longest_coil)
        #     print(f"Longest coil in {cube_dimension}D cube (Length={path_length}): {longest_coil}")
        #     longest_coil_on_bits = node_num_to_bits_on(cube_dimension, longest_coil)
        #     print(longest_coil_on_bits)
        #     print("---------------------------")
        #     for bits_on in node_num_to_bits_on(cube_dimension, longest_coil):
        #         print(bits_on)
        #     print("---------------------------")
        #     print_bit_statistics(longest_coil_on_bits)
        #     exit()

            
    print()
    calculate_averages(avgs, log_file_name)
    parse_results_file(log_file_name, log_file_name+".csv")
    args.logger.close()
