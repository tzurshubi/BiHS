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
from utils.utils import *
# from sage.graphs.connectivity import TriconnectivitySPQR
# from sage.graphs.graph import Graph


# Define default input values
# --date 4_8_24 --number_of_graphs 1 --graph_type grid --size_of_graphs 6 6 --run_uni
DEFAULT_LOG = True                  # True # False
DEFAULT_DATE = "cubes"              # "SM_Grids" / "cubes" / "mazes" / "Check_Sparse_Grids"
DEFAULT_NUMBER_OF_GRAPHS = 1        # 10
DEFAULT_GRAPH_TYPE = "cube"         # "grid" / "cube" / "manual" / "maze"
DEFAULT_SIZE_OF_GRAPHS = [7,7]      # dimension of cube
DEFAULT_PER_OF_BLOCKS = 16          # 4 / 8 / 12 / 16
DEFAULT_HEURISTIC = "bcc_heuristic" # "bcc_heuristic" / "mis_heuristic" / "heuristic0" / "reachable_heuristic" / "bct_is_heuristic" /
DEFAULT_SNAKE = True                # True # False
DEFAULT_RUN_UNI = True             # True # False
DEFAULT_RUN_BI = True               # True # False
DEFAULT_RUN_MULTI = False           # True # False
DEFAULT_SOLUTION_VERTICES = []    # [] # for multidirectional search on cubes
DEFAULT_ALGO = "full"               # "basic" # "light" # "full"
DEFAULT_BSD = True                  # True # False
DEFAULT_CUBE_FIRST_DIMENSIONS = 7   # 3 # 4 # 5 # 6 # 7

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
    # print(f"*SEARCH* Graph Name: {name_of_graph}, Graph Size: {args.size_of_graphs}, Start: {start}, Goal: {goal}, Search Type: {search_type}, Heuristic: {heuristic}")
    # Load the graph
    # print("tzsh:"+current_directory+base_dir+"data/graphs/" + name_of_graph.replace(" ", "_") + ".json")
    G = load_graph_from_file(current_directory+base_dir+"data/graphs/" + name_of_graph.replace(" ", "_") + ".json")
    G_original = G.copy()
    if args.graph_type=="cube":
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

    blocks = []
    logs = {}

    for node in range(args.size_of_graphs[0] * args.size_of_graphs[1]):
        if node not in G:
            blocks.append(node)

    if isinstance(goal,int):
        while goal not in G:
            goal -= 1
    if isinstance(start,int):
        while start not in G:
            start += 1

    meet_point = None
    meet_points = None
    g_values = []
    tracemalloc.start()
    start_time = time.time()
    args.start_time = start_time

    # Run heuristic search to find LSP in the graph
    if search_type == "unidirectional":
        path, expansions, generated = unidirectional_search(G, start, goal, heuristic, snake, args)
    elif search_type == "bidirectional":
        path, expansions, generated, moved_OPEN_to_AUXOPEN, meet_point, g_values = bidirectional_search(G, start, goal, heuristic, snake, args)
    elif search_type == "multidirectional":
        path, expansions, generated, meet_points = multidirectional_search(G, start, goal, args.solution_vertices, heuristic, snake, args)

    # Collect logs
    end_time = time.time()
    memory_snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()
    logs["memory[kB]"] = math.floor(
        sum(stat.size for stat in memory_snapshot.statistics("lineno")) / 1024
    )
    logs["time[ms]"] = math.floor(1000 * (end_time - start_time))
    logs["expansions"] = expansions
    logs["generated"] = generated
    if search_type == "bidirectional":
        logs["moved_OPEN_to_AUXOPEN"] = moved_OPEN_to_AUXOPEN
        logs["g_values"] = g_values

    # Save the graph as PNG with the path if found
    meet_points = meet_points if meet_points else [meet_point]
    if not meet_points is None and not meet_points[0] is None:
        if args.graph_type=="grid":
            save_table_as_png(
                args.size_of_graphs[0],
                args.size_of_graphs[1],
                blocks,
                current_directory+base_dir+"data/graphs/" + name_of_graph.replace(" ", "_") + "_solved.png",
                path,
                meet_points,
            )
        elif args.graph_type=="cube":
            display_graph_with_path_and_points(G_original,"",current_directory+base_dir+"data/graphs/" + name_of_graph.replace(" ", "_") + "_solved.png",path,[meet_point])
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
    bsd = args.bsd
    cube_first_dims = args.cube_first_dims

    if log:
        log_file_name = "logs"
        if graph_type=="cube":
            log_file_name = f"results_{size_of_graphs[0]}d_cube_{heuristic}{"_snake" if snake else ""}{"_uni" if run_uni else ""}{"_bi" if run_bi else ""}{"_multi" if run_multi else ""}"
        else:
            log_file_name = f"results_{size_of_graphs[0]}x{size_of_graphs[1]}_{graph_type}_{per_blocked}per_blocked_{heuristic}{"_snake" if snake else ""}{"_uni" if run_uni else ""}{"_bi" if run_bi else ""}{"_multi" if run_multi else ""}"
        with open(log_file_name, 'w') as file:
            file.write(f"-------------\ndate: {date}\nnumber_of_graphs:{number_of_graphs}\ngraph_type:{graph_type}\nsize_of_graphs:{size_of_graphs}\nheuristic:{heuristic}\nsnake:{snake}\nrun_uni:{run_uni}\nrun_bi:{run_bi}\nrun_multi:{run_multi}\n-------------\n\n")
        args.log_file_name = log_file_name

    print("--------------------------")
    print("date:", date)
    print("number_of_graphs:", number_of_graphs)
    print("graph_type:", graph_type)
    print("size_of_graphs:", size_of_graphs)
    print("heuristic:", heuristic)
    print("snake:", snake)
    print("run_uni:", run_uni)
    print("run_bi:", run_bi)
    print("run_multi:", run_multi)
    print("cube_first_dims:", cube_first_dims)

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
            start = 2**cube_first_dims - 1 # 7 # 15 # 31
            goal = 0
            # log_file_name = "results_"+name_of_graph+"_"+heuristic[:3]
        elif graph_type=="manual":
            name_of_graph = f"paper_graph_{i}"
            start = "s"  # 0 # "s"
            goal = "t"  # size_of_graphs[0] * size_of_graphs[1] - 1  # "t"

        args.log_file_name = log_file_name
        name_of_graph=f"{date}/"+name_of_graph
        print("\n"+name_of_graph)
        if log:
            with open(log_file_name, 'w' if i==0 else 'a') as file:
                file.write("\n"+name_of_graph)            
        
        # unidirectional
        if run_uni:
            # unidirectional s-t
            logs, path, _ = search(
                name_of_graph, start, goal, "unidirectional", heuristic, snake, args
            )
            avgs["uni_st"]["expansions"].append(logs['expansions'])
            avgs["uni_st"]["time"].append(logs['time[ms]'])
            print(f"! unidirectional s-t. expansions: {logs['expansions']:,}, time: {logs['time[ms]']:,} [ms], memory: {logs['memory[kB]']:,} [kB], path length: {len(path)-1:,} [edges], generated: {logs['generated']}")
            if log:
                with open(log_file_name, 'w' if i==0 else 'a') as file:
                    file.write(f"\n! unidirectional s-t. expansions: {logs['expansions']:,}, time: {logs['time[ms]']:,} [ms], memory: {logs['memory[kB]']:,} [kB], path length: {len(path)-1:,} [edges], generated: {logs['generated']}")
            results.append(
                {
                    "# blocks": i,
                    "Search type": "unidirectional s-t",
                    "# expansions": logs["expansions"],
                    "Time [ms]": logs["time[ms]"],
                    "Memory [kB]": logs["memory[kB]"],
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
                print(f"! unidirectional t-s. expansions: {logs['expansions']:,}, time: {logs['time[ms]']:,} [ms], memory: {logs['memory[kB]']:,} [kB], path length: {len(path)-1:,} [edges], generated: {logs['generated']}")
                if log:
                    with open(log_file_name, 'a') as file:
                        file.write(f"\n! unidirectional t-s. expansions: {logs['expansions']:,}, time: {logs['time[ms]']:,} [ms], memory: {logs['memory[kB]']:,} [kB], path length: {len(path)-1:,} [edges], generated: {logs['generated']}")
                results.append(
                    {
                        "# blocks": i,
                        "Search type": "unidirectional t-s",
                        "# expansions": logs["expansions"],
                        "Time [ms]": logs["time[ms]"],
                        "Memory [kB]": logs["memory[kB]"],
                        "[g_F,g_B]": "[N/A,N/A]",
                        "Grid with Solution": "file_path_here",  # Update with actual file path if needed
                    }
                )

        # bidirectional
        if run_bi:
            logs, path, meet_point = search(
                name_of_graph, start, goal, "bidirectional", heuristic, snake,args
            )
            avgs["bi"]["expansions"].append(logs['expansions'])
            avgs["bi"]["time"].append(logs['time[ms]'])
            print(f"! bidirectional. expansions: {logs['expansions']:,}, time: {logs['time[ms]']:,} [ms], path length: {len(path)-1:,} [edges], g_F: {logs['g_F']:,}, g_B: {logs['g_B']:,}, generated: {logs['generated']}, MM: {abs(logs['g_F']-logs['g_B'])}, MMPER: {100*abs(logs['g_F']-logs['g_B'])/len(path)-1}%") # , memory: {logs['memory[kB]']:,} [kB] , moved_OPEN_to_AUXOPEN:{logs['moved_OPEN_to_AUXOPEN']}
            print(f"expanded states with g over C*/2 ({(len(path)-1)/2}): {len([s for s in logs["g_values"] if s > (len(path)-1)/2])}")
            if log:
                with open(log_file_name, 'a') as file:
                    file.write(f"\n! bidirectional. expansions: {logs['expansions']:,}, time: {logs['time[ms]']:,} [ms], memory: {logs['memory[kB]']:,} [kB], path length: {len(path)-1:,} [edges], g_F: {logs['g_F']:,}, g_B: {logs['g_B']:,}, generated: {logs['generated']}, moved_OPEN_to_AUXOPEN:{logs['moved_OPEN_to_AUXOPEN']}\n\n")
            results.append(
                {
                    "# blocks": i,
                    "Search type": "bidirectional",
                    "# expansions": logs["expansions"],
                    "Time [ms]": logs["time[ms]"],
                    "Memory [kB]": logs["memory[kB]"],
                    "[g_F,g_B]": f"[{logs['g_F']},{logs['g_B']}]",
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
            print(f"! multidirectional. expansions: {logs['expansions']:,}, time: {logs['time[ms]']:,} [ms], path length: {len(path)-1:,} [edges], generated: {logs['generated']}") # , memory: {logs['memory[kB]']:,} [kB] , moved_OPEN_to_AUXOPEN:{logs['moved_OPEN_to_AUXOPEN']}
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
