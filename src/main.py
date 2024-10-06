import pandas as pd

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


def search(
    name_of_graph,
    size_of_graphs,
    start,
    goal,
    search_type,
    heuristic,
):
    print(f"Running search with parameters: Graph Name: {name_of_graph}, Graph Size: {size_of_graphs}, Start: {start}, Goal: {goal}, Search Type: {search_type}, Heuristic: {heuristic}")
    # Load the graph
    G = load_graph_from_file("data/graphs/" + name_of_graph.replace(" ", "_") + ".json")
    blocks = []
    logs = {}
    for node in range(size_of_graphs[0] * size_of_graphs[1]):
        if node not in G:
            blocks.append(node)
    while goal not in G:
        goal -= 1
    while start not in G:
        start += 1

    meet_point = None
    tracemalloc.start()
    start_time = time.time()
    # Run heuristic search to find LSP in the graph
    if search_type == "unidirectional":
        path, expansions = uniHS_for_LSP(G, start, goal, heuristic)
    elif search_type == "bidirectional":
        path, expansions, meet_point = biHS_for_LSP(G, start, goal, heuristic)
    end_time = time.time()
    memory_snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()
    logs["memory[kB]"] = math.floor(
        sum(stat.size for stat in memory_snapshot.statistics("lineno")) / 1024
    )
    logs["time[ms]"] = math.floor(1000 * (end_time - start_time))
    logs["expansions"] = expansions

    if meet_point:
        save_table_as_png(
            size_of_graphs[0],
            size_of_graphs[1],
            blocks,
            "data/graphs/" + name_of_graph.replace(" ", "_") + "_solved.png",
            path,
            [meet_point],
        )
        meet_point_index = path.index(meet_point)
        logs["g_F"] = len(path[:meet_point_index])
        logs["g_B"] = len(path[meet_point_index + 1 :])

    # print(path)
    # print(
    #     f"Longest Simple Path from {start} to {goal}: {path} with length={len(path)-1}"
    # )
    # print("Number of expansions:", expansions)
    return logs, path, meet_point


from sage.graphs.connectivity import TriconnectivitySPQR
from sage.graphs.graph import Graph


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
results_df = pd.DataFrame(columns=columns)

date = "6_10_24"
number_of_graphs = 1
size_of_graphs = [3,3]

columns = [
    "# blocks",
    "Search type",
    "# expansions",
    "Time [ms]",
    "Memory [kB]",
    "[g_F,g_B]",
    "Grid with Solution",
]
results = []

for i in range(0, number_of_graphs):
    try:
        # Inputs
        name_of_graph = f"{size_of_graphs[0]}x{size_of_graphs[1]}_grid_with_random_blocks_{i}"
        start = 0  # "s"
        goal = size_of_graphs[0] * size_of_graphs[1] - 1  # "t"
        heuristic = (
            "mis_heuristic"  # "heuristic0" / "reachable_heuristic" / "bcc_heuristic" / "mis_heuristic"
        )

        # name_of_graph='manual_graph_0'
        # start = "s"
        # goal="t"

        print("--------------------------")
        name_of_graph=f"{date}/"+name_of_graph
        print(name_of_graph)

        # # unidirectional s-t
        # logs, path, _ = search(
        #     name_of_graph, size_of_graphs, start, goal, "unidirectional", heuristic
        # )
        # print(
        #     f"unidirectional s-t. expansions: {logs['expansions']}, time: {logs['time[ms]']} [ms], memory: {logs['memory[kB]']} [kB], path length: {len(path)-1} [edges]"
        # )
        # results.append(
        #     {
        #         "# blocks": i,
        #         "Search type": "unidirectional s-t",
        #         "# expansions": logs["expansions"],
        #         "Time [ms]": logs["time[ms]"],
        #         "Memory [kB]": logs["memory[kB]"],
        #         "[g_F,g_B]": "[N/A,N/A]",
        #         "Grid with Solution": "file_path_here",  # Update with actual file path if needed
        #     }
        # )

        # # unidirectional t-s
        # logs, path, _ = search(
        #     name_of_graph, size_of_graphs, goal, start, "unidirectional", heuristic
        # )
        # print(
        #     f"unidirectional t-s. expansions: {logs['expansions']}, time: {logs['time[ms]']} [ms], memory: {logs['memory[kB]']} [kB], path length: {len(path)-1} [edges]"
        # )
        # results.append(
        #     {
        #         "# blocks": i,
        #         "Search type": "unidirectional t-s",
        #         "# expansions": logs["expansions"],
        #         "Time [ms]": logs["time[ms]"],
        #         "Memory [kB]": logs["memory[kB]"],
        #         "[g_F,g_B]": "[N/A,N/A]",
        #         "Grid with Solution": "file_path_here",  # Update with actual file path if needed
        #     }
        # )

        # bidirectional
        logs, path, meet_point = search(
            name_of_graph, size_of_graphs, start, goal, "bidirectional", heuristic
        )
        print(
            f"bidirectional. expansions: {logs['expansions']}, time: {logs['time[ms]']} [ms], memory: {logs['memory[kB]']} [kB], path length: {len(path)-1} [edges], g_F: {logs['g_F']}, g_B: {logs['g_B']}"
        )
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
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
        break
    finally:
        # Convert results to a DataFrame
        results_df = pd.DataFrame(results)

        # Save the DataFrame to an Excel file
        results_df.to_excel("search_results.xlsx", index=False, engine="xlsxwriter")
