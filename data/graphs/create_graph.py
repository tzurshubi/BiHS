import networkx as nx
import json
import random
import numpy as np
import matplotlib.patches as patches
import os


# from ...src import *
import matplotlib.pyplot as plt


def random_5x5_nodes_from_13x13_grid():
    blocks = []
    grid = np.arange(169).reshape(13, 13)

    # Select a random top-left corner for the 5x5 sub-grid
    start_row = random.randint(0, 13 - 5)
    start_col = random.randint(0, 13 - 5)
    print(f"Top-left corner: ({start_row}, {start_col})")

    # Extract the 5x5 sub-grid
    sub_grid = grid[start_row:start_row + 5, start_col:start_col + 5]

    # Extract the rhombus shape within the 5x5 grid
    blocks.append(sub_grid[0, 2])  # Middle element in the first row
    blocks.extend(sub_grid[1, 1:4:1])  # Two middle elements in the second row
    blocks.extend(sub_grid[2, 0:5])  # All elements in the third row
    blocks.extend(sub_grid[3, 1:4:1])  # Two middle elements in the fourth row
    blocks.append(sub_grid[4, 2])  # Middle element in the last row

    return blocks


    return sub_grid.tolist()

    # Print the result
    print("Randomly selected 5x5 grid from the 13x13 grid:")
    print(sub_grid)


def create_random_graph(n, p=0.1):
    """
    Create a random graph with n vertices where each edge is included with probability p.

    Parameters:
    n (int): Number of vertices.
    p (float): Probability of edge creation. Default is 0.1.

    Returns:
    networkx.Graph: The created random graph.
    """
    G = nx.Graph()

    # Add n nodes to the graph
    G.add_nodes_from(range(n))

    # Add edges between pairs of nodes with probability p
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                G.add_edge(i, j)

    return G


# Function to create an n-dimensional cube graph
def create_nd_cube_graph(n):
    # Create an n-dimensional cube graph
    G = nx.hypercube_graph(n)

    # Relabel nodes to be integers
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    return G


# Function to create a grid graph and remove random vertices
def create_grid_graph_w_random_blocks(x, y, n):
    # Create a grid graph
    G = nx.grid_2d_graph(x, y)

    # Relabel nodes to be single integers
    mapping = {(i, j): i * y + j for i in range(x) for j in range(y)}
    G = nx.relabel_nodes(G, mapping)

    # Randomly select and remove n vertices
    nodes_to_remove = random.sample(list(G.nodes()), n)
    G.remove_nodes_from(nodes_to_remove)

    return G, nodes_to_remove


# Function to create a grid graph and remove specified vertices
def create_grid_graph_with_specified_blocks(x, y, nodes_to_remove):
    """
    Create a grid graph of size x by y and remove specified vertices.

    Parameters:
    x (int): Number of rows.
    y (int): Number of columns.
    nodes_to_remove (list): List of nodes to remove from the graph.

    Returns:
    networkx.Graph: The created grid graph with specified vertices removed.
    """
    # Create a grid graph
    G = nx.grid_2d_graph(x, y)

    # Relabel nodes to be single integers
    mapping = {(i, j): i * y + j for i in range(x) for j in range(y)}
    G = nx.relabel_nodes(G, mapping)

    # Remove specified nodes
    G.remove_nodes_from(nodes_to_remove)

    return G


# Function to display the graph
def display_graph(graph, title="Graph", filename=None):
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(graph)
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=500,
        node_color="skyblue",
        font_size=10,
        font_weight="bold",
        edge_color="gray",
    )
    plt.title(title)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

# Function to display a graph with optional highlighted path and points.
def display_graph_with_path_and_points(graph, title="Graph", filename=None, path=None, points=None):
    """
    Display a graph with optional highlighted path and points.
    
    Args:
        graph (networkx.Graph): The graph to display.
        title (str): Title of the graph.
        filename (str, optional): If provided, saves the graph as a PNG to the given filename.
        path (list, optional): List of vertices to highlight as a path. Their edges will be blue.
        points (list, optional): List of vertices to highlight in red.
    """
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(graph)

    # Default node and edge styles
    node_colors = ["red" if node in (points or []) else "skyblue" for node in graph.nodes()]
    edge_colors = [
        "blue" if (u, v) in zip(path or [], (path or [])[1:]) or (v, u) in zip(path or [], (path or [])[1:]) else "gray"
        for u, v in graph.edges()
    ]
    edge_widths = [2 if (u, v) in zip(path or [], (path or [])[1:]) or (v, u) in zip(path or [], (path or [])[1:]) else 1 for u, v in graph.edges()]

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
        print(f"Graph saved to {filename}")
    else:
        plt.show()

# Function to save the graph as JSON
def save_graph_to_file(graph, filename):
    data = nx.node_link_data(graph)
    with open(filename, "w") as f:
        json.dump(data, f)


def save_table_as_png(
    rows, cols, black_cells=[], filename="grid.png", path=None, points=None
):
    # Create a figure and axis with a bit of padding to ensure all borders are shown
    fig, ax = plt.subplots(figsize=(cols, rows), dpi=100)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

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
    if points:
        for point in points:
            point_x, point_y = point % cols, point // cols
            ax.plot(point_x, point_y, "bo", markersize=20)  # Blue dot

    # Save the figure
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


date = "cubes"
number_of_graphs = 10
graph_type = "cube" # "grid" # "cube" # "manual" # maze"
dimension_of_graphs = [3,3] # dimension for cube
suffled_blocks = list(range(dimension_of_graphs[0] * dimension_of_graphs[1]))
random.shuffle(suffled_blocks)

# Create folder
folder_path = "data/graphs/" + date + "/"
if not os.path.exists(folder_path):
    # Create the folder
    os.makedirs(folder_path)

if graph_type=="maze":
    blocks = [5, 14, 15, 16, 18, 20, 21, 22, 23, 24, 29, 40, 42, 44, 46, 47, 48, 50] + list(range(66, 71)) + [72, 73] + list(range(75, 78)) + [91] + list(range(93, 99)) + [100, 101, 102, 104] + list(range(117, 122)) + list(range(123, 129)) + [138] + list(range(144, 150)) + [151, 153, 154]


for i in range(0, number_of_graphs):
    # Create a grid
    if graph_type=="grid":
        name_of_graph = (
            f"{dimension_of_graphs[0]}x{dimension_of_graphs[1]}_grid_with_random_blocks_{i}"
        )
        num_of_blocks = int(i % (dimension_of_graphs[0] * dimension_of_graphs[1] / 2))
        G, blocks = create_grid_graph_w_random_blocks(dimension_of_graphs[0], dimension_of_graphs[1], num_of_blocks)

        # Create grid with specified blocks
        blocks = suffled_blocks[0:i] # suffled_blocks[0:i]
        G = create_grid_graph_with_specified_blocks(dimension_of_graphs[0], dimension_of_graphs[1], blocks)

    # Create a cube
    if graph_type=="cube":
        name_of_graph=f"{i}d_hypercube"
        G = create_nd_cube_graph(i)
        if G.has_edge(0, 1):
            G.remove_edge(0, 1)

    # Create a maze
    if graph_type=="maze":
        name_of_graph = (
            f"{dimension_of_graphs[0]}x{dimension_of_graphs[1]}_maze_with_blocks_and_random_removals_{i}"
        )
        # Create grid with specified blocks
        # blocks = [5, 14, 15, 16, 18, 20, 21, 22, 23, 24, 29, 40, 42, 44, 46, 47, 48, 50] + list(range(66, 71)) + [72, 73] + list(range(75, 78)) + [91] + list(range(93, 99)) + [100, 101, 102, 104] + list(range(117, 122)) + list(range(123, 129)) + [138] + list(range(144, 150)) + [151, 153, 154]
        # for j in range(0,i):
        if i>0:
            blocks = list(set(blocks)-set(random_5x5_nodes_from_13x13_grid()))
        G = create_grid_graph_with_specified_blocks(
            dimension_of_graphs[0], dimension_of_graphs[1], blocks
        )

    # Create a manual graph
    # if graph_type=="manual":
    #     name_of_graph = f"paper_graph_{i}"
    #     G = nx.Graph()
    #     G.add_nodes_from("x")
    #     G.add_edges_from(
    #         [("s", "1"),("s", "2"),("1", "2"),
    #         ("3", "4"),("4", "5"),("3", "5"),
    #         ("6", "7"),("6", "t"),("7", "t"),
    #         ("2", "3"),("2", "6"),("3", "6"),
    #         ]
    #     )
    #     G.add_edges_from(
    #     [("s", "v1"), ("s", "y"), ("s", "v2"), ("v1","v2"),("v1", "y"),("x", "v1"),("x", "v2"),("x", "y"),
    #     ("y", "z'"), ("y", "z"), ("z", "z'"),
    #     ("w4", "x"), ("w4", "t"),
    #     ("x","w1"),("w1", "w2"), ("w2", "w3"),("w3", "t"),
    #     ("u1", "x"), ("u1", "u2"), ("u2", "u3"), ("u3", "t"),
    #     ("u1", "u4"), ("u2", "u4"),
    #     ("u4", "u5"), ("u5", "u2"),("u1", "u5")
    #     ]
    #         )

    # G = create_random_graph(25, 0.2)

    # G = create_nd_cube_graph(9)

    # Logs
    print("Graph created with:")
    print("number of nodes: ", G.number_of_nodes())
    print("number of edges: ", G.number_of_edges())
    display_graph(G, "Current Graph", "current_graph.png")

    # Save the graph as JSON
    save_graph_to_file(G, folder_path + name_of_graph.replace(" ", "_") + ".json")

    # If it's a Grid, Save it as PNG
    if graph_type=="grid" or graph_type=="maze" : 
        save_table_as_png(
            dimension_of_graphs[0],
            dimension_of_graphs[1],
            blocks,
            folder_path + name_of_graph.replace(" ", "_") + ".png",
            None,
            None,
        )
    
    # Save the graph as PNG
    else:
        display_graph(
            G,
            name_of_graph.replace(" ", "_"),
            folder_path + name_of_graph.replace(" ", "_") + ".png",
        )
