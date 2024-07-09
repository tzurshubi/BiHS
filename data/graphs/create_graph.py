import networkx as nx
import json
import random

# from ...src import *
import matplotlib.pyplot as plt


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

    return G


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


# Function to save the graph as JSON
def save_graph_to_file(graph, filename):
    data = nx.node_link_data(graph)
    with open(filename, "w") as f:
        json.dump(data, f)


# Inputs
name_of_graph = "4x4 grid with blocks"

# Create the graph
# G = nx.Graph()
# G.add_nodes_from("stabc")
# G.add_edges_from(
#     [("s", "a"), ("s", "c"), ("a", "c"), ("c", "b"), ("c", "t"), ("b", "t")]
# )

G = create_grid_graph_with_specified_blocks(4,4, [9,14])

# x, y, n = 10, 10, 20
# G = create_grid_graph_w_random_blocks(x, y, n)
# G = create_nd_cube_graph(9)

# Logs
print("Graph created with:")
print("number of nodes: ", G.number_of_nodes())
print("number of edges: ", G.number_of_edges())
display_graph(G, "Simple Graph", "simple_graph.png")

# Save the graph as JSON
save_graph_to_file(G, "data/graphs/" + name_of_graph.replace(" ", "_") + ".json")
