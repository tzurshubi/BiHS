import networkx as nx
import json

print("models/graph")

# def load_graph_from_file(file_path):
#     G = nx.Graph()
#     with open(file_path, "r") as file:
#         data = json.load(file)
#         G.add_nodes_from(data["nodes"])
#         G.add_edges_from(data["edges"])
#     return G


def load_graph_from_file(file_path):
    G = nx.Graph()
    with open(file_path, "r") as file:
        data = json.load(file)
        # Extract node labels
        nodes = [node["id"] for node in data["nodes"]]
        G.add_nodes_from(nodes)
        # Extract edges
        edges = [(edge["source"], edge["target"]) for edge in data["links"]]
        G.add_edges_from(edges)
    return G


def save_graph_to_file(G, file_path):
    data = {"nodes": list(G.nodes), "edges": list(G.edges)}
    with open(file_path, "w") as file:
        json.dump(data, file)
