from models.graph import *
from algorithms.unidirectional_search import *
from algorithms.bidirectional_search import *

# Inputs
name_of_graph = "4x4 grid with blocks"
search_type = "unidirectional"  # "unidirectional" / "bidirectional"

# Load the graph
G = load_graph_from_file("data/graphs/" + name_of_graph.replace(" ", "_") + ".json")

# Define start and goal nodes
start = 0  # "s"
goal = 15  # "t"

# Define the desired heuristic
heuristic = "bcc_heuristic"  # "heuristic0" / "reachable_heuristic" / "bcc_heuristic"

# Run heuristic search to find LSP in the graph
if search_type == "unidirectional":
    path, expansions = uniHS_for_LSP(G, start, goal, heuristic)
elif search_type == "bidirectional":
    path, expansions = biHS_for_LSP(G, start, goal, heuristic)
print("Longest Simple Path from {} to {}:".format(start, goal), path)
print("Number of expansions:", expansions)
