# Import SageMath library
from sage.all import *
from sage.graphs.connectivity import TriconnectivitySPQR

# Simple calculation using SPQR Tree in SageMath
def main():
    # Create a simple graph
    G = Graph([(1, 2), (1, 4), (1, 8), (1, 12), (3, 4), (2, 3),
               (2, 13), (3, 13), (4, 5), (4, 7), (5, 6), (5, 8),
               (5, 7), (6, 7), (8, 11), (8, 9), (8, 12), (9, 10),
               (9, 11), (9, 12), (10, 12)])

    # Decompose the graph into triconnected components and get the SPQR tree
    tric = TriconnectivitySPQR(G)
    T = tric.get_spqr_tree()

    # Print the SPQR Tree
    print("SPQR Tree vertices:", T.vertices())
    print("SPQR Tree edges:", T.edges())

if __name__ == "__main__":
    main()
