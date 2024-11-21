def is_valid_hypercube_vertex(vertex, dimension):
    """
    Check if a vertex is valid in the hypercube of the given dimension.
    """
    return 0 <= vertex < 2**dimension

def are_neighbors(vertex1, vertex2):
    """
    Check if two vertices are neighbors in a hypercube.
    Two vertices are neighbors if they differ by exactly one bit.
    """
    xor_result = vertex1 ^ vertex2
    return bin(xor_result).count('1') == 1

def is_legal_coil(dimension, path):
    """
    Check if the given path is a legal coil in the hypercube of the given dimension.
    """
    # Check if all vertices are valid for the hypercube dimension
    for vertex in path:
        if not is_valid_hypercube_vertex(vertex, dimension):
            print(f"Invalid vertex: {vertex} is not in the range for dimension {dimension}.")
            return False

    # Check if consecutive vertices are neighbors
    for i in range(len(path) - 1):
        if not are_neighbors(path[i], path[i + 1]):
            print(f"Invalid edge: {path[i]} and {path[i + 1]} are not neighbors.")
            return False

    # Check if the first and last vertices are neighbors to form a coil
    if not are_neighbors(path[0], path[-1]):
        print(f"Invalid edge: {path[0]} and {path[-1]} are not neighbors (coil not closed).")
        return False

    # Check if non-consecutive vertices (excluding the first and last connection) are neighbors
    for i in range(len(path)):
        for j in range(i + 2, len(path)):
            # Skip the first-last connection
            if i == 0 and j == len(path) - 1:
                continue
            if are_neighbors(path[i], path[j]):
                print(f"Invalid connection: {path[i]} and {path[j]} are neighbors but not consecutive.")
                return False

    print("The path is a legal coil in the hypercube.")
    return True

# Example usage
dimension = 4
# path = [0, 32, 36, 44, 45, 47, 39, 7, 6, 22, 30, 31, 27, 11, 9, 1]
path = [0, 8, 12, 13, 15, 7, 3, 1]

if is_legal_coil(dimension, path):
    print("!!! COIL !!!")
    # print(" The path satisfies all conditions.")
else:
    print("--- NOT COIL ---")
    # print("The path does not satisfy the conditions.")
