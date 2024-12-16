import math

class State:
    def __init__(self, graph, path, snake = False, max_dim_crossed = None):
        self.graph = graph  # graph is a NetworkX graph
        self.path = path  # path is a list of vertices representing the path
        self.path_vertices_bitmap = self.compute_path_vertices_bitmap()
        self.g = len(path) - 1
        self.head = path[-1] if path else None
        if snake: 
            self.path_vertices_and_neighbors_bitmap = self.compute_path_vertices_and_neighbors_bitmap()
            if max_dim_crossed: self.max_dim_crossed = max_dim_crossed
            else: self.max_dim_crossed = self.compute_max_dim_crossed()


    def compute_path_vertices_bitmap(self):
        """Compute the bitmap for the vertices in the path, excluding the head."""
        bitmap = 0
        for vertex in self.path[:-1]:  # Exclude the head (last vertex)
            bitmap |= 1 << vertex
        return bitmap
    
    def compute_max_dim_crossed(self):
        """Compute the maximum dimension crossed based on the binary representation of vertices."""
        max_dim = 0
        for i in range(1, len(self.path)):
            # XOR between consecutive vertices
            diff = self.path[i] ^ self.path[i - 1]
            # Find the highest bit position (dimension) that is set
            if diff > 0:
                max_dim = max(max_dim, int(math.log2(diff)))
        return max_dim

    def compute_path_vertices_and_neighbors_bitmap(self):
        """
        Compute the bitmap for the vertices in the path, excluding the head and its neighbors.
        This includes setting bits for all the vertices in the path and their neighbors.
        """
        bitmap = 0
        # Iterate over the vertices in the path, excluding the head (last vertex in the path)
        head = self.path[-1]
        for vertex in self.path[:-1]:
            # Set the bit for the vertex itself
            bitmap |= 1 << vertex

            # Set bits for all neighbors of the vertex
            for neighbor in self.graph.neighbors(vertex):
                if neighbor!=head:
                    bitmap |= 1 << neighbor

        return bitmap



    def g(self):
        return len(self.path) - 1

    def pi(self):
        return set(self.path)

    def head(self):
        return self.path[-1] if self.path else None

    def tail(self):
        return self.path[:-1] if len(self.path) > 1 else []

    def successor(self, snake=False, directionF = True):
        """
        Generate successors of the current state.
        Args:
            snake (bool): Whether to consider the snake constraint.
        Returns:
            list[State]: List of successor states.
        """
        successors = []
        head = self.head

        if head is not None:
            for neighbor in self.graph.neighbors(head):
                # Check if the neighbor is valid
                if (not snake and not self.path_vertices_bitmap & (1 << neighbor)) or (
                    snake and not self.path_vertices_and_neighbors_bitmap & (1 << neighbor)
                ):
                    # Create the new path
                    new_path = self.path + [neighbor]
                    
                    if snake:
                        # Calculate the new max_dim_crossed incrementally
                        dimension_crossed = int(math.log2(head ^ neighbor))  # Dimension of the XOR
                        if dimension_crossed <= self.max_dim_crossed + 1 or not directionF:
                            # Append the new state with updated max_dim_crossed
                            new_max_dim_crossed = max(self.max_dim_crossed, dimension_crossed)
                            successors.append(State(self.graph, new_path, snake, max_dim_crossed=new_max_dim_crossed))
                    else:
                        successors.append(State(self.graph, new_path, snake, max_dim_crossed=None))

        return successors

    

    def shares_vertex_with(self, other_state, snake = False):
        """
        Check if this state shares any vertex (excluding heads) with another state.
        if snake: also check if a vertex of this state is a neighbor of a vertex of the other state.
        """
        if not snake: return (self.path_vertices_bitmap & other_state.path_vertices_bitmap) != 0
        else: return (self.path_vertices_and_neighbors_bitmap & other_state.path_vertices_bitmap) != 0

