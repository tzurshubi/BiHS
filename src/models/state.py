class State:
    def __init__(self, graph, path, snake = False):
        self.graph = graph  # graph is a NetworkX graph
        self.path = path  # path is a list of vertices representing the path
        self.path_vertices_bitmap = self.compute_path_vertices_bitmap()
        self.g = len(path) - 1
        self.head = path[-1] if path else None
        if snake: self.path_vertices_and_neighbors_bitmap = self.compute_path_vertices_and_neighbors_bitmap()


    def compute_path_vertices_bitmap(self):
        """Compute the bitmap for the vertices in the path, excluding the head."""
        bitmap = 0
        for vertex in self.path[:-1]:  # Exclude the head (last vertex)
            bitmap |= 1 << vertex
        return bitmap
    
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

    def successor(self, snake = False):
        successors = []
        head = self.head
        if head is not None:
            for neighbor in self.graph.neighbors(head):
                # if neighbor not in self.pi():
                if (not snake and not self.path_vertices_bitmap & (1 << neighbor)) or (snake and not self.path_vertices_and_neighbors_bitmap & (1 << neighbor)):
                    new_path = self.path + [neighbor]
                    successors.append(State(self.graph, new_path, snake))
        return successors
    

    def shares_vertex_with(self, other_state, snake = False):
        """
        Check if this state shares any vertex (excluding heads) with another state.
        if snake: also check if a vertex of this state is a neighbor of a vertex of the other state.
        """
        if not snake: return (self.path_vertices_bitmap & other_state.path_vertices_bitmap) != 0
        else: return (self.path_vertices_and_neighbors_bitmap & other_state.path_vertices_bitmap) != 0

