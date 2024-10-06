class State:
    def __init__(self, graph, path):
        self.graph = graph  # graph is a NetworkX graph
        self.path = path  # path is a list of vertices representing the path
        self.bitmap = self.compute_bitmap()

    def compute_bitmap(self):
        """Compute the bitmap for the vertices in the path, excluding the head."""
        bitmap = 0
        for vertex in self.path[:-1]:  # Exclude the head (last vertex)
            bitmap |= 1 << vertex
        return bitmap

    def g(self):
        return len(self.path) - 1

    def pi(self):
        return set(self.path)

    def head(self):
        return self.path[-1] if self.path else None

    def tail(self):
        return self.path[:-1] if len(self.path) > 1 else []

    def successor(self):
        successors = []
        head = self.head()
        if head is not None:
            for neighbor in self.graph.neighbors(head):
                if neighbor not in self.pi():
                    new_path = self.path + [neighbor]
                    successors.append(State(self.graph, new_path))
        return successors

    def shares_vertex_with(self, other_state):
        """Check if this state shares any vertex (excluding heads) with another state."""
        return (self.bitmap & other_state.bitmap) != 0
