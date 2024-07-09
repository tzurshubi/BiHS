class State:
    def __init__(self, graph, path):
        self.graph = graph  # graph is a NetworkX graph
        self.path = path  # path is a list of vertices representing the path

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
