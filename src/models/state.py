import math

class State:
    def __init__(self, graph, path, snake = False, max_dim_crossed = None):
        self.graph = graph  # graph is a NetworkX graph
        self.path = path  # path is a list of vertices representing the path
        self.path_vertices_bitmap = self.compute_path_vertices_bitmap()
        self.g = len(path) - 1
        self.head = path[-1] if path else None
        if snake: 
            self.illegal, self.path_vertices_and_neighbors_bitmap = self.compute_path_vertices_and_neighbors_bitmap()
            if max_dim_crossed: self.max_dim_crossed = max_dim_crossed
            elif self.path==[7]: self.max_dim_crossed = 2
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
        illegal = set(self.path)
        # Iterate over the vertices in the path, excluding the head (last vertex in the path)
        head = self.path[-1]
        for vertex in self.path[:-1]:
            # Set the bit for the vertex itself
            bitmap |= 1 << vertex

            # Set bits for all neighbors of the vertex
            for neighbor in self.graph.neighbors(vertex):
                illegal.add(neighbor)
                if neighbor!=head:
                    bitmap |= 1 << neighbor

        return illegal, bitmap

    def g(self):
        return len(self.path) - 1

    def pi(self):
        return set(self.path)

    def head(self):
        return self.path[-1] if self.path else None

    def tail(self):
        return self.path[:-1] if len(self.path) > 1 else []

    def successor(self, args, snake=False, directionF = True):
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
                    
                    if snake and args.graph_type == "cube":
                        # Calculate the new max_dim_crossed incrementally
                        dimension_crossed = int(math.log2(head ^ neighbor))  # Dimension of the XOR
                        if dimension_crossed <= self.max_dim_crossed + 1 or not directionF:
                            # Append the new state with updated max_dim_crossed
                            new_max_dim_crossed = max(self.max_dim_crossed, dimension_crossed)
                            successor_state = State(self.graph, new_path, snake, max_dim_crossed=new_max_dim_crossed)
                            successors.append(successor_state)
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

    def __add__(self, other):
        """
        Concatenate two State paths when they share an endpoint (head or tail).
        Result path goes from the non-shared endpoint of the left state to the
        non-shared endpoint of the right state, passing through the shared endpoint.

        Example with s1=[0,1,3], s2=[3,2,6], s3=[6,4,5]:

            s1 + s2 -> [0,1,3,2,6]
            s2 + s1 -> [6,2,3,1,0]
            s2 + s3 -> [3,2,6,4,5]
            s3 + s2 -> [5,4,6,2,3]
        """
        if not isinstance(other, State):
            return NotImplemented

        # Must be same graph object
        if self.graph is not other.graph:
            raise ValueError("Cannot add states from different graphs.")

        # Handle empty-path edge cases
        if not self.path:
            return other
        if not other.path:
            return self

        p = self.path
        q = other.path
        s0, s1 = p[0], p[-1]
        o0, o1 = q[0], q[-1]

        # Endpoints
        self_ends = {s0, s1}
        other_ends = {o0, o1}
        common = list(self_ends & other_ends)

        if not common:
            raise ValueError(
                f"Cannot concatenate: paths do not share an endpoint. "
                f"self endpoints = ({s0}, {s1}), "
                f"other endpoints = ({o0}, {o1})"
            )

        if len(common) > 1:
            # Ambiguous case: both endpoints in common (e.g., cycles or identical segment).
            # You can relax this if you want, but for now we fail loudly.
            raise ValueError(
                f"Ambiguous concatenation: paths share multiple endpoints {common}."
            )

        c = common[0]  # shared endpoint

        # x = other endpoint of self, y = other endpoint of other
        x = s1 if s0 == c else s0
        y = o1 if o0 == c else o0

        # Orient self from x -> c
        if p[0] == x and p[-1] == c:
            p_or = p
        elif p[0] == c and p[-1] == x:
            p_or = list(reversed(p))
        else:
            raise ValueError(
                f"Path self does not have shared endpoint {c} as an endpoint, "
                f"or endpoints changed unexpectedly."
            )

        # Orient other from c -> y
        if q[0] == c and q[-1] == y:
            q_or = q
        elif q[0] == y and q[-1] == c:
            q_or = list(reversed(q))
        else:
            raise ValueError(
                f"Path other does not have shared endpoint {c} as an endpoint, "
                f"or endpoints changed unexpectedly."
            )

        # Ensure simple path: allow overlap only at c
        used = set(p_or)
        for v in q_or[1:]:  # skip the shared c
            if v in used:
                raise ValueError(
                    f"Concatenation would repeat vertex {v}, "
                    "so the result would not be a simple path."
                )
            used.add(v)

        new_path = p_or + q_or[1:]

        # Decide whether new state is a "snake"
        snake = hasattr(self, "path_vertices_and_neighbors_bitmap") or \
                hasattr(other, "path_vertices_and_neighbors_bitmap")

        return State(self.graph, new_path, snake=snake)

    def __radd__(self, other):
        """
        Support patterns like sum([s1, s2, s3]) with start=0.
        """
        if other == 0:
            return self
        return self.__add__(other)

