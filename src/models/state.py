import math
STORE_PATH = True  # Set to False to save memory in snake mode

class State:
    __slots__ = [
        'graph',
        'num_graph_vertices',
        'head',
        'tailtip',
        'parent',
        'g',
        'h',
        'meet_points',
        'path',  # in snake mode: None
        'path_vertices_bitmap',  # visited vertices excluding head
        'path_vertices_and_neighbors_bitmap',  # body vertices + their neighbors, excluding head
        'max_dim_crossed',
        'snake',
        'illegal',  # in snake mode: int bitmap (not a set)
        'traversed_buffer_dimension',
    ]

    def __init__(self, graph, path, meet_points=None, snake=False, max_dim_crossed=None, parent=None):
        self.graph = graph
        self.num_graph_vertices = len(graph.nodes)
        self.meet_points = list(meet_points) if meet_points else []
        self.parent = parent
        self.snake = snake
        self.traversed_buffer_dimension = False
        self.g = len(path) - 1
        self.head = path[-1] if path else None
        self.tailtip = path[0] if path else None
        self.path_vertices_bitmap = self._compute_path_vertices_bitmap_from_path(path)
        if STORE_PATH: self.path = path
        else: self.path = None

        if snake:
            self.illegal, self.path_vertices_and_neighbors_bitmap = self.compute_path_vertices_and_neighbors_bitmap(path)
            if max_dim_crossed is not None:
                self.max_dim_crossed = max_dim_crossed
            else:
                if len(path) == 1: 
                    self.max_dim_crossed = path[0].bit_length() - 1
                else:
                    self.max_dim_crossed = self._compute_max_dim_crossed_from_path(path)
        else: # not snake
            self.path_vertices_and_neighbors_bitmap = 0
            self.illegal = set()
            self.max_dim_crossed = max_dim_crossed
            return

        # if snake and not STORE_PATH:
        # # ---- snake mode: do NOT keep full path list ----
        # # We allow initialization from a list ONCE, then discard it (path=None).
        #     if not path:
        #         raise ValueError("snake State requires a non-empty path (at least [start]).")

        #     self.g = len(path) - 1
        #     self.head = path[-1]
        #     self.tailtip = path[0]
        #     self.path = path if STORE_PATH else None

        #     # Build initial bitmaps from the given list (one-time cost).
        #     self.path_vertices_bitmap = self._compute_path_vertices_bitmap_from_path(path)
        #     illegal_bitmap, pvan_bitmap = self._compute_pvan_from_path(path)
        #     self.path_vertices_and_neighbors_bitmap = pvan_bitmap
        #     self.illegal = illegal_bitmap  # int bitmap in snake mode

        #     # max_dim_crossed: keep your old logic, but it needs the list only here
        #     if max_dim_crossed is not None:
        #         self.max_dim_crossed = max_dim_crossed
        #     else:
        #         # preserve your “single-vertex shortcuts” if you want
        #         if path == [7]:   self.max_dim_crossed = 2
        #         elif path == [15]:  self.max_dim_crossed = 3
        #         elif path == [31]:  self.max_dim_crossed = 4
        #         elif path == [63]:  self.max_dim_crossed = 5
        #         elif path == [127]: self.max_dim_crossed = 6
        #         elif path == [255]: self.max_dim_crossed = 7
        #         else:
        #             self.max_dim_crossed = self._compute_max_dim_crossed_from_path(path)

    # ----------------------------
    # Construction helpers
    # ----------------------------

    @classmethod
    def from_reversed(cls, state):
        if not isinstance(state, State):
            raise TypeError("from_reversed() expects a State instance.")

        # If snake mode has no stored path, we must materialize (rare operation).
        p = state.materialize_path()
        new_path = list(reversed(p))

        snake = hasattr(state, "path_vertices_and_neighbors_bitmap") and state.path_vertices_and_neighbors_bitmap is not None
        if snake and state.max_dim_crossed is not None:
            return cls(state.graph, new_path, meet_points=list(state.meet_points), snake=True, max_dim_crossed=state.max_dim_crossed)
        return cls(state.graph, new_path, meet_points=list(state.meet_points), snake=snake)

    @classmethod
    def snake_from_fields(
        cls,
        graph,
        head,
        g,
        path_vertices_bitmap,
        path_vertices_and_neighbors_bitmap,
        illegal_bitmap,
        meet_points=None,
        parent=None,
        max_dim_crossed=None,
    ):
        """
        Fast constructor for snake states.
        Respects STORE_PATH:
        - if STORE_PATH True: store full path (via parent pointers)
        - else: keep path=None
        """
        obj = cls.__new__(cls)
        obj.graph = graph
        obj.head = head
        obj.g = g
        obj.parent = parent
        obj.meet_points = list(meet_points) if meet_points else []
        obj.snake = True

        # Respect STORE_PATH
        if STORE_PATH:
            if parent is None:
                obj.path = [head]
                obj.tailtip = head
            else:
                # parent.path might be None if STORE_PATH was False earlier, so fall back
                parent_path = parent.path if parent.path is not None else parent.materialize_path()
                obj.path = parent_path + [head]
                obj.tailtip = parent_path[0]
        else:
            obj.path = None
            obj.tailtip = parent.tailtip if parent is not None else head

        obj.path_vertices_bitmap = path_vertices_bitmap
        obj.path_vertices_and_neighbors_bitmap = path_vertices_and_neighbors_bitmap
        obj.illegal = illegal_bitmap
        obj.max_dim_crossed = max_dim_crossed

        # Always initialize this slot
        obj.traversed_buffer_dimension = parent.traversed_buffer_dimension if parent is not None else False

        return obj

    # ----------------------------
    # Bitmap computation (one-time from list)
    # ----------------------------

    @staticmethod
    def _compute_path_vertices_bitmap_from_path(path):
        bitmap = 0
        # exclude head
        for v in path[:-1]:
            bitmap |= 1 << v
        return bitmap

    def _compute_pvan_from_path(self, path):
        """
        Returns (illegal_bitmap, pvan_bitmap) for snake constraints.
        pvan_bitmap: body vertices + their neighbors, excluding head.
        illegal_bitmap: same + head bit set (so illegal includes head).
        """
        head = path[-1]
        pvan = 0
        illegal = 0

        # illegal includes all path vertices
        for v in path:
            illegal |= 1 << v

        # for each body vertex, add itself + its neighbors (excluding head in pvan)
        for v in path[:-1]:
            pvan |= 1 << v
            for nb in self.graph.neighbors(v):
                illegal |= 1 << nb
                if nb != head:
                    pvan |= 1 << nb

        # ensure illegal includes head (already does), pvan excludes head by construction
        return illegal, pvan

    @staticmethod
    def _compute_max_dim_crossed_from_path(path):
        max_dim = 0
        for i in range(1, len(path)):
            diff = path[i] ^ path[i - 1]
            if diff > 0:
                max_dim = max(max_dim, int(math.log2(diff)))
        return max_dim

    # ----------------------------
    # Path materialization (only when needed)
    # ----------------------------

    def materialize_path(self):
        """
        Reconstruct the path by following parent pointers.
        Works for snake mode (path=None) and normal mode.
        """
        if self.path is not None:
            return list(self.path)

        # snake mode: walk back through parents
        nodes = []
        cur = self
        while cur is not None:
            nodes.append(cur.head)
            cur = cur.parent
        nodes.reverse()
        return nodes

    def illegal_set(self):
        """
        Only if you really need a Python set (expensive).
        """
        if isinstance(self.illegal, set):
            return set(self.illegal)
        bm = int(self.illegal)
        out = set()
        i = 0
        while bm:
            if bm & 1:
                out.add(i)
            bm >>= 1
            i += 1
        return out

    # ----------------------------
    # Convenience methods
    # ----------------------------

    def pi(self):
        # avoid accidental huge allocations during search
        return set(self.materialize_path())

    def tail(self):
        p = self.materialize_path()
        return p[:-1] if len(p) > 1 else []

    # ----------------------------
    # Successors (NO path allocation in snake mode)
    # ----------------------------
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
                            successor_state = State(self.graph, new_path, self.meet_points, snake, new_max_dim_crossed, self)
                            successors.append(successor_state)
                    else:
                        successors.append(State(self.graph, new_path, meet_points=self.meet_points, snake=snake, max_dim_crossed=None, parent=self))

        return successors

    def shares_vertex_with(self, other_state, snake=False):
        if not snake:
            return (self.path_vertices_bitmap & other_state.path_vertices_bitmap) != 0
        return (self.path_vertices_and_neighbors_bitmap & other_state.path_vertices_bitmap) != 0


    def compute_path_vertices_and_neighbors_bitmap(self, path):
        """
        Compute the bitmap for the vertices in the path, excluding the head and its neighbors.
        This includes setting bits for all the vertices in the path and their neighbors.
        """
        bitmap = 0
        illegal = set(path)
        # Iterate over the vertices in the path, excluding the head (last vertex in the path)
        head = path[-1]
        for vertex in path[:-1]:
            # Set the bit for the vertex itself
            bitmap |= 1 << vertex

            # Set bits for all neighbors of the vertex
            for neighbor in self.graph.neighbors(vertex):
                illegal.add(neighbor)
                if neighbor!=head:
                    bitmap |= 1 << neighbor
        return illegal, bitmap

    # ----------------------------
    # Concatenation: forces materialization (rare operation)
    # ----------------------------

    def __add__(self, other):
        if not isinstance(other, State):
            return NotImplemented
        if self.graph is not other.graph:
            raise ValueError("Cannot add states from different graphs.")

        p = self.materialize_path()
        q = other.materialize_path()

        if not p:
            return other
        if not q:
            return self

        s0, s1 = p[0], p[-1]
        o0, o1 = q[0], q[-1]

        common = list({s0, s1} & {o0, o1})
        if len(common) != 1:
            raise ValueError(f"Cannot concatenate: endpoints mismatch. self=({s0},{s1}) other=({o0},{o1})")

        c = common[0]
        x = s1 if s0 == c else s0
        y = o1 if o0 == c else o0

        # orient p from x -> c
        if p[0] == x and p[-1] == c:
            p_or = p
        elif p[0] == c and p[-1] == x:
            p_or = list(reversed(p))
        else:
            raise ValueError("Unexpected endpoint orientation for self.")

        # orient q from c -> y
        if q[0] == c and q[-1] == y:
            q_or = q
        elif q[0] == y and q[-1] == c:
            q_or = list(reversed(q))
        else:
            raise ValueError("Unexpected endpoint orientation for other.")

        used = set(p_or)
        for v in q_or[1:]:
            if v in used:
                raise ValueError(f"Concatenation would repeat vertex {v}.")
            used.add(v)

        new_path = p_or + q_or[1:]

        # Decide snake-ness by the flag, not by whether we stored a path list
        snakeish = self.snake or other.snake

        out = State(
            self.graph,
            new_path,
            meet_points=self.meet_points + other.meet_points + [c],
            snake=snakeish,
        )

        # Keep these consistent (State will drop path if STORE_PATH=False, but tailtip still should be correct)
        out.tailtip = new_path[0]
        out.traversed_buffer_dimension = self.traversed_buffer_dimension or other.traversed_buffer_dimension

        return out



    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)