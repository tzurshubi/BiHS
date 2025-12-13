import math

class State:
    __slots__ = [
        'graph',
        'head',
        'parent',
        'g',
        'meet_points',
        'path',  # in snake mode: None
        'path_vertices_bitmap',  # visited vertices excluding head
        'path_vertices_and_neighbors_bitmap',  # body vertices + their neighbors, excluding head
        'max_dim_crossed',
        'snake',
        'illegal',  # in snake mode: int bitmap (not a set)
    ]

    def __init__(self, graph, path, meet_points=None, snake=False, max_dim_crossed=None, parent=None):
        self.graph = graph
        self.meet_points = list(meet_points) if meet_points else []
        self.parent = parent
        self.snake = snake

        if not snake:
            # ---- normal mode: keep full path ----
            self.path = path
            self.g = len(path) - 1
            self.head = path[-1] if path else None
            self.path_vertices_bitmap = self._compute_path_vertices_bitmap_from_path(path)
            self.path_vertices_and_neighbors_bitmap = 0
            self.illegal = set()
            self.max_dim_crossed = None
            return

        # ---- snake mode: do NOT keep full path list ----
        # We allow initialization from a list ONCE, then discard it (path=None).
        if not path:
            raise ValueError("snake State requires a non-empty path (at least [start]).")

        self.g = len(path) - 1
        self.head = path[-1]
        self.path = None  # critical memory save

        # Build initial bitmaps from the given list (one-time cost).
        self.path_vertices_bitmap = self._compute_path_vertices_bitmap_from_path(path)
        illegal_bitmap, pvan_bitmap = self._compute_pvan_from_path(path)
        self.path_vertices_and_neighbors_bitmap = pvan_bitmap
        self.illegal = illegal_bitmap  # int bitmap in snake mode

        # max_dim_crossed: keep your old logic, but it needs the list only here
        if max_dim_crossed is not None:
            self.max_dim_crossed = max_dim_crossed
        else:
            # preserve your “single-vertex shortcuts” if you want
            if path == [7]:   self.max_dim_crossed = 2
            elif path == [15]:  self.max_dim_crossed = 3
            elif path == [31]:  self.max_dim_crossed = 4
            elif path == [63]:  self.max_dim_crossed = 5
            elif path == [127]: self.max_dim_crossed = 6
            elif path == [255]: self.max_dim_crossed = 7
            else:
                self.max_dim_crossed = self._compute_max_dim_crossed_from_path(path)

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
        Fast constructor for snake states without allocating a path list.
        """
        obj = cls.__new__(cls)
        obj.graph = graph
        obj.head = head
        obj.g = g
        obj.parent = parent
        obj.meet_points = list(meet_points) if meet_points else []
        obj.path = None
        obj.path_vertices_bitmap = path_vertices_bitmap
        obj.path_vertices_and_neighbors_bitmap = path_vertices_and_neighbors_bitmap
        obj.illegal = illegal_bitmap
        obj.max_dim_crossed = max_dim_crossed
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

    def successor(self, args, snake=False, directionF=True):
        successors = []
        head = self.head
        if head is None:
            return successors

        if not snake:
            # original behavior: allocates list
            for nb in self.graph.neighbors(head):
                if not (self.path_vertices_bitmap & (1 << nb)):
                    new_path = self.materialize_path() + [nb]
                    successors.append(State(self.graph, new_path, self.meet_points, snake=False))
            return successors

        # snake mode: purely bitmap-based, no list allocations
        for nb in self.graph.neighbors(head):
            if self.path_vertices_and_neighbors_bitmap & (1 << nb):
                continue

            if args.graph_type == "cube":
                dim = int(math.log2(head ^ nb))
                if dim > self.max_dim_crossed + 1 and directionF:
                    continue
                new_max_dim_crossed = max(self.max_dim_crossed, dim)
            else:
                new_max_dim_crossed = self.max_dim_crossed

            # body bitmap for new state excludes new head, so we add old head to body
            new_pvb = self.path_vertices_bitmap | (1 << head)

            # update pvan incrementally:
            # start from old pvan, then add old head and all its neighbors except the new head
            new_pvan = self.path_vertices_and_neighbors_bitmap | (1 << head)
            new_illegal = self.illegal | (1 << nb) | (1 << head)  # ensure bits exist

            for nn in self.graph.neighbors(head):
                new_illegal |= 1 << nn
                if nn != nb:
                    new_pvan |= 1 << nn

            # also keep illegal containing head + all neighbors/body/head
            # (pvan excludes new head by definition, which we already maintained)
            succ = State.snake_from_fields(
                graph=self.graph,
                head=nb,
                g=self.g + 1,
                path_vertices_bitmap=new_pvb,
                path_vertices_and_neighbors_bitmap=new_pvan,
                illegal_bitmap=new_illegal,
                meet_points=self.meet_points,
                parent=self,
                max_dim_crossed=new_max_dim_crossed,
            )
            succ.snake = True
            successors.append(succ)

        return successors

    def shares_vertex_with(self, other_state, snake=False):
        if not snake:
            return (self.path_vertices_bitmap & other_state.path_vertices_bitmap) != 0
        return (self.path_vertices_and_neighbors_bitmap & other_state.path_vertices_bitmap) != 0

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
        if not common:
            raise ValueError(f"Cannot concatenate: paths do not share an endpoint. self=({s0},{s1}) other=({o0},{o1})")
        if len(common) > 1:
            raise ValueError(f"Ambiguous concatenation: paths share multiple endpoints {common}.")

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

        # if either operand was snake-ish, result will be snake-ish
        snakeish = (self.path is None) or (other.path is None)
        return State(self.graph, new_path, self.meet_points + other.meet_points + [c], snake=snakeish)

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)
