import math

class State:
    __slots__ = [
        'graph',
        'head',
        'parent',
        'g',
        'h',
        'meet_points',
        'path',  # now ALWAYS stored (also in snake mode)
        'path_vertices_bitmap',  # visited vertices excluding head
        'path_vertices_and_neighbors_bitmap',  # body vertices + their neighbors, excluding head
        'max_dim_crossed',
        'snake',
        'illegal',  # in snake mode: int bitmap (not a set)
        'traversed_buffer_dimension',
    ]

    def __init__(self, graph, path, meet_points=None, snake=False, max_dim_crossed=None, parent=None):
        self.graph = graph
        self.meet_points = list(meet_points) if meet_points else []
        self.parent = parent
        self.snake = snake
        self.path = list(path) if path is not None else None  # always keep path list
        self.traversed_buffer_dimension = False

        if not self.path:
            raise ValueError("State requires a non-empty path (at least [start]).")

        self.g = len(self.path) - 1
        self.head = self.path[-1]
        self.h = 0  # if you rely on default h existing

        if not snake:
            # ---- normal mode ----
            self.path_vertices_bitmap = self._compute_path_vertices_bitmap_from_path(self.path)
            self.path_vertices_and_neighbors_bitmap = 0
            self.illegal = set()
            self.max_dim_crossed = None
            return

        # ---- snake mode: keep path, but also keep bitmaps ----
        self.path_vertices_bitmap = self._compute_path_vertices_bitmap_from_path(self.path)
        illegal_bitmap, pvan_bitmap = self._compute_pvan_from_path(self.path)
        self.path_vertices_and_neighbors_bitmap = pvan_bitmap
        self.illegal = illegal_bitmap  # int bitmap in snake mode

        if max_dim_crossed is not None:
            self.max_dim_crossed = max_dim_crossed
        else:
            if self.path == [7]:     self.max_dim_crossed = 2
            elif self.path == [15]:  self.max_dim_crossed = 3
            elif self.path == [31]:  self.max_dim_crossed = 4
            elif self.path == [63]:  self.max_dim_crossed = 5
            elif self.path == [127]: self.max_dim_crossed = 6
            elif self.path == [255]: self.max_dim_crossed = 7
            else:
                self.max_dim_crossed = self._compute_max_dim_crossed_from_path(self.path)

    # ----------------------------
    # Construction helpers
    # ----------------------------

    @classmethod
    def from_reversed(cls, state):
        if not isinstance(state, State):
            raise TypeError("from_reversed() expects a State instance.")

        new_path = list(reversed(state.path))
        if state.snake and state.max_dim_crossed is not None:
            return cls(state.graph, new_path, meet_points=list(state.meet_points), snake=True, max_dim_crossed=state.max_dim_crossed)
        return cls(state.graph, new_path, meet_points=list(state.meet_points), snake=state.snake)

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
        traversed_buffer_dimension: bool = False,
    ):
        """
        Constructor for snake states.
        Now ALSO stores a full path list.
        """
        if parent is None:
            # fallback: you can still build a 1-vertex path
            path = [head]
        else:
            # build full path list
            path = parent.path + [head]

        obj = cls.__new__(cls)
        obj.graph = graph
        obj.head = head
        obj.g = g
        obj.h = 0
        obj.parent = parent
        obj.meet_points = list(meet_points) if meet_points else []
        obj.snake = True

        obj.path = path  # IMPORTANT: store path again
        obj.path_vertices_bitmap = path_vertices_bitmap
        obj.path_vertices_and_neighbors_bitmap = path_vertices_and_neighbors_bitmap
        obj.illegal = illegal_bitmap
        obj.max_dim_crossed = max_dim_crossed
        obj.traversed_buffer_dimension = traversed_buffer_dimension
        return obj

    # ----------------------------
    # Bitmap computation
    # ----------------------------

    @staticmethod
    def _compute_path_vertices_bitmap_from_path(path):
        bitmap = 0
        for v in path[:-1]:
            bitmap |= 1 << v
        return bitmap

    def _compute_pvan_from_path(self, path):
        head = path[-1]
        pvan = 0
        illegal = 0

        for v in path:
            illegal |= 1 << v

        for v in path[:-1]:
            pvan |= 1 << v
            for nb in self.graph.neighbors(v):
                illegal |= 1 << nb
                if nb != head:
                    pvan |= 1 << nb

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
    # Path helpers (now trivial)
    # ----------------------------

    def materialize_path(self):
        # path is always stored now
        return list(self.path)

    def pi(self):
        return set(self.path)

    def tail(self):
        return self.path[:-1] if len(self.path) > 1 else []

    # ----------------------------
    # Successors
    # ----------------------------

    def successor(self, args, snake=False, directionF=True, ignore_max_dim_crossed=False):
        successors = []
        head = self.head
        if head is None:
            return successors

        if not snake:
            for nb in self.graph.neighbors(head):
                if not (self.path_vertices_bitmap & (1 << nb)):
                    new_path = self.path + [nb]
                    successors.append(State(self.graph, new_path, self.meet_points, snake=False))
            return successors

        # snake mode: bitmap legality checks, but now also stores path in successor
        for nb in self.graph.neighbors(head):
            if self.path_vertices_and_neighbors_bitmap & (1 << nb):
                continue

            if args.graph_type == "cube":
                dim = int(math.log2(head ^ nb))
                if dim > self.max_dim_crossed + 1 and directionF and not ignore_max_dim_crossed:
                    continue
                new_max_dim_crossed = max(self.max_dim_crossed, dim)
            else:
                new_max_dim_crossed = self.max_dim_crossed

            new_pvb = self.path_vertices_bitmap | (1 << head)

            new_pvan = self.path_vertices_and_neighbors_bitmap | (1 << head)
            new_illegal = self.illegal | (1 << nb) | (1 << head)

            for nn in self.graph.neighbors(head):
                new_illegal |= 1 << nn
                if nn != nb:
                    new_pvan |= 1 << nn

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
                traversed_buffer_dimension=self.traversed_buffer_dimension,
            )
            successors.append(succ)

        return successors

    def shares_vertex_with(self, other_state, snake=False):
        if not snake:
            return (self.path_vertices_bitmap & other_state.path_vertices_bitmap) != 0
        return (self.path_vertices_and_neighbors_bitmap & other_state.path_vertices_bitmap) != 0

    def __add__(self, other):
        if not isinstance(other, State):
            return NotImplemented
        if self.graph is not other.graph:
            raise ValueError("Cannot add states from different graphs.")

        p = self.path
        q = other.path

        if not p:
            return other
        if not q:
            return self

        s0, s1 = p[0], p[-1]
        o0, o1 = q[0], q[-1]

        # exactly one shared endpoint
        common = list({s0, s1} & {o0, o1})
        if len(common) != 1:
            raise ValueError(
                f"Cannot concatenate paths: endpoints mismatch. "
                f"self=({s0},{s1}), other=({o0},{o1})"
            )

        c = common[0]

        # orient self: x -> c
        if p[-1] == c:
            p_or = p
        elif p[0] == c:
            p_or = list(reversed(p))
        else:
            raise RuntimeError("Unexpected orientation for self path.")

        # orient other: c -> y
        if q[0] == c:
            q_or = q
        elif q[-1] == c:
            q_or = list(reversed(q))
        else:
            raise RuntimeError("Unexpected orientation for other path.")

        # ensure no vertex repetition
        used = set(p_or)
        for v in q_or[1:]:
            if v in used:
                raise ValueError(f"Concatenation would repeat vertex {v}.")
            used.add(v)

        new_path = p_or + q_or[1:]

        snakeish = self.snake or other.snake
        return State(
            graph=self.graph,
            path=new_path,
            meet_points=self.meet_points + other.meet_points + [c],
            snake=snakeish,
        )


    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)
