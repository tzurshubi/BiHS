import math

STORE_PATH = False  # Set to False to save memory in snake mode

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
        'path',  # List if STORE_PATH=True or parent is None, else None
        'path_vertices_bitmap',  # visited vertices excluding head
        'path_vertices_and_neighbors_bitmap',  # body vertices + their neighbors, excluding head
        'max_dim_crossed',
        'snake',
        'illegal',  # in snake mode: int bitmap; in non-snake: set
        'traversed_buffer_dimension',
    ]

    def __init__(self, graph, path, meet_points=None, snake=False, max_dim_crossed=None, parent=None):
        self.graph = graph
        self.num_graph_vertices = len(graph.nodes)
        self.meet_points = list(meet_points) if meet_points else []
        self.parent = parent
        self.snake = snake
        self.traversed_buffer_dimension = False
        
        # --- 1. Identify Head and Tail ---
        if not path:
            raise ValueError("Path cannot be empty.")
            
        # If parent exists, 'path' might be the full path OR just [new_head] (optimization)
        self.head = path[-1]
        
        # --- 2. Initialize Metrics (g, tailtip) ---
        if parent is None:
            # Root Node (Start of search)
            self.g = len(path) - 1
            self.tailtip = path[0]
            # Force storage for root nodes so history isn't lost
            self.path = path 
        else:
            # Successor Node
            self.g = parent.g + 1
            self.tailtip = parent.tailtip
            # Conditional Storage
            self.path = path if STORE_PATH else None

        # --- 3. Compute Bitmaps & Constraints ---
        if parent is None:
            # === FULL COMPUTATION (From scratch) ===
            self.path_vertices_bitmap = self._compute_path_vertices_bitmap_from_path(path)
            
            if snake:
                self.illegal, self.path_vertices_and_neighbors_bitmap = self.compute_path_vertices_and_neighbors_bitmap(path)
                if max_dim_crossed is not None:
                    self.max_dim_crossed = max_dim_crossed
                else:
                    if len(path) == 1: 
                        self.max_dim_crossed = path[0].bit_length() - 1
                    else:
                        self.max_dim_crossed = self._compute_max_dim_crossed_from_path(path)
            else:
                self.path_vertices_and_neighbors_bitmap = 0
                self.illegal = set()
                self.max_dim_crossed = max_dim_crossed

        else:
            # === INCREMENTAL UPDATE (Optimization) ===
            # Update visited bitmap: Parent's bitmap + Parent's Head (which was previously excluded)
            self.path_vertices_bitmap = parent.path_vertices_bitmap | (1 << parent.head)

            if snake:
                # 3a. Max Dim Crossed
                if max_dim_crossed is not None:
                    self.max_dim_crossed = max_dim_crossed
                else:
                    self.max_dim_crossed = parent.max_dim_crossed

                # 3b. PVAN (Body + Neighbors, Exclude Head)
                # Parent PVAN covers (ParentBody + Neighbors). 
                # We must add ParentHead (now Body) and its Neighbors.
                # We must exclude CurrentHead.
                
                pvan = parent.path_vertices_and_neighbors_bitmap
                pvan |= (1 << parent.head) # Add parent head to body
                
                # Add neighbors of parent head
                # (Optimization: If graph structure allows, use a precomputed bitmap for neighbors)
                for nb in self.graph.neighbors(parent.head):
                    pvan |= (1 << nb)
                
                # Exclude the current head
                pvan &= ~(1 << self.head)
                self.path_vertices_and_neighbors_bitmap = pvan

                # 3c. Illegal (All Path Vertices + All Neighbors)
                # Parent Illegal covers (ParentPath + Neighbors).
                # We just need to add CurrentHead and Neighbors(CurrentHead).
                ill = parent.illegal
                ill |= (1 << self.head)
                for nb in self.graph.neighbors(self.head):
                    ill |= (1 << nb)
                self.illegal = ill
            else:
                self.path_vertices_and_neighbors_bitmap = 0
                self.illegal = set()
                self.max_dim_crossed = max_dim_crossed


    # ----------------------------
    # Construction helpers
    # ----------------------------

    @classmethod
    def from_reversed(cls, state):
        if not isinstance(state, State):
            raise TypeError("from_reversed() expects a State instance.")

        # Must materialize because we are reversing the order
        p = state.materialize_path()
        new_path = list(reversed(p))

        snake = hasattr(state, "path_vertices_and_neighbors_bitmap") and state.path_vertices_and_neighbors_bitmap is not None
        # Creates a new ROOT state (parent=None), so full path is stored automatically
        if snake and state.max_dim_crossed is not None:
            return cls(state.graph, new_path, meet_points=list(state.meet_points), snake=True, max_dim_crossed=state.max_dim_crossed)
        return cls(state.graph, new_path, meet_points=list(state.meet_points), snake=snake)

    # ----------------------------
    # Bitmap computation
    # ----------------------------

    @staticmethod
    def _compute_path_vertices_bitmap_from_path(path):
        bitmap = 0
        for v in path[:-1]: # exclude head
            bitmap |= 1 << v
        return bitmap

    def _compute_max_dim_crossed_from_path(self, path):
        max_dim = 0
        for i in range(1, len(path)):
            diff = path[i] ^ path[i - 1]
            if diff > 0:
                max_dim = max(max_dim, int(math.log2(diff)))
        return max_dim

    def compute_path_vertices_and_neighbors_bitmap(self, path):
        """
        Full computation for root nodes.
        Returns: (illegal_bitmap, pvan_bitmap)
        """
        pvan = 0
        illegal = 0
        head = path[-1]
        
        # Illegal includes all path vertices
        for v in path:
            illegal |= 1 << v

        # PVAN includes body vertices + neighbors, excludes head
        for v in path[:-1]:
            pvan |= 1 << v
            for nb in self.graph.neighbors(v):
                illegal |= 1 << nb # Update illegal with neighbors
                if nb != head:
                    pvan |= 1 << nb
        
        # Ensure head neighbors are in illegal (head is already in illegal)
        for nb in self.graph.neighbors(head):
            illegal |= 1 << nb

        return illegal, pvan

    # ----------------------------
    # Path materialization
    # ----------------------------

    def materialize_path(self):
        """
        Reconstruct the path by following parent pointers.
        Essential when STORE_PATH is False.
        """
        if self.path is not None:
            return list(self.path)

        # Walk back through parents
        nodes = []
        cur = self
        while cur is not None:
            nodes.append(cur.head)
            cur = cur.parent
        nodes.reverse()
        return nodes

    def illegal_set(self):
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
        return set(self.materialize_path())

    def tail(self):
        p = self.materialize_path()
        return p[:-1] if len(p) > 1 else []

    # ----------------------------
    # Successors (Optimized)
    # ----------------------------
    def successor(self, args, snake=False, directionF=True):
        successors = []
        head = self.head

        # Pre-calculate masks/values to avoid repetitive attribute lookups
        p_bitmap = self.path_vertices_bitmap
        pvan_bitmap = self.path_vertices_and_neighbors_bitmap if snake else 0
        
        # Iterate neighbors
        for neighbor in self.graph.neighbors(head):
            neighbor_mask = 1 << neighbor
            
            # Fast Validity Check
            if snake:
                if pvan_bitmap & neighbor_mask:
                    continue
            else:
                if p_bitmap & neighbor_mask:
                    continue

            # Determine Path Argument for new State
            if STORE_PATH:
                new_path = self.materialize_path() + [neighbor]
            else:
                # Optimization: Only pass the new head. 
                # The State constructor will use 'self' (passed as parent) to infer history.
                new_path = [neighbor]

            # Logic for Hypercubes (Snake constraints)
            if snake and args.graph_type == "cube":
                dimension_crossed = int(math.log2(head ^ neighbor))
                
                if dimension_crossed <= self.max_dim_crossed + 1 or not directionF:
                    new_max = max(self.max_dim_crossed, dimension_crossed)
                    successor_state = State(
                        self.graph, 
                        new_path, 
                        self.meet_points, 
                        snake, 
                        new_max, 
                        parent=self
                    )
                    successors.append(successor_state)
            else:
                # Standard Graph / Non-Snake
                successors.append(State(
                    self.graph, 
                    new_path, 
                    meet_points=self.meet_points, 
                    snake=snake, 
                    max_dim_crossed=None, 
                    parent=self
                ))

        return successors

    def shares_vertex_with(self, other_state, snake=False):
        # Note: If STORE_PATH=False, we rely on bitmaps which is fine.
        if not snake:
            return (self.path_vertices_bitmap & other_state.path_vertices_bitmap) != 0
        return (self.path_vertices_and_neighbors_bitmap & other_state.path_vertices_bitmap) != 0

    # ----------------------------
    # Concatenation
    # ----------------------------

    def __add__(self, other):
        if not isinstance(other, State):
            return NotImplemented
        if self.graph is not other.graph:
            raise ValueError("Cannot add states from different graphs.")

        # Must materialize to merge
        p = self.materialize_path()
        q = other.materialize_path()

        if not p: return other
        if not q: return self

        s0, s1 = p[0], p[-1]
        o0, o1 = q[0], q[-1]

        common = list({s0, s1} & {o0, o1})
        if len(common) != 1:
            raise ValueError(f"Cannot concatenate: endpoints mismatch. self=({s0},{s1}) other=({o0},{o1})")

        c = common[0]
        x = s1 if s0 == c else s0
        y = o1 if o0 == c else o0

        # Orientation logic
        if p[0] == x and p[-1] == c: p_or = p
        else: p_or = list(reversed(p))

        if q[0] == c and q[-1] == y: q_or = q
        else: q_or = list(reversed(q))

        used = set(p_or)
        for v in q_or[1:]:
            if v in used:
                raise ValueError(f"Concatenation would repeat vertex {v}.")
            used.add(v)

        new_path = p_or + q_or[1:]
        snakeish = self.snake or other.snake

        # Result is a new Root State (parent=None), so full path is stored.
        out = State(
            self.graph,
            new_path,
            meet_points=self.meet_points + other.meet_points + [c],
            snake=snakeish,
        )
        
        # Merge buffer dimensions
        out.traversed_buffer_dimension = self.traversed_buffer_dimension or other.traversed_buffer_dimension
        return out

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)