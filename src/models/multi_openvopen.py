from collections import defaultdict
from models.state import State
from models.openvopen import Openvopen


class MultiOpenvopen:
    """
    Store states (simple paths) for multiple segments, and find the longest
    non-overlapping concatenation of one path per segment.

    Example segments:
        seg1: s -> v1
        seg2: v1 -> v2
        seg3: v2 -> t

    We assume that for each segment, all stored states already have the correct
    (start, goal) endpoints; concatenation is done by skipping the duplicate
    meeting vertex between segments.
    """

    def __init__(self, graph, snake: bool, n, segments, start, goal, solution_vertices):
        """
        multiOPENvOPEN with n vertices.
        For each segment (u,v) we maintain simple paths (u,...,v) in buckets, indexed by their length.
        :param snake: whether to use path_vertices_and_neighbors_bitmap (snake)
                      or path_vertices_bitmap (ordinary simple path).
        """
        self.graph = graph
        self.snake = snake
        self.n = n
        self.start = start
        self.goal = goal
        self.solution_vertices = solution_vertices
        self.openvopen = Openvopen(graph, start, goal)
        self.segments = segments
        for segment_key, segment in self.segments.items():
            segment["paths"] = [[] for _ in range(n)] # paths indexed by length (bucketed by g)


        self.counter = 0  # total number of stored states (paths)

    def _state_bitmap_and_len(self, state):
        bitmap = (
            state.path_vertices_and_neighbors_bitmap
            if self.snake
            else state.path_vertices_bitmap
        )
        length = len(state.path) - 1
        return bitmap, length

    def insert_state(self, segment_id, state):
        """
        Insert a state (simple path) into a given segment.
        segment_id can be any hashable key (e.g. 0, 1, "s-v1", etc.)
        """
        bitmap, length = self._state_bitmap_and_len(state)
        self._segments[segment_id].append(
            {"state": state, "bitmap": bitmap, "length": length}
        )
        self.counter += 1

    def remove_state(self, segment_id, state):
        """
        Remove a specific state from a segment.
        Linear search; you can optimize if needed.
        """
        entries = self._segments.get(segment_id, [])
        new_entries = []
        removed = False
        for entry in entries:
            if entry["state"] is state:
                removed = True
                self.counter -= 1
                continue
            new_entries.append(entry)
        self._segments[segment_id] = new_entries
        return removed

    def     add_paths_to_segment(self, segment_key, seg_paths):
        """
        Add multiple simple paths (lists of vertices) to a segment.
        Creates State objects internally.
        """
        for path_state in seg_paths:
            self.segments[segment_key]["paths"][path_state.g].append(path_state)
            self.counter += 1
            
    def update_su_vt_paths(self, segment_key, vu_paths):
        """
        Given new paths in the segment {v,u}, update openvopen with all concatenations
        of these paths with existing paths in other segments {s,v} and {t,u}.
        """
        longest_path_added = None
        longest_path_added_len = -1
        for vu_path in vu_paths:
            v = vu_path.path[0]
            u = vu_path.path[-1]
            uv_path = State.from_reversed(vu_path)

            # For each sv_path in OPENvOPEN that doesn’t overlap with vu_path:
                # Add su_path = sv_path*vu_path to OPENvOPEN
                # P1 = find_longest_non_overlapping_state(su_path)
            s_v_paths = self.openvopen.get_states_ending_in(v, is_f=True)
            if v == self.start: s_v_paths = [State(self.graph, [], self.snake)]
            for s_v_path in s_v_paths:
                if not uv_path.shares_vertex_with(s_v_path, self.snake):
                    new_su_path = s_v_path + vu_path
                    self.openvopen.insert_state(new_su_path,is_f=True)
                    _, _, st_path, st_path_len, _, _,  = self.openvopen.find_longest_non_overlapping_state(new_su_path,is_f=True, best_path_length=longest_path_added_len, f_max=float('inf'), snake=self.snake)
                    if st_path_len > longest_path_added_len:
                        longest_path_added_len = st_path_len
                        longest_path_added = st_path

            # For each ut_path in OPENvOPEN that doesn’t overlap with vu_path:
                # Add vt_path = vu_path*ut_path to OPENvOPEN
                # P2 = find_longest_non_overlapping_state(vt_path)
            t_u_paths = self.openvopen.get_states_ending_in(u, is_f=False)
            if u == self.goal: t_u_paths = [State(self.graph, [], self.snake)]
            for t_u_path in t_u_paths:
                if not vu_path.shares_vertex_with(t_u_path, self.snake):
                    new_vt_path = vu_path + t_u_path
                    self.openvopen.insert_state(new_vt_path,is_f=False)
                    _, _, st_path, st_path_len, _, _,  = self.openvopen.find_longest_non_overlapping_state(new_vt_path,is_f=False, best_path_length=longest_path_added_len, f_max=float('inf'), snake=self.snake)
                    if st_path_len > longest_path_added_len:
                        longest_path_added_len = st_path_len
                        longest_path_added = st_path
            
        return longest_path_added, longest_path_added_len

    def __len__(self):
        return self.counter
