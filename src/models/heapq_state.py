# import heapq
# import itertools

# class HeapqState:
#     def __init__(self):
#         self.heap = []
#         self._counter = itertools.count()  # tie-breaker

#     def push(self, state, f_value):
#         # We want: max-g, then max-f, then FIFO (insertion order).
#         # Negate g and f to use Python's min-heap; counter preserves stability.
#         entry = (
#             -state.g,             # primary key (largest g first)
#             -f_value,             # secondary key (largest f first)
#             next(self._counter),  # tertiary key (insertion order)
#             state
#         )
#         heapq.heappush(self.heap, entry)

#     def pop(self):
#         ng, nf, _, state = heapq.heappop(self.heap)  # ng = -g, nf = -f
#         f = -nf
#         g = -ng
#         return f, g, state

#     def top(self):
#         ng, nf, _, state = self.heap[0]
#         f = -nf
#         g = -ng
#         return f, g, state

#     def __len__(self):
#         return len(self.heap)

# SECOND PRIORITY BY G (MAX-F, MAX-G, FIFO)

# import heapq
# import itertools

# class HeapqState:
#     def __init__(self):
#         self.heap = []
#         self._counter = itertools.count()  # tie‑breaker

#     def push(self, state, f_value):
#         # We want: max‑f, then max‑g, then FIFO (or insertion order)
#         # => negate f and g to use Python's min‑heap,
#         #    then use a counter so states never have to compare to each other.
#         entry = (
#             -f_value,          # primary key (largest f first)
#             -state.g,          # secondary key (largest g first). optional tie-breaker.
#             next(self._counter),  # tertiary key (insertion order)
#             state
#         )
#         heapq.heappush(self.heap, entry)

#     def pop(self):
#         nf, ng, _, state = heapq.heappop(self.heap)
#         return -nf, -ng, state
#         # nf, _, state = heapq.heappop(self.heap)
#         # return -nf, None, state

#     def top(self):
#         nf, ng, _, state = self.heap[0]
#         return -nf, -ng, state
#         # nf, _, state = self.heap[0]
#         # return -nf, None, state

#     def __len__(self):
#         return len(self.heap)

# Version commented out above: Nov 2025

# import heapq
# import itertools

# class HeapqState:
#     def __init__(self):
#         self.heap = []
#         self._counter = itertools.count()
#         # map: path_key (tuple of vertices) -> entry
#         # (the entry is the "currently alive" heap entry for that path)
#         self._alive = {}

#     def _path_key(self, state):
#         """Convert a state's path to a hashable key."""
#         return tuple(state.path)

#     def push(self, state, f_value):
#         """
#         Push a state keyed by its path.
#         If another state with the same path exists, the new one becomes the
#         only 'alive' entry for that path; the old one will be lazily discarded.
#         """
#         path_key = self._path_key(state)

#         # largest f first, then largest g, then FIFO
#         entry = [
#             -f_value,              # primary key (max f)
#             -state.g,              # secondary key (max g)
#             next(self._counter),   # tertiary key (insertion order)
#             state                  # payload
#         ]
#         heapq.heappush(self.heap, entry)
#         # this path_key now refers to this entry as the alive one
#         self._alive[path_key] = entry

#     def remove(self, state):
#         """
#         Lazily remove a state by its path.
#         Any state with the same path (even a different State instance) will be
#         considered removed, because we key only by the path.
#         """
#         path_key = self._path_key(state)
#         entry = self._alive.pop(path_key, None)
#         if entry is not None:
#             # Mark the entry as removed; _clean_top will get rid of it later.
#             entry[-1] = None

#     def _clean_top(self):
#         """
#         Remove invalid or stale entries from the top of the heap.
#         Called by pop/top.
#         """
#         while self.heap:
#             nf, ng, _, state = self.heap[0]
#             # If already marked removed
#             if state is None:
#                 heapq.heappop(self.heap)
#                 continue

#             path_key = self._path_key(state)
#             current_entry = self._alive.get(path_key)

#             # If this path is no longer alive, or the alive entry is not this one,
#             # then this heap entry is stale and should be popped.
#             if current_entry is not self.heap[0]:
#                 heapq.heappop(self.heap)
#                 continue

#             # Otherwise, top of heap is valid and alive
#             break

#     def pop(self):
#         """
#         Pop the best (f, g, state), skipping removed/stale entries.
#         """
#         self._clean_top()
#         if not self.heap:
#             raise IndexError("pop from empty HeapqState")

#         nf, ng, _, state = heapq.heappop(self.heap)
#         path_key = self._path_key(state)
#         # Remove from alive map if still there
#         self._alive.pop(path_key, None)
#         return -nf, -ng, state

#     def top(self):
#         """
#         Peek at the best (f, g, state) without removing it.
#         """
#         self._clean_top()
#         if not self.heap:
#             raise IndexError("top from empty HeapqState")
#         nf, ng, _, state = self.heap[0]
#         return -nf, -ng, state

#     def __len__(self):
#         # Clean stale entries so len() reflects only alive states
#         self._clean_top()
#         return len(self._alive)

#     def is_empty(self):
        # """Return True if no alive states remain in the heap."""
        # self._clean_top()
        # return len(self._alive) == 0


# Version commented out above: Dec 2025

import heapq
import itertools

class HeapqState:
    def __init__(self):
        self.heap = []
        self._counter = itertools.count()
        self._alive = {}

    def _state_key(self, state):
        """
        Hashable identity of a State.
        Works when state.path is None (snake mode).
        """
        # If you want to distinguish normal vs snake, include a flag.
        snake_mode = state.snake 

        if snake_mode:
            return (
                1,  # snake flag
                state.head,
                state.path_vertices_bitmap,
                state.path_vertices_and_neighbors_bitmap,
                # optionally include max_dim_crossed if it affects successors / legality
                getattr(state, "max_dim_crossed", None),
            )
        else:
            # Normal mode: you can still key by full path if you want,
            # but a safer "identity" is also (head, visited bitmap).
            # If you truly need exact path identity, keep tuple(state.path).
            return (
                0,
                tuple(state.path),
            )

    def push(self, state, f_value):
        key = self._state_key(state)

        entry = [
            -f_value,            # max f
            -state.g,            # max g
            next(self._counter), # FIFO tiebreak
            key,                 # store key so we don't recompute from state later
            state
        ]
        heapq.heappush(self.heap, entry)
        self._alive[key] = entry

    def remove(self, state):
        key = self._state_key(state)
        entry = self._alive.pop(key, None)
        if entry is not None:
            entry[-1] = None  # mark as removed

    def _clean_top(self):
        while self.heap:
            nf, ng, _, key, state = self.heap[0]

            if state is None:
                heapq.heappop(self.heap)
                continue

            current = self._alive.get(key)
            if current is not self.heap[0]:
                heapq.heappop(self.heap)
                continue

            break

    def pop(self):
        self._clean_top()
        if not self.heap:
            raise IndexError("pop from empty HeapqState")

        nf, ng, _, key, state = heapq.heappop(self.heap)
        self._alive.pop(key, None)
        return -nf, -ng, state

    def top(self):
        self._clean_top()
        if not self.heap:
            raise IndexError("top from empty HeapqState")
        nf, ng, _, key, state = self.heap[0]
        return -nf, -ng, state

    def __len__(self):
        self._clean_top()
        return len(self._alive)

    def is_empty(self):
        self._clean_top()
        return len(self._alive) == 0
