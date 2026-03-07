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
                state.path_vertices,
                state.path_vertices_and_neighbors,
                # optionally include max_dim_crossed if it affects successors / legality
                getattr(state, "max_dim_crossed", None),
            )
        else:
            # Normal mode: you can still key by full path if you want,
            # but a safer "identity" is also (head, visited bitmap).
            # If you truly need exact path identity, keep tuple(state.path).
            return (
                0,
                state.head,
                state.path_vertices,
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
