import heapq
import itertools

class HeapqState:
    def __init__(self):
        self.pq_priority = []  # Max-heap sorted strictly by MM priority
        self.pq_f_value = []   # Max-heap sorted strictly by true f-value
        self._counter = itertools.count()
        self._alive = {}       # Maps state_key to (entry_p, entry_f)

    def _state_key(self, state):
        """
        Hashable identity of a State.
        Works when state.path is None (snake mode).
        """
        snake_mode = state.snake 

        if snake_mode:
            return (
                1,  # snake flag
                state.head,
                state.path_vertices,
                state.path_vertices_and_neighbors,
                getattr(state, "max_dim_crossed", None),
            )
        else:
            return (
                0,
                state.head,
                state.path_vertices,
            )

    def push(self, state, priority, f_value):
        key = self._state_key(state)
        count = next(self._counter)

        # entry_p tie-breaks on f_value, then g, then FIFO
        entry_p = [
            -priority,           
            -f_value,            
            -state.g,            
            count,               
            key,                 
            state
        ]
        
        # entry_f tie-breaks on priority, then g, then FIFO
        entry_f = [
            -f_value,            
            -priority,           
            -state.g,            
            count,               
            key,                 
            state
        ]

        heapq.heappush(self.pq_priority, entry_p)
        heapq.heappush(self.pq_f_value, entry_f)
        self._alive[key] = (entry_p, entry_f)

    def remove(self, state):
        key = self._state_key(state)
        entries = self._alive.pop(key, None)
        if entries is not None:
            entries[0][-1] = None  # mark state in entry_p as removed
            entries[1][-1] = None  # mark state in entry_f as removed

    def _clean_top_priority(self):
        while self.pq_priority:
            entry = self.pq_priority[0]
            if entry[-1] is None:
                heapq.heappop(self.pq_priority)
                continue
            
            key = entry[-2]
            alive_entries = self._alive.get(key)
            # Check if this specific entry object is the currently alive one
            if alive_entries is None or alive_entries[0] is not entry:
                heapq.heappop(self.pq_priority)
                continue

            break

    def _clean_top_f_value(self):
        while self.pq_f_value:
            entry = self.pq_f_value[0]
            if entry[-1] is None:
                heapq.heappop(self.pq_f_value)
                continue
            
            key = entry[-2]
            alive_entries = self._alive.get(key)
            # Check if this specific entry object is the currently alive one
            if alive_entries is None or alive_entries[1] is not entry:
                heapq.heappop(self.pq_f_value)
                continue

            break

    def pop(self):
        self._clean_top_priority()
        if not self.pq_priority:
            raise IndexError("pop from empty HeapqState")

        entry_p = heapq.heappop(self.pq_priority)
        neg_priority, neg_f_value, neg_g, count, key, state = entry_p
        
        # Removing from _alive guarantees it will be skipped by _clean_top_f_value
        self._alive.pop(key, None)
        
        return -neg_priority, -neg_f_value, -neg_g, state

    def top(self):
        self._clean_top_priority()
        if not self.pq_priority:
            raise IndexError("top from empty HeapqState")
            
        neg_priority, neg_f_value, neg_g, count, key, state = self.pq_priority[0]
        return -neg_priority, -neg_f_value, -neg_g, state

    def max_f(self):
        """Returns the true maximum f-value in the queue for termination bounds."""
        self._clean_top_f_value()
        if not self.pq_f_value:
            return float("-inf")
        return -self.pq_f_value[0][0]

    def __len__(self):
        # The true number of valid states
        return len(self._alive)

    def is_empty(self):
        return len(self._alive) == 0