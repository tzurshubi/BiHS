import heapq
import itertools

class HeapqState:
    def __init__(self):
        self.heap = []
        self._counter = itertools.count()  # tie‑breaker

    def push(self, state, f_value):
        # We want: max‑f, then max‑g, then FIFO (or insertion order)
        # => negate f and g to use Python's min‑heap,
        #    then use a counter so states never have to compare to each other.
        entry = (
            -f_value,          # primary key (largest f first)
            -state.g,          # secondary key (largest g first). optional tie-breaker.
            next(self._counter),  # tertiary key (insertion order)
            state
        )
        heapq.heappush(self.heap, entry)

    def pop(self):
        nf, ng, _, state = heapq.heappop(self.heap)
        return -nf, -ng, state
        # nf, _, state = heapq.heappop(self.heap)
        # return -nf, None, state

    def top(self):
        nf, ng, _, state = self.heap[0]
        return -nf, -ng, state
        # nf, _, state = self.heap[0]
        # return -nf, None, state

    def __len__(self):
        return len(self.heap)
