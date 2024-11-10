import heapq


class HeapqState:
    def __init__(self):
        self.heap = []
        self.index = 0

    def push(self, state, f_value):
        # Push a new state onto the heap with the given f_value
        heapq.heappush(self.heap, (-f_value, self.index, state, f_value))
        self.index += 1

    def pop(self):
        # Pop the state with the highest priority
        return heapq.heappop(self.heap)

    def top(self):
        # Return the state with the highest priority without removing it
        return self.heap[0]

    def __len__(self):
        return len(self.heap)
    
    def __iter__(self):
        # Allow iteration over the heap elements
        return iter(self.heap)