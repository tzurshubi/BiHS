from utils.utils import *

def BHK(graph, start, goal, args=None):
    """
    Solves the Longest Simple Path problem using Bellman-Held-Karp Dynamic Programming.
    
    Parameters:
    - graph: An adjacency list (dict or list) where graph[u] returns an iterable of neighbors.
             Alternatively, an object with a .neighbors(u) method.
    - start: The starting vertex (integer).
    - goal: The target vertex (integer).
    - args: Optional object/dict for passing additional parameters.
    
    Returns:
    - longest_path: A list of vertices representing the longest simple path from start to goal.
    - stats: A dictionary containing search statistics.
    """
        
    stats = {
        'expansions': 0,
        'states_generated': 0,
        'max_depth_reached': 0
    }
    
    # State tracking: 
    # dp[(visited_mask, current_vertex)] = length_of_longest_path
    # parent[(visited_mask, current_vertex)] = previous_vertex
    start_mask = 1 << start
    dp = { (start_mask, start): 0 }
    parent = { (start_mask, start): None }
    
    # Process layer by layer (BFS style based on path length)
    current_layer = { (start_mask, start): 0 }
    
    longest_path_length = -1
    best_end_state = None
    
    depth = 0
    while current_layer:
        next_layer = {}
        stats['max_depth_reached'] = depth
        
        for (mask, u), length in current_layer.items():
            stats['expansions'] += 1
            
            # If we reach the goal, record it. We do not expand further from the goal
            # because we are looking for a simple path *to* the goal.
            if u == goal:
                if length > longest_path_length:
                    longest_path_length = length
                    best_end_state = (mask, u)
                continue
                
            for v in graph.neighbors(u):
                # Check if neighbor 'v' has already been visited using bitwise AND
                if not (mask & (1 << v)):
                    new_mask = mask | (1 << v)
                    new_state = (new_mask, v)
                    new_length = length + 1
                    
                    stats['states_generated'] += 1
                    
                    # If this is a newly discovered state, or we found a longer path to it
                    if new_state not in dp or new_length > dp[new_state]:
                        dp[new_state] = new_length
                        parent[new_state] = u
                        next_layer[new_state] = new_length
                        
        current_layer = next_layer
        depth += 1

    
    # Reconstruct the path via backpointers
    path = []
    if best_end_state is not None:
        curr = best_end_state
        while curr is not None:
            mask, u = curr
            path.append(u)
            prev_u = parent.get(curr)
            
            if prev_u is not None:
                # Toggle off the current vertex's bit to find the previous mask
                prev_mask = mask ^ (1 << u)
                curr = (prev_mask, prev_u)
            else:
                curr = None
        path.reverse()
    
    return path, stats