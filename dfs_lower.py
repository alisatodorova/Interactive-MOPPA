'''
Depth-first search (DFS) algorithm, guided by the lower bounds,
obtained from the Single-objective value iteration (i.e., single_vi.py)
'''


def dfs_lower(G, S, t, lower_bounds, max_iter=None):
    """
    Finds the shortest path from S to T, guided by the lower bounds
    :param G: Multi-objective search graph G = (V, E)
    :param S: Starting node
    :param t: Target node
    :param lower_bounds: Lower bounds, obtained from the Single-objective value iteration (i.e., single_vi.py)
    :param max_iter: Maximum iterations. By default (i.e., max_iter=None), the algorithm runs until convergence.
    For experimenting with stopping criteria, set the max_iter to a number
    :return Shortest path
    """

    i = 0  # track iterations of the algorithm

    stack = [(S, 0, [S])]  # (starting node, cost, path)
    # where cost is from previous_state to S, path is from S to current_state (i.e., S)

    while stack:
        current_node, cost, path = stack.pop()  # cost=total cost up to the current_node

        if current_node == t:
            print(f"Path {path} with cost {cost}")
            return path

        next_node = G[current_node]

        for neighbor, edge_cost in next_node.items():  # edge_cost=cost from current_node to neighbor
            total_cost = cost + edge_cost
            new_path = path + [neighbor]  # Add the neighbor to the path
            stack.append((neighbor, total_cost, new_path))
            stack.sort(key=lambda x: lower_bounds[x[0]],
                       reverse=True)  # Sorts in descending order w.r.t. the lower bound

        i += 1
        if max_iter is not None and i >= max_iter:
            print("The algorithm has reached the given maximum iterations, but has found no solution.")
            return None

    print("The algorithm has terminated, but no solution was found.")
    return None
