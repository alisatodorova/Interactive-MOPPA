'''
Depth-first search (DFS) algorithm, guided by the lower bounds,
obtained from the Single-objective value iteration (i.e., single_vi.py)
'''


def dfs_lower(G, S, T, lower_bounds):
    """
    Finds the shortest path from S to T, guided by the lower bounds
    :param G: Multi-objective search graph G = (V, E)
    :param S: Starting node
    :param T: Terminating (ending) node
    :param lower_bounds: Lower bounds, obtained from the Single-objective value iteration (i.e., single_vi.py)
    :return Shortest path
    """

    stack = [(S, 0, [S])]  # (starting node, cost, path)
    # where cost is from previous_state to S, path is from S to current_state (i.e., S)

    while stack:
        current_node, cost, path = stack.pop()  # cost=total cost up to the current_node

        if current_node == T:
            print(f"Path {path} with cost {cost}")
            return path

        next_node = G[current_node]

        for neighbor, edge_cost in next_node.items():  # edge_cost=cost from current_node to neighbor
            total_cost = cost + edge_cost
            new_path = path + [neighbor]  # Add the neighbor to the path
            stack.append((neighbor, total_cost, new_path))
            stack.sort(key=lambda x: lower_bounds[x[0]],
                       reverse=True)  # Sorts in descending order w.r.t. the lower bound

    print("Sorry! No solution was found!")
    return None
