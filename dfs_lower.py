"""
Depth-first search (DFS) algorithm, guided by the lower bounds,
obtained from the Single-objective value iteration (i.e., single_vi_iter.py)
"""

import numpy as np
import single_vi_iter
import time


def dfs_lower(G, S, T, t, U, max_iter=None):
    """
    Given a target t, the method finds the shortest path from S to T, guided by the lower bounds.
    :param G: Multi-objective search graph G = (V, E)
    :param S: Starting node
    :param T: Terminating (ending) node
    :param t: Target
    :param U: Upper bounds, computed in the outer_loop.py
    :param max_iter: Maximum iterations. By default (i.e., max_iter=None), the algorithm runs until convergence.
    For experimenting with stopping criteria, set the max_iter to a number
    :return Shortest path and its cost; Updated upper bounds
    """
    start = time.time()

    lower_length = single_vi_iter.single_value_iter(G, T, 'length')
    lower_crossing = single_vi_iter.single_value_iter(G, T, 'crossing')  # Change objectives as needed
    # lower_crossing = single_vi_iter.single_value_iter(G, T, '>2.9m')

    i = 0  # track iterations of the algorithm

    cost_history = np.array([0, 0])  # Cost of the path we've seen so far

    stack = [(S, cost_history, [S])]  # (starting node, cost so far, path), where path is from S to current_state

    current_best_path = []

    while stack:
        current_node, current_cost, path = stack.pop()  # current_cost is the total cost up to the current_node

        if current_node == T:
            if np.all(np.less_equal(current_cost, U)):
                U = current_cost  # Update the upper bound as full exact path to T is an upper bound with value=current_cost
                current_best_path = path.copy()

            continue

        neighbor_list = []  # List of all neighbor nodes

        for neighbor in G.neighbors(current_node):
            edge = G[current_node][neighbor]
            edge_list = [v for k, v in edge.items()]  # Stores only the values of the edges' properties

            # cost = np.array([edge['length'], edge['crossing']]) #Test
            cost = np.array([edge_list[0]['length'], edge_list[0]['crossing']])  # Cost in both objectives to go from S to neighbor
            # cost = np.array([edge_list[0]['length'], edge_list[0]['>2.9m']]) # Change objectives as needed

            result = current_cost + cost + np.array([lower_length[neighbor], lower_crossing[neighbor]])  # This is the new lower bound

            # Pruning paths that won't be Pareto-better compared to the current upper bound
            if np.any(np.greater(result, U)):  # If it's outside of target region, ignore it
                continue

            distance = np.sum(np.abs(t - result))  # Manhattan distance to see how close we are to the target

            neighbor_list.append((neighbor, distance, (current_cost + cost)))

        # The goal is to be as close as possible to the target
        neighbor_list.sort(key=lambda x: x[1], reverse=True)  # Sorts in descending order w.r.t. distance
        for n in neighbor_list:
            path_copy = path.copy()
            path_copy.append(n[0])
            stack.append((n[0], n[2], path_copy))  # (neighbor, cost, path)

        i += 1
        if max_iter is not None and i >= max_iter:
            print("The algorithm has reached the given maximum iterations, but has found no solution.")
            break

    print(f'DFS iterations: {i}')
    end = time.time()
    elapsed_seconds = (end - start)
    print("DFS time elapsed in seconds: " + str(elapsed_seconds))

    return current_best_path, U
