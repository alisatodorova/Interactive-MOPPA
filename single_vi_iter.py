"""
Single-objective value iteration: v_n \gets {\arg\min}_{n' \in N_G(n)} c(n,n')+v_{n'}
"""

import numpy as np


def single_value_iter(G, T, objective, max_iter=None):

    """
    Calculates the value vector for each node
    :param G: Single-objective graph G = (V, E)
    :param T: Terminating (ending) node
    :param objective: Objective
    :param max_iter: Maximum iterations
    :return: Optimal value vector for each node
    """

    v_n = {}  # Value vector
    next_node = {}
    edge_cost = []

    for n in G:
        v_n[n] = np.inf  # Initialisation of nodes

        if n == T:  # We've reached the terminal state
            v_n[n] = 0

    converged = False
    while not converged: #TODO: Is this correct?
    # for i in range(max_iter):  # or until convergence
        # print(f'Single-Objective Value Iteration number: {i}')
        converged = True

        for e in G.edges(data=True):
            n1, n2 = e[0], e[1]  # The nodes

            # {\arg\min}_{n' \in N_G(n)} c(n,n')+v_{n'}
            cost = e[2][objective]
            edge_cost.append(cost)

            result1 = min(cost + v_n[n2], v_n[n1])
            result2 = min(cost + v_n[n1], v_n[n2])

            if v_n[n1] != result1 or v_n[n2] != result2:
                converged = False

            # Next nodes
            if v_n[n1] != result1:
                next_node[n1] = n2

            if v_n[n2] != result2:
                next_node[n2] = n1

            v_n[n1] = result1
            v_n[n2] = result2

        # converged = True
        if converged:  # Check for convergence
            break

    return v_n
