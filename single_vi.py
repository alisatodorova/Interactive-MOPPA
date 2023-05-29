'''
Single-objective value iteration: v_n \gets {\arg\min}_{n' \in N_G(n)} c(n,n')+v_{n'}
'''

import numpy as np


def single_vi(G, max_iter=1000, gamma=1.0, threshold=1e-8):
    """
    Calculates the value vector for each node
    :param G: Single-objective graph G = (V, E)
    :param max_iter: Maximum iterations of the method
    :param gamma: Discount factor, where 1.0 means that future rewards are equally valued to immediate rewards
    :param threshold: Convergence threshold
    :return: Optimal value vector for each node
    """

    num_nodes = len(G)

    v_n = np.zeros(num_nodes) #Initialise each node to 0

    for i in range(max_iter): #or until convergence
        v_n_copy = np.copy(v_n)

        for n in G:
            if not G[n]: #We've reached the terminal state
                continue
            max_value = -np.inf

            for n_next in G[n]:
                cost = G[n][n_next]
                result = np.min(cost + gamma * v_n_copy[n_next])
                max_value = max(max_value, result)

            v_n[n] = max_value

        if np.max(np.abs(v_n - v_n_copy)) < threshold: #Check for convergence
            break

    return v_n
