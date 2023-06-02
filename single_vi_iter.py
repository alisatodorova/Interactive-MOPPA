'''
Single-objective value iteration: v_n \gets {\arg\min}_{n' \in N_G(n)} c(n,n')+v_{n'}
'''

import numpy as np

def single_value_iter(G, max_iter=1000, threshold=1e-8):

    """
    Calculates the value vector for each node
    :param G: Single-objective graph G = (V, E)
    :param max_iter: Maximum iterations of the method
    :param gamma: Discount factor, where 1.0 means that future rewards are equally valued to immediate rewards
    :param threshold: Convergence threshold
    :return: Optimal value vector for each node
    """

    # gamma=1.0
    # num_nodes = len(G)
    #
    # v_n = np.zeros(num_nodes) #Initialise each node to 0
    #
    # for i in range(max_iter): #or until convergence
    #     v_n_copy = np.copy(v_n)
    #
    #     for n in G:
    #         if not G[n]: #We've reached the terminal state
    #             continue
    #
    #         max_value = -np.inf
    #         vals = []
    #
    #         for n_next in G[n]:
    #             cost = G[n][n_next]
    #             result = np.min(cost + gamma * v_n_copy[n_next])
    #             max_value = max(max_value, result)
    #
    #         v_n[n] = max_value
    #
    #     if np.max(np.abs(v_n - v_n_copy)) < threshold: #Check for convergence
    #         break
    #
    # return v_n

    v_n = {}
    threshold = 1e-8
    max_iter = 1000
    for i in range(max_iter):  # or until convergence
        v_n_copy = v_n.copy()
        for n in G:
            max_value = -np.inf
            for n_next in G[n]:
                for key in G:
                    cost = key
                    result = np.min(cost + n_next)
                    max_value = max(max_value, result)
                v_n[n] = max_value

            converged = all(n in v_n_copy and abs(v_n[n] - v_n_copy[n]) < threshold for n in v_n)
            if converged:
                break

    return v_n
