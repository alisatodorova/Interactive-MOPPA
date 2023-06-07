"""
Multi-Objective Value Iteration:
\mathcal{V}_{n} \gets Prune \left({\bigcup}_{n' \in N_G(n)} \overrightarrow{c} (n,n')+\mathcal{V}_{n'}\right)
To implement this, we follow the Pareto Value Iteration of Theorem 2 from
D.J White. Multi-objective infinite-horizon discounted markov decision processes. Journal of Mathematical Analysis and Applications,
89(2):639–647, 1982
and the implementation of Willem Röpke. Ramo: Rational agents with multiple objectives. https://github.com/wilrop/mo-game-theory, 2022
"""

import numpy as np

def multi_vi(G, T):
    """
    Multi-objective value iteration
    :param G: Multi-objective search graph G = (V, E)
    :param T: Terminating (ending) node
    :return The set of all value vectors n
    """

    v_n = {}  # Value vector
    objectives = ('length', 'crossing')

    for n in G:
        v_n[n] = np.inf  # Initialisation of nodes

        if n == T:
            v_n[n] = 0

        for neighbor in G.neighbors(n):
            edge = G[n][neighbor]
            edge_list = [v for k, v in edge.items()]  # Stores only the values of the edges' properties
            v_n[neighbor] = np.array([edge_list[0]['length'], edge_list[0]['crossing']])
            for i in objectives:
                cost = edge_list[0][i]

            result = cost + v_n[neighbor]

        v_n[n] = result

    return v_n

#Run this in once a Pareto-front and then back-propagate??

