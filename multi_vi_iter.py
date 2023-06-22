"""
Multi-Objective Value Iteration:
\mathcal{V}_{n} \gets Prune \left({\bigcup}_{n' \in N_G(n)} \overrightarrow{c} (n,n')+\mathcal{V}_{n'}\right)
where for Prune we follow the methods pareto_dominates, p_prune and pvi from
Roijers, D. M., Röpke, W., Nowe, A., & Radulescu, R. (2021).
On Following Pareto-Optimal Policies in Multi-Objective Planning and Reinforcement Learning.
Paper presented at Multi-Objective Decision Making Workshop 2021.
http://modem2021.cs.nuigalway.ie/papers/MODeM_2021_paper_3.pdf
https://github.com/rradules/POP-following/tree/main
"""

import copy
import time
import numpy as np


def pareto_dominates(a, b):
    """Check if the vector in b Pareto dominates vector a.

    Note: The original code has been modified to work for our minimization problem.

    Args:
        a (ndarray): A numpy array.
        b (ndarray): A numpy array.

    Returns:
        bool: Whether vector b dominates vector a.
    """
    a = np.array(a)
    b = np.array(b)
    return np.all(a <= b) and np.any(a < b)


def p_prune(candidates):
    """Create a Pareto coverage set from a set of candidate points.

    References:
        .. [1] Roijers, D. M., & Whiteson, S. (2017). Multi-objective decision making. 34, 129–129.
            https://doi.org/10.2200/S00765ED1V01Y201704AIM034

    Args:
        candidates (Set[Tuple]): A set of vectors.

    Returns:
        Set[Tuple]: A Pareto coverage set.
    """
    pcs = set()
    while candidates:
        vector = candidates.pop()

        for alternative in candidates:
            if pareto_dominates(alternative, vector):
                vector = alternative

        to_remove = set(vector)
        for alternative in candidates:
            if pareto_dominates(vector, alternative):
                to_remove.add(alternative)

        candidates -= to_remove
        pcs.add(vector)
    return pcs


def pvi(G, T, objectives):
    """
    Pareto Value Iteration, a.k.a. Multi-Objective Value Iteration
    :param G: Multi-objective search graph G = (V, E)
    :param T: Terminating (ending) node
    :param objectives: Objectives
    :return: The set of value vectors for each node
    """

    start = time.time()  # Timer
    nd_vectors = [set([tuple(np.full(2, np.inf)) for _ in range(len(G.nodes))]) for _ in range(len(G.nodes))]  # Initialisation of nodes
    j = 0  # Counter for iterations

    for n, current_node in enumerate(G.nodes):
        if current_node == T:  # We've reached the terminal state
            nd_vectors[n] = set([(0, 0) for _ in G.nodes])  # The set of value vectors for T is always (0,0)
            break

    while True:  # Run until convergence
        old_vectors = copy.deepcopy(nd_vectors)

        for n, current_node in enumerate(G.nodes):
            if current_node == T:
                continue

            for nk, neighbor in enumerate(G.nodes):
                if neighbor not in G.neighbors(current_node):
                    continue

                edge = G[current_node][neighbor]
                edge_list = [v for k, v in edge.items()]  # Stores only the values of the edges' properties

                cost = []
                for i in objectives:
                    cost.append(edge_list[0][i])

                cost = np.array(cost)
                results = nd_vectors[n].copy()

                for value_vec in nd_vectors[nk]:
                    results.add(tuple(cost+value_vec))  # The set of candidate vectors

                results = p_prune(results)  # Pareto pruning

                nd_vectors[n] = results

        j += 1

        if nd_vectors == old_vectors:  # Check for convergence
            break

        nd_vectors = copy.deepcopy(nd_vectors)  # Else perform a deep copy and go again

    print(f'Iterations: {j}')
    end = time.time()
    elapsed_seconds = (end - start)
    print("Seconds elapsed: " + str(elapsed_seconds))

    return nd_vectors

