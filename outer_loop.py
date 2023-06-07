"""
Outer-loop: Selecting the target direction

Note: The Gaussian process and Acquisition function are used as blackboxes from
"Ordered Preference Elicitation Strategies for Supporting Multi-Objective Decision Making"
by Luisa M. Zintgraf, Diederik M. Roijers, Sjoerd Linders, Catholijn M. Jonker, and Ann Nowé,
which was published at AAMAS (Autonomous Agents and Multi-Agent Systems), Stockholm 2018.
https://github.com/lmzintgraf/gp_pref_elicit
"""

import numpy as np
import networkx as nx
import time

import dfs_lower

from lmzintgraf_gp_pref_elicit import dataset, gaussian_process, acquisition_function
from lmzintgraf_gp_pref_elicit.gp_utilities import utils_user as utils_user


def outer(G, S, T, d):
    """
    Selects the target direction
    :param G: Multi-objective search graph G = (V, E)
    :param S: Starting node
    :param T: Terminating (ending) node
    :param d: Objectives
    :return Target t; Recommended path p and its value (cost) v_p
    """

    start = time.time()

    # Initialise the Gaussian process for 2 objectives
    gp = gaussian_process.GPPairwise(num_objectives=2, std_noise=0.01, kernel_width=0.15, prior_mean_type='zero', seed=None)

    P = []  # Pareto set
    val_p = []  # value vectors w.r.t. p, i.e., v^{p_1}, v^{p_2}
    val_vector_p_star = []  # value vectors w.r.t. p^*
    compare_ps_pstar = []

    # Path initialisation
    for i in d:
        p = nx.shortest_path(G, source=S, target=T, weight=i, method='dijkstra')  # Dijkstra's algorithm
        P.append(p)

        # Computes the total cost associated with the path and objective, i.e., the value of the path
        val_obj1 = nx.path_weight(G, path=p, weight='length')
        val_obj2 = nx.path_weight(G, path=p, weight='crossing')
        val_p.append(np.array([val_obj1, val_obj2]))

    # Candidate Targets, i.e., the most optimistic points
    C = [min(val_p[0][0], val_p[1][0]), min(val_p[0][1], val_p[1][1])]
    C = [np.array(C)]

    # The most pessimistic points form the upper bounds
    U = [max(val_p[0][0], val_p[1][0]), max(val_p[0][1], val_p[1][1])]

    # User ranking: Compare paths in P
    user_preference = utils_user.UserPreference(num_objectives=2, std_noise=0.1)
    add_noise = True
    ground_utility = user_preference.get_preference(val_p, add_noise=add_noise)  # This is the ground-truth utility, i.e., the true utility

    # Add the comparisons to the GP
    comparisons = dataset.DatasetPairwise(num_objectives=2)
    comparisons.add_single_comparison(val_p[np.argmax(ground_utility)], val_p[np.argmin(ground_utility)])  # This is user ranking of their preferences
    gp.update(comparisons)

    # Find the path the user likes best and has the maximum a posteriori (MAP) estimate
    u_v, _ = gp.get_predictive_params(val_p, True)  # The maximum a posteriori (MAP) estimate is the mean from gaussian_process.get_predictive_params()
    p_star_index = np.argmax(u_v)
    p_star = P[p_star_index]

    # Computes the total cost associated with the path and objective, i.e., the value of the path
    val_p_star1 = nx.path_weight(G, path=p_star, weight='length')
    val_p_star2 = nx.path_weight(G, path=p_star, weight='crossing')
    val_vector_p_star.append(np.array([val_p_star1, val_p_star2]))

    # Initialise the acquisition function
    input_domain = np.array(C)  # set of Candidate targets
    acq_fun = acquisition_function.DiscreteAcquirer(input_domain=input_domain, query_type='ranking', seed=123, acquisition_type='expected improvement')

    # while C:
    while len(C) != 0:
        # Pick the Candidate target which has the highest value from the acquisition function
        expected_improvement = acquisition_function.get_expected_improvement(input_domain, gp, acq_fun.history)
        t_index = np.argmax(expected_improvement)
        t = input_domain[t_index]

        # Remove t from C
        indices = [i for i, x in enumerate(C) if np.all(x == t)]
        for index in sorted(indices, reverse=True):
            del C[index]

        # Inner-loop approach with DFS guided by the lower-bounds computed from the single-objective value iteration
        p_s, val_p_s, new_U = dfs_lower.dfs_lower(G, S, T, t, U) # Change max_iter when doing experiments
        U = new_U

        # If v^p_s improves in the target region
        if np.any(np.less(val_p_s, U)):
            P = P.append(p_s)

            # Compare p^s to p^∗ and add comparison to the GP ▷ User ranking, i.e., is the new path preferred to the current, maximum one?
            compare_ps_pstar.append(np.array([val_p_s, val_vector_p_star]))
            ranking_new_paths = user_preference.get_preference(compare_ps_pstar, add_noise=add_noise)

            # Add the comparisons to the GP
            comparisons.add_single_comparison(compare_ps_pstar[np.argmax(ranking_new_paths)], compare_ps_pstar[np.argmin(ranking_new_paths)])
            gp.update(comparisons)

            # if u(v^{p^s}) > u(v^{p^*}) then
            u_v_p_s, _ = gp.get_predictive_params(val_p_s, True)  # The maximum a posteriori (MAP) estimate is the mean from gaussian_process.get_predictive_params()
            u_v_p_star, _ = gp.get_predictive_params(val_vector_p_star, True)

            if u_v_p_s > u_v_p_star:
                # p^∗ ← p^s
                p_star = p_s

            # Compute new candidate targets based on v^{p^s} and add to C
            new_C = [min(val_p_s[0][0], val_p_s[1][0]), min(val_p_s[0][1], val_p_s[1][1])]
            C = np.append(new_C)

    end = time.time()
    elapsed_seconds = (end - start)
    print("Outer-loop time elapsed in seconds: " + str(elapsed_seconds))

    return t, p_star, val_vector_p_star, P

