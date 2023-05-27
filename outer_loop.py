'''
Outer-loop: Selecting the target direction
'''

import inner_loop
import networkx as nx

import gaussian_process
import acquisition_function

def outer(G, S, T, d):
    """
    Selects the target direction
    :param G: Multi-objective search graph G = (V, E)
    :param S: Starting node
    :param T: Terminating (ending) node
    :param d: Objectives
    :return Recommended path p and its value (cost) v_p
    """

    '''
    The Gaussian process and Acquisition function are used as blackboxes from
    "Ordered Preference Elicitation Strategies for Supporting Multi-Objective Decision Making"
    by Luisa M. Zintgraf, Diederik M. Roijers, Sjoerd Linders, Catholijn M. Jonker, and Ann Nowé,
    which was published at AAMAS (Autonomous Agents and Multi-Agent Systems), Stockholm 2018.
    https://github.com/lmzintgraf/gp_pref_elicit
    '''

    gp = gaussian_process.GPPairwise(num_objectives=2, std_noise=0.01, kernel_width=0.15, prior_mean_type='zero', seed=None) #Initialise the Gaussian process for 2 objectives
    # acq_fun = acquisition_function.DiscreteAcquirer(input_domain, query_type='ranking', seed, acquisition_type='expected improvement') #Initialise acquisition function
    #TODO: input_domain and seed are what?

    P = []  # Pareto set

    # Path initialisation
    for i in d:
        p = nx.shortest_path(G, source=S, target=T, weight=i, method='dijkstra')  # Dijkstra algorithm
        P.append(p)

    C = min(P, key=lambda p: (p[0], p[1])) #Candidate target, where the lambda function
    # compares the tuples based on their first and second elements

    # User ranking: Compare paths in P and add comparisons to GP
    #TODO: Compare paths in P and add comparisons to GP

    p_star = max(P, key=lambda p: gp(p)) #path the user likes best and has the maximum a posteriori (MAP) estimate

    while C:
        t = max(C, key=lambda v: acq_fun(v)) #Pick candidate target with highest value acquisition function
        #TODO: remove t from C
        p = inner_loop(t, G, S, T) #TODO: inner-loop

        #TODO: if vp improves in the target region
        P.append(p)
        #TODO: Compare p to p∗ and add comparison to the GP
        if gp(p) > gp(p_star):
            p_star = p
            #TODO: Compute new candidate targets based on vp and add to C
        #End If
    #End While

    return p_star, v_p_star
