'''
Outer-loop: Selecting the target direction
'''

import inner_loop
import networkx as nx

import gaussian_process
import acquisition_function

def outer(G, S, T):
    """
    Selects the target direction
    :param G: Multi-objective search graph G = (V, E)
    :param S: Starting node
    :param T: Terminating (ending) node
    :return Recommended path p and its value (cost) v_p
    """

    '''
    The Gaussian process and Acquisition function are used as blackboxes from
    "Ordered Preference Elicitation Strategies for Supporting Multi-Objective Decision Making"
    by Luisa M. Zintgraf, Diederik M. Roijers, Sjoerd Linders, Catholijn M. Jonker, and Ann Nowé,
    which was published at AAMAS (Autonomous Agents and Multi-Agent Systems), Stockholm 2018.
    https://github.com/lmzintgraf/gp_pref_elicit
    '''
    #TODO: import/initialise GP and acquisition function
    gp = gaussian_process.GPPairwise(num_objectives=2)
    af = acquisition_function.DiscreteAcquirer()

    u = gp.initialise_gaussian_process()  # Initialise the Gaussian process
    u_hat = af.initialise_acquirer()  # Initialise acquisition function

    P = set() #Pareto set

    # Path initialisation
    for i in objectives:
        p = nx.shortest_path(G, source = i, method = 'dijkstra')
        P = P.add(p)

    return p, v_p
