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
    #TODO: import/initialise GP and acquisition function
    gp = gaussian_process.GPPairwise(num_objectives=2)
    af = acquisition_function.DiscreteAcquirer()
    u = gp.initialise_gaussian_process()  # Initialise the Gaussian process
    u_hat = af.initialise_acquirer()  # Initialise acquisition function

    P = set() #Pareto set

    # Path initialisation
    for i in d:
        p = nx.shortest_path(G, source=S, target=T, weight = i, method='dijkstra')
        obj_val = G.nodes[p[-1]][i] #Objectives' values where -1 retrieves the value of the ending node in the path
        P = P.add(p)

    C = (min(obj_val,), min(obj_val)) #Candidate target

    return p, v_p
