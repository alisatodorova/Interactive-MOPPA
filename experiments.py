# Import libraries
import geopandas as gpd
import momepy
import networkx as nx
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib
import outer_loop
from lmzintgraf_gp_pref_elicit.gp_utilities import utils_user

# Legend
legend_text = []
legend_labels = []
# Colors
cmap = matplotlib.colormaps['tab10']  # Colormap
exclude_colors = ['green', 'blue']
new_cmap = [color for color in cmap.colors if color not in exclude_colors]
new_cmap = colors.ListedColormap(new_cmap)
# Plot
fig, ax = plt.subplots(figsize=(14, 14), dpi=600)

# Map
map_amsterdam = gpd.read_file("Sidewalk_width_crossings.geojson")

# Objectives
objectives = ('length', 'crossing')

# Create a NetworkX graph from the map
G = momepy.gdf_to_nx(map_amsterdam, approach='primal')

#Full map ~11401 nodes and radius 800m
S = (119998.5393221767, 485722.64175419795) # very first
T = (121544.5105401219, 486594.5264401745) # very last

#Small map
# S = (120548.6120283842, 486088.19577846595)
# T = (121015.06629881046, 485829.2834579833)

# Distance between S and T
p_ST = nx.shortest_path(G, source=S, target=T, weight='length', method='dijkstra')  # Dijkstra's algorithm
# Computes the total cost associated with the path and objective, i.e., the value of the path
distance = nx.path_weight(G, path=p_ST, weight='length')
print(f"Distance between S and T is {distance*0.001}km.")

# The path from my proposed algorithm
t, p_star, val_vector_p_star, p_star_utility, P, val_p = outer_loop.outer(G, S, T, objectives)
print(f"Target {t}; Path with cost {val_vector_p_star}")

# Alternative paths from the Pareto set P
for i, path in enumerate(P):
    if path != p_star:
        print(f"Alternative path {i} with cost {val_p}")

# pvi_paths = [(757.2800000000002, 4.0)]
# user_preference = utils_user.UserPreference(num_objectives=2, std_noise=0.1, seed=123)  # seed=123
# add_noise = False
# PVI_utility = user_preference.get_preference(pvi_paths, add_noise=add_noise)
# print(f"Ground-truth utility for paths from PVI: {np.max(PVI_utility)}")
# regret = np.subtract(np.max(PVI_utility), p_star_utility)
# print(f"Regret:{regret}")  # The difference in utility between the path we recommend in the end and an optimal path.


### Plot experiments ###

# All nodes and edges
nx.draw(G, {n: [n[0], n[1]] for n in list(G.nodes)}, ax=ax, node_size=3)

# Color p* in green
p_star_edges = list(zip(p_star[:-1], p_star[1:]))
green_p_star = 'green'
green_p_star_lwidth = 2.5 + len(P)  # Line width
legend_label = f"Path p*"
nx.draw_networkx_edges(G, pos={n: [n[0], n[1]] for n in list(G.nodes)}, edgelist=p_star_edges, ax=ax,
                       edge_color=green_p_star, width=green_p_star_lwidth)
legend_text.append(
    plt.Line2D([], [], color=green_p_star, linestyle='-', linewidth=green_p_star_lwidth, label=legend_label))
legend_labels.append(legend_label)


# Color the remaining paths in P
for i, path in enumerate(P):
    path_edges = list(zip(path[:-1], path[1:]))
    color = new_cmap(i % new_cmap.N)
    lwidth = 1.0 + (i % len(P)) * 1.0  # Line width

    if path != p_star:
        legend_label = f"Alternative path"
        nx.draw_networkx_edges(G, pos={n: [n[0], n[1]] for n in list(G.nodes)}, edgelist=path_edges, ax=ax,
                               edge_color=color, width=lwidth)
        legend_text.append(
            plt.Line2D([], [], color=color, linestyle='-', linewidth=lwidth, label=legend_label))
        legend_labels.append(legend_label)


# Start & end node
ax.scatter(S[0], S[1], c='yellow', marker='o', s=100, label='Starting Node')
ax.scatter(T[0], T[1], c='red', marker='o', s=100, label='Terminating Node')
ax.text(S[0], S[1], 'S', fontsize=20, ha='center', va='bottom')
ax.text(T[0], T[1], 'T', fontsize=20, ha='center', va='bottom')
start_text = plt.scatter([], [], c='yellow', marker='o', s=100)
end_text = plt.scatter([], [], c='red', marker='o', s=100)
start_label = 'Starting Node'
end_label = 'Ending Node'

# Create legend
all_handles = [start_text, end_text] + legend_text
all_labels = [start_label, end_label] + legend_labels
ax.legend(handles=all_handles, labels=all_labels)
ax.legend(handles=all_handles, labels=all_labels, loc='upper left')

# Save the image
folder_path = 'experiments'
file_name = 'ex28.png'
file_path = folder_path + '/' + file_name
plt.savefig(file_path, bbox_inches='tight')
