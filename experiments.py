# Import libraries
import numpy as np
import geopandas as gpd
import momepy
import networkx as nx
from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib
import outer_loop

legend_text = []  # Legend
cmap = matplotlib.colormaps['tab10']  # Colormap
fig, ax = plt.subplots(figsize=(14, 14), dpi=600)

map = gpd.read_file("Sidewalk_width_crossings.geojson")

# Objectives
objective1 = map['length']
objective2 = map['crossing']
objective3 = map['obstacle_free_width']
objectives = ('length', 'crossing')

# Create a NetworkX graph from the map
G = momepy.gdf_to_nx(map, approach='primal')
nodes = G.nodes
edges = G.edges


#Smaller map:
#ex1:
# S = (122245.37633330293, 486126.8581684635) #very first node
# T = (122246.77932030056, 486223.5791244763) #t = cost

#ex2:
# S = (122245.37633330293, 486126.8581684635) #very first node
# T = (122320.31466476223, 486327.5294561802)

# S = (122245.37633330293, 486126.8581684635)
# T = (122384.20250442973, 486270.65737816785) #AxisError

#Small map:
# ex6
# S = (120549.11715177551, 486040.41438763676) #not very first node
# T = (120939.06590611176, 485820.4983572203)

#ex7
# S = (120548.6120283842, 486088.19577846595) #vey first
# T = (121015.06629881046, 485829.2834579833) #very last

#Full map
#ex8
S = (119998.5393221767, 485722.64175419795) #very first
T = (121544.5105401219, 486594.5264401745) #very last

t, p_star, val_vector_p_star, P = outer_loop.outer(G, S, T, objectives)
# print(f"Target {t}; Path {p_star} with cost {val_vector_p_star}")
for i, path in enumerate(P):
    print(f"Path {i+1}:", path)

### Plot experiments ###

# All nodes and edges
nx.draw(G, {n: [n[0], n[1]] for n in list(G.nodes)}, ax=ax, node_size=3)


# Path p* from DFS
green_path_edges = list(zip(p_star[:-1], p_star[1:]))
green_p_star = 'green'
green_p_star_lwidth = 2.5 + len(P)  # Line width
legend_label = f"Path p* from DFS"
nx.draw_networkx_edges(G, pos={n: [n[0], n[1]] for n in list(G.nodes)}, edgelist=green_path_edges, ax=ax,
                       edge_color=green_p_star, width=green_p_star_lwidth)
legend_text.append(plt.Line2D([], [], color=green_p_star, linestyle='-', linewidth=green_p_star_lwidth, label=legend_label))

# All paths from P
for i, path in enumerate(P):
    if path == p_star:
        continue

    path_edges = list(zip(path[:-1], path[1:]))

    color = cmap(i % cmap.N)
    lwidth = 1.0 + (i % len(P)) * 1.0 # Line width

    legend_label = f"Alternative path"
    nx.draw_networkx_edges(G, pos={n: [n[0], n[1]] for n in list(G.nodes)}, edgelist=path_edges, ax=ax,
                           edge_color=color, width=lwidth)
    legend_text.append(plt.Line2D([], [], color=color, linestyle='-', linewidth=lwidth, label=legend_label))


# Start & end node
ax.scatter(S[0], S[1], c='yellow', marker='o', s=100, label='Starting Node')
ax.scatter(T[0], T[1], c='red', marker='o', s=100, label='Terminating Node')
ax.text(S[0], S[1], 'S', fontsize=20, ha='center', va='bottom')
ax.text(T[0], T[1], 'T', fontsize=20, ha='center', va='bottom')


legend_text = [legend_text[P.index(p)] for p in P]
legend_labels = [handle.get_label() for handle in legend_text]

start_text = plt.scatter([], [], c='yellow', marker='o', s=100)
end_text = plt.scatter([], [], c='red', marker='o', s=100)
start_label = 'Starting Node'
end_label = 'Ending Node'

all_handles = [start_text, end_text] + legend_text
all_labels = [start_label, end_label] + legend_labels
ax.legend(all_handles, all_labels)

# Save the image
folder_path = 'experiments'
file_name = 'ex11.png'
file_path = folder_path + '/' + file_name
plt.savefig(file_path)
