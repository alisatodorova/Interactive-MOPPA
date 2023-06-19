# Import libraries
import geopandas as gpd
import momepy
import networkx as nx
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib
import outer_loop

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
map_amsterdam = gpd.read_file("Sidewalk_width_crossings_smaller.geojson")

# Objectives
objectives = ('length', 'crossing')
# objectives = ('length', '0.9-1.8m')
# objectives = ('length', '1.8-2.9m')
# objectives = ('length', '<0.9m')
# objectives = ('length', '>2.9m')

# Create a NetworkX graph from the map
G = momepy.gdf_to_nx(map_amsterdam, approach='primal')


#Smaller map: ~800 nodes
#ex1:
S = (122245.37633330293, 486126.8581684635) #very first node
T = (122320.31466476223, 486327.5294561802)

#ex2:
# S = (122245.37633330293, 486126.8581684635) #very first node
# T = (122320.31466476223, 486327.5294561802) #very last node

# S = (122245.37633330293, 486126.8581684635)
# T = (122384.20250442973, 486270.65737816785) #AxisError

#Small map: radius 250m and 1000 nodes
# ex6
# S = (120549.11715177551, 486040.41438763676) #not very first node
# T = (120939.06590611176, 485820.4983572203)

#ex7
# S = (120548.6120283842, 486088.19577846595) #vey first
# T = (121015.06629881046, 485829.2834579833) #very last

#Full map ~11401 nodes and radius 800m
#ex1
# S = (119998.5393221767, 485722.64175419795) #very first
# T = (121544.5105401219, 486594.5264401745) #very last

#ex2 ~0.5km distance
# S = (120107.50109162027, 485143.8697083206)
# T = (120016.87004460393, 485271.50645493163)

#ex3 ~ 1.4km distance
# S = (120522.88677087355, 485884.8214429696)
# T = (120773.95779829, 485200.20212105685)

# #ex
# S = (120722.03948820339, 485331.8028751559)
# T = (121552.83295955718, 485778.4098562119)

#ex
# S = (121230.02895176529, 485470.03881079936)
# T = (121128.56143258157, 485478.90656836913)
# S = (121173.2397207758, 485502.6246644313)
# T = (121147.78161871164, 485427.8479137057) #use this
# T = (121167.58299078527, 485558.52833030646) #use this
# S = (121301.20803747902, 485508.92503021995)
# T = (121174.2829937735, 485512.55548926245)


# Distance between S and T
for i in objectives:
    p_ST = nx.shortest_path(G, source=S, target=T, weight=i, method='dijkstra')  # Dijkstra's algorithm
    # Computes the total cost associated with the path and objective, i.e., the value of the path
    distance = nx.path_weight(G, path=p_ST, weight='length')
print(f"Distance between S and T is {distance*0.001}km.")

# The path from our proposed algorithm
t, p_star, val_vector_p_star, P = outer_loop.outer(G, S, T, objectives)
print(f"Target {t}; Path {p_star} with cost {val_vector_p_star}")

# Alternative paths from the Pareto set P
for i, path in enumerate(P):
    if path != p_star:
        print(f"Alternative path {i}:", path)


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
file_name = 'ex70.png'
file_path = folder_path + '/' + file_name
plt.savefig(file_path, bbox_inches='tight')
