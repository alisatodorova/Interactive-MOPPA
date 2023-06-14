# Import libraries
import geopandas as gpd
import momepy
import networkx as nx
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib
import outer_loop


# Create a 5x5 grid graph
G = nx.Graph()

# Add nodes to the graph
for i in range(5):
    for j in range(5):
        G.add_node((i, j))

# Add edges to form the grid structure
for i in range(4):
    for j in range(5):
        G.add_edge((i, j), (i + 1, j))
        G.add_edge((i, j), (i, j + 1))

# Assign costs for length and number of crossings
length_costs = [
    [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6],
    [3, 4, 5, 6, 7],
    [4, 5, 6, 7, 8],
    [5, 6, 7, 8, 9]
]

crossing_costs = [
    [1, 1, 1, 1, 1],
    [1, 2, 2, 2, 1],
    [1, 2, 3, 2, 1],
    [1, 2, 2, 2, 1],
    [1, 1, 1, 1, 1]
]

# Assign the costs for each edge
for u, v in G.edges():
    u_x, u_y = u
    v_x, v_y = v
    length_cost = length_costs[u_x][u_y]
    crossings_cost = crossing_costs[u_x][u_y]
    G[u][v]['length'] = length_cost
    G[u][v]['crossing'] = crossing_costs
print(G.edges(data=True))

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


objectives = ('length', 'crossing')

S = (0, 0)
T = (3, 5)


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
file_name = 'ex17.png'
file_path = folder_path + '/' + file_name
plt.savefig(file_path, bbox_inches='tight')
