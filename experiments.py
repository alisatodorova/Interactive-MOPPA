# Import libraries
import numpy as np
import geopandas as gpd
import momepy
import networkx as nx
from shapely.geometry import Point
import matplotlib.pyplot as plt
import outer_loop

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

t, p_star, val_vector_p_star = outer_loop.outer(G, S, T, objectives)
print(f"Target {t}; Path {p_star} with cost {val_vector_p_star}")

# Plot experiments
start_node = gpd.GeoDataFrame({'geometry': [Point(119998.5393221767, 485722.64175419795)]})
end_node = gpd.GeoDataFrame({'geometry': [Point(121544.5105401219, 486594.5264401745)]}) #TODO: Is this the S and T I picked or the map's
fig, ax = plt.subplots(figsize=(14, 14), dpi=600)
# All nodes and edges
nx.draw(G, {n: [n[0], n[1]] for n in list(G.nodes)}, ax=ax, node_size=3)
# Start & end node
start_node.plot(ax=ax, color='green')
end_node.plot(ax=ax, color='purple')
# Path p_star
path_edges = list(zip(p_star[:-1], p_star[1:]))
nx.draw_networkx_edges(G, pos={n: [n[0], n[1]] for n in list(G.nodes)}, edgelist=path_edges, ax=ax, edge_color='red', width=2)

# Save the image
folder_path = 'experiments'
file_name = 'ex8.png'
file_path = folder_path + '/' + file_name
plt.savefig(file_path)
