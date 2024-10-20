from itertools import combinations
import networkx as nx
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt
import sys

def haversine(lat1, long1, lat2, long2):
    # Convert degrees to radians
    lat1, long1, lat2, long2 = map(radians, [lat1, long1, lat2, long2])
    
    # Apply the formula
    diff_lat = lat2 - lat1
    diff_long = long2 - long1
    
    a = sin(diff_lat / 2)**2 + cos(lat1) * cos(lat2) * sin(diff_long / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    # Earth's radius in miles
    r = 3958.8
    distance = r * c
    return distance

def is_tree(G):
    if (nx.is_connected(G) and len(G.edges) == len(G.nodes) - 1):
        return True
    return False

def generate_spanning_trees(G):
    nodes = list(G.nodes)

    # For each edge subset, check if it is a tree
    for edges in combinations(G.edges(data=True), len(nodes) - 1):
        # Create the graph which has edges equal to |V| - 1.
        T = nx.Graph()
        T.add_nodes_from(nodes)
        for u, v, data in edges:
            T.add_edge(u, v, **data)

        if (is_tree(T)):
            yield T

def exhaustive_search(G, shortest_path_dict):
    # Generate all spanning trees
    counter = 0
    min_cost = sys.float_info.max
    Tmin = None
    for T in generate_spanning_trees(G):
        print('Current spanning tree count =' + str(counter))
        counter += 1

        # Keep track of the tree with the minimum cost
        cost = network_cost(T, shortest_path_dict)
        if (Tmin is None or  cost < min_cost):
            Tmin = T
            min_cost = cost

    # Print the tree with the minimum cost
    plt.clf()
    nx.draw(Tmin, with_labels=False, node_color='red', font_weight='bold', node_size=700)
    plt.show()
    plt.title("Minimum Tree - Exhaustive Search")
    print('Cost of minimum tree including trip costs =' + str(min_cost))

    # Get the minimum spanning tree
    mst = nx.minimum_spanning_tree(G, weight='weight', algorithm='kruskal')

    # Print the minimum spanning tree
    plt.clf()
    nx.draw(mst, with_labels=False, node_color='red', font_weight='bold', node_size=700)
    plt.title("Minimum spanning Tree - Kruskal's Algorithm")
    plt.show()

    print('Cost of minimum spanning tree with kruskal algorithm =' + str(network_cost(mst, shortest_path_dict)))

    if (Tmin == mst):
        print('Minimum tree is the same as minimum spanning tree')
    else:
        print('Minimum tree is not the same as minimum spanning tree')

    return

# Calculate the shortest path for all pairs, update the path dictionary and return the
# all pairs cost.
def all_pairs_shortest_paths(G, shortest_path_dict, exclude_pairs=None):
    # Update the shortest paths
    unique_pairs = list(combinations(G.nodes, 2))
    for u, v in unique_pairs:
        if (exclude_pairs is not None and (u, v) in exclude_pairs):
            continue

        # Find the shortest path
        shortest_path = nx.shortest_path(G, source=u, target=v)

        # Get the edges in the shortest path and update the dictionary
        shortest_path_edges = list(zip(shortest_path[:-1], shortest_path[1:]))
        shortest_path_dict[(u, v)] = shortest_path_edges

    # Calculate all pairs shortest paths
    cost = 0.0
    for u, v in shortest_path_dict.keys():
        trips = 0
        if (G.get_edge_data(u, v) is not None):
            trips = G.get_edge_data(u, v)['trips']

        # For each edge in the shortest path, multiply the weight with the number of trips
        for edge in shortest_path_edges:
            cost += G.edges[edge]['weight'] * trips
    
    return cost

# Network Cost = Sum of Edge Weights + Variable cost of trips
def network_cost(T, shortest_path_dict):
    cost = 0.0
    for u, v, data in T.edges(data=True):
        cost += data['weight']
    
    cost += all_pairs_shortest_paths(G, shortest_path_dict)

    return cost

def reverse_delete(G, shortest_path_dict):
    Gmin = G.copy()
    iters = 0

    while (True):
        # Check if network has transformed in a tree or max number of allowed iterations have occurred.
        if (is_tree(Gmin)):
            break

        # Pick an edge to delete which has the highest effect in decreasing cost.
        min_cost = network_cost(Gmin, shortest_path_dict)
        Gnew = None
        for edge in Gmin.edges:
            Gtemp = Gmin.copy()
            Gtemp.remove_edge(edge[0], edge[1])
            if (nx.is_connected(Gtemp)):
                new_cost = network_cost(Gtemp, shortest_path_dict)
                if (new_cost < min_cost):
                    Gnew = Gtemp.copy()
                    min_cost = new_cost
        
        if (Gnew is not None):
            Gmin = Gnew.copy()
        else:
            break

        iters += 1

    # Print the network with the minimum cost
    print('Iterations = ' + str(iters))
    min_cost = network_cost(Gmin, shortest_path_dict)
    plt.clf()
    nx.draw(Gmin, with_labels=False, node_color='red', font_weight='bold', node_size=700)
    plt.show()
    plt.title("Minimum Tree - Reverse Delete")
    print('Cost of minimum tree including trip costs with reverse delete =' + str(min_cost))

    return

# Adjust display settings
top_k = 5
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.expand_frame_repr', False)  # Disable column width shrinking
pd.set_option('display.width', 1000)  # Set width large enough to wrap
pd.set_option('display.colheader_justify', 'left')  # Justify column headers for better readability

# Load NYC taxi data
# https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data
df = pd.read_csv('./data/yellow_tripdata_2016-02.csv')
print(df.columns)

print(df.head())

# Grouping by the 'from' and 'to' columns and counting the occurrences
agg_df = df.groupby(['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']).agg(count=('fare_amount', 'size')).reset_index()

# Clean data by including only valid pickup and dropoff entries
agg_df = agg_df.loc[(agg_df['pickup_latitude'] != agg_df['dropoff_latitude']) | (agg_df['pickup_longitude'] != agg_df['dropoff_longitude'])]
agg_df = agg_df.sort_values(by='count', ascending=False).head(top_k)
print(agg_df)

# Construct the small network for which we have trip data.
G = nx.Graph()

for index, row in agg_df.iterrows():
    # Extract latitude and longitude
    from_node = (row['pickup_latitude'], row['pickup_longitude'])
    
    if not from_node in G:
        G.add_node(from_node)
    
    to_node = (row['dropoff_latitude'], row['dropoff_longitude'])
    
    if not to_node in G:
        G.add_node(from_node)
    
    # Calculate edge weight 
    distance = haversine(from_node[0], from_node[1], to_node[0], to_node[1])

    # Add the edge
    G.add_edge(from_node, to_node, weight=distance, trips=row['count'])
        
# Add the remaining edges to create a complete graph.
for node1 in G.nodes:
    for node2 in G.nodes:
        if (node1 != node2 and (not G.has_edge(node1, node2))):
            # Calculate edge weight 
            distance = haversine(node1[0],node1[1],node2[0],node2[1])

            # Add the edge
            G.add_edge(node1, node2, weight=distance, trips=0)

# Visualize the created graph
plt.clf()
nx.draw(G, with_labels=False, node_color='lightblue', font_weight='bold', node_size=700)
plt.title("Filtered Small Network")
plt.show()

shortest_path_dict = dict()
# Exhaustive Search Algorithm - Find tree with the minimum cost among all possible spanning trees
exhaustive_search(G, shortest_path_dict)

# Reverse Delete Algorithm - Find the optimal tree by systematically deleting edges
reverse_delete(G, shortest_path_dict)