import osmnx as ox
import networkx as nx
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt


# # Get the road network for Manhattan, New York, USA
# place_name = "Manhattan, New York, USA"
# G = ox.graph_from_place(place_name, network_type='drive')
# print(len(G.nodes))
# print(len(G.edges))

# # Plot the road network
# ox.plot_graph(G, figsize=(10, 10), node_size=10, edge_linewidth=1)

# # Step 2: Compute the minimum spanning tree (MST)
# # NetworkX’s minimum_spanning_tree function requires weights; here we use 'length' (edge distance)
# mst = nx.minimum_spanning_tree(G.to_undirected(), weight='length')
# print(len(mst.nodes))
# print(len(mst.edges))                               

# # Step 3: Visualize the MST
# fig, ax = ox.plot_graph(mst, edge_color='red', edge_linewidth=1, node_size=10)

# Load NYC taxi data and find nearest nodes
# https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-
def Haversine(lat1, long1, lat2, long2):
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

# Adjust display settings
top_k = 10
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.expand_frame_repr', False)  # Disable column width shrinking
pd.set_option('display.width', 1000)  # Set width large enough to wrap
pd.set_option('display.colheader_justify', 'left')  # Justify column headers for better readability

df = pd.read_csv('./data/yellow_tripdata_2016-02.csv')
print(df.columns)

# Grouping by the 'from' and 'to' columns and counting the occurrences
aggregated_df = df.groupby(['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']).agg(count=('fare_amount', 'size')).reset_index()
aggregated_df = aggregated_df.sort_values(by='count', ascending=False).head(top_k)
print(aggregated_df)

# Construct the small network for which we have trip data.
G = nx.Graph()

for index, row in aggregated_df.iterrows():
    # Extract latitude and longitude
    from_node = (row['pickup_latitude'], row['pickup_longitude'])
    
    if not from_node in G:
        G.add_node(from_node)
    
    to_node = (row['dropoff_latitude'], row['dropoff_longitude'])
    
    if not to_node in G:
        G.add_node(from_node)
        
# Connect all the nodes with each other to create a complete graph.
for node1 in G.nodes:
    for node2 in G.nodes:
        if (node1 != node2 and not (node1, node2) in G.edges):
            # Add the edge
            G.add_edge(node1, node2)

# Visualize the created graph
nx.draw(G, with_labels=False, node_color='lightblue', font_weight='bold', node_size=700)
plt.show()