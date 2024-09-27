import networkx as nx
import random
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np

# Parameters
N = 100 # Number of nodes
T = 50  # Number of time steps
base_reputation = 50  # Center of initial random distribution
sigma = 10  # Standard deviation for initial reputation
interaction_prob = 0.1  # Probability of interaction at each time step

# Create the graph
G = nx.barabasi_albert_graph(N, 3)

# Step 1: Assign initial reputation values
for node in G.nodes:
    G.nodes[node]['reputation'] = random.gauss(base_reputation, sigma)

# Step 2: Initialize neighbor reputations (now that all node reputations are assigned)
for node in G.nodes:
    G.nodes[node]['neighbor_reputations'] = {n: G.nodes[n]['reputation'] for n in G.neighbors(node)}

# Step 3: Interaction Function
def interaction(G, node, neighbor):
    neighbor_reputation = G.nodes[neighbor]['reputation']
    base_prob_coop = 0.5  # Base probability of cooperation
    coop_prob = base_prob_coop + (neighbor_reputation - base_reputation) / (2 * base_reputation)
    coop_prob = np.clip(coop_prob, 0, 1)  # Ensure it's within [0,1]
    return random.random() < coop_prob  # True for cooperative, False for uncooperative

# Step 4: Update reputation based on interaction
def update_reputation(G, node, neighbor, cooperative):
    delta = 10  # Amount by which reputation changes
    if cooperative:
        G.nodes[node]['reputation'] += delta
        G.nodes[neighbor]['reputation'] += delta
    else:
        G.nodes[node]['reputation'] -= delta
        G.nodes[neighbor]['reputation'] -= delta
    
    # Update the neighbor reputations tracking
    G.nodes[node]['neighbor_reputations'][neighbor] = G.nodes[neighbor]['reputation']
    G.nodes[neighbor]['neighbor_reputations'][node] = G.nodes[node]['reputation']

# Step 5: Color nodes based on reputation (Red for low, Green for high)
def get_node_color(reputation):
    max_rep = base_reputation + sigma  # Max reputation
    min_rep = base_reputation - sigma  # Min reputation

    # Normalize the reputation to a value between 0 and 1
    normalized_rep = (reputation - min_rep) / (max_rep - min_rep)
    normalized_rep = np.clip(normalized_rep, 0, 1)  # Ensure it's within [0,1]

    return (1 - normalized_rep, normalized_rep, 0)  # Red to green scale
# Step 6: Visualize the graph dynamically
plt.ion()  # Turn on interactive mode
layout = nx.spring_layout(G)
# Simulate interactions over time
for t in range(T):
    for node in G.nodes:
        if random.random() < interaction_prob:  # Chance to initiate interaction
            neighbors = list(G.neighbors(node))
            if neighbors:
                neighbor = random.choice(neighbors)
                cooperative = interaction(G, node, neighbor)
                update_reputation(G, node, neighbor, cooperative)

    # Step 7: Update the plot after each time step
    colors = [get_node_color(G.nodes[node]['reputation']) for node in G.nodes]

    # Clear the previous plot 
    plt.clf()
    # Redraw the graph with new colors but keepign the same layout
    nx.draw(G, pos=layout, with_labels=True, node_color=colors)
    


    plt.title(f"Time Step: {t+1}")
    plt.pause(0.5)  # Pause to allow the visualization to update (adjust the pause time as needed)

plt.ioff()  # Turn off interactive mode after the simulation is done
plt.show()  # Display the final graph



# plot the reputation values

reputation_values = [G.nodes[node]['reputation'] for node in G.nodes]
plt.hist(reputation_values, bins=20)
plt.xlabel('Reputation')
plt.ylabel('Frequency')
plt.title('Distribution of Reputation Values')
plt.show()
