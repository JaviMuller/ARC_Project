import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt

# Parametersss
N = 50 # Number of nodes
num_simulations = 5 # Number of simulationsss per percentage
T = 50 # Number of time steps
initial_wealth = 10  # Initial wealth for all nodes
cooperation_gain = 0.2  # Gain when both cooperate
defection_loss = 0.05 # Loss when both defect
coop_defect_loss = 1  # Loss when one cooperates and the other defects for the cooperator
coop_defect_gain = 0.4  # Gain when one cooperates and the other defects for the defector
standing_increase = 1  # Standing increase for cooperation
standing_decrease = 1  # Standing decrease for defection
random_interaction_prob = 0.0005  # 0.05% chance of random interaction

# Probabilities of interacting with neighbors at different distances
interaction_probs = [0.8, 0.15, 0.04, 0.01]
#make sure the sum of the probabilities is 1



# Node behaviors
COOPERATOR = 'cooperator'
DEFECTOR = 'defector'
DISCRIMINATOR = 'discriminator'




# Assign initial wealth and standing to each node, distribution of defectors will depend on value fed to the function, descriminators and coopeartors will be assigned randomly
def assign_initial_values(N, defector_fraction):
    G = nx.barabasi_albert_graph(N, 3)  # Create a Barabasi-Albert graph with N nodes and 3 edges to attach from a new node
    for node in G.nodes:
        G.nodes[node]['wealth'] = initial_wealth
        G.nodes[node]['standing'] = 5  # Standing starts at 5
        if random.random() < defector_fraction:
            G.nodes[node]['behavior'] = DEFECTOR
        else:
            # Assign a random behavior between cooperator and descriminator
            G.nodes[node]['behavior'] = random.choice([COOPERATOR, DISCRIMINATOR])
    return G
            


# Helper function to choose a node's neighbor based on interaction probabilities
def choose_neighbor(G, node):
    neighbors = {1: list(G.neighbors(node))}
    
   
    for i in range(2, 5):
        neighbors[i] = []
        for n in neighbors[i-1]:
            neighbors[i] += list(G.neighbors(n))
        neighbors[i] = list(set(neighbors[i]) - set(neighbors[1]))  # Avoid overlap with 1st degree
    
    degree = np.random.choice([1, 2, 3, 4], p=interaction_probs)
    if neighbors[degree]:
        return random.choice(neighbors[degree])
    return None

def discriminator_decision(G, discriminator, neighbor):
    max_standing = 5  # Adjust based on your model
    neighbor_standing = G.nodes[neighbor]['standing']
    
    # Higher standing -> Higher probability of cooperation
    cooperation_prob = neighbor_standing / max_standing
    
    # Make decision based on probability
    if random.random() < cooperation_prob:
        return 'cooperate'
    else:
        return 'defect'




def interact(G, node_a, node_b):
    behavior_a = G.nodes[node_a]['behavior']
    behavior_b = G.nodes[node_b]['behavior']
    
  
    if behavior_a == COOPERATOR and behavior_b == COOPERATOR:
        G.nodes[node_a]['wealth'] += cooperation_gain
        G.nodes[node_b]['wealth'] += cooperation_gain
        G.nodes[node_a]['standing'] += standing_increase
        G.nodes[node_b]['standing'] += standing_increase
    
    elif behavior_a == DEFECTOR and behavior_b == DEFECTOR:
        G.nodes[node_a]['wealth'] -= defection_loss
        G.nodes[node_b]['wealth'] -= defection_loss
        G.nodes[node_a]['standing'] -= standing_decrease
        G.nodes[node_b]['standing'] -= standing_decrease
    
    elif behavior_a == COOPERATOR and behavior_b == DEFECTOR:
        G.nodes[node_a]['wealth'] -= coop_defect_loss
        G.nodes[node_b]['wealth'] += coop_defect_gain
        G.nodes[node_a]['standing'] -= standing_decrease
        G.nodes[node_b]['standing'] -= standing_decrease
    
    elif behavior_a == DEFECTOR and behavior_b == COOPERATOR:
        G.nodes[node_a]['wealth'] += coop_defect_gain
        G.nodes[node_b]['wealth'] -= coop_defect_loss
        G.nodes[node_a]['standing'] -= standing_decrease
        G.nodes[node_b]['standing'] -= standing_decrease
    
    elif behavior_a == DISCRIMINATOR and behavior_b != DISCRIMINATOR:
        decision_a = discriminator_decision(G, node_a, node_b)  # Discriminator A decides based on B's standing
        if decision_a == 'cooperate' and behavior_b == COOPERATOR:
            G.nodes[node_a]['wealth'] += cooperation_gain
            G.nodes[node_b]['wealth'] += cooperation_gain
            G.nodes[node_a]['standing'] += standing_increase
            G.nodes[node_b]['standing'] += standing_increase
        if decision_a == 'cooperate' and behavior_b == DEFECTOR:
            G.nodes[node_a]['wealth'] -= coop_defect_loss
            G.nodes[node_b]['wealth'] += coop_defect_gain
            G.nodes[node_a]['standing'] += standing_increase
            G.nodes[node_b]['standing'] += standing_increase
        if decision_a == 'defect' and behavior_b == COOPERATOR:
            G.nodes[node_a]['wealth'] += coop_defect_gain
            G.nodes[node_b]['wealth'] -= coop_defect_loss
            G.nodes[node_a]['standing'] += standing_increase
            G.nodes[node_b]['standing'] += standing_increase
        if decision_a == 'defect' and behavior_b == DEFECTOR:
            G.nodes[node_a]['wealth'] -= defection_loss
            G.nodes[node_b]['wealth'] -= defection_loss
            G.nodes[node_a]['standing'] -= standing_decrease
            G.nodes[node_b]['standing'] -= standing_decrease
    elif  behavior_a != DISCRIMINATOR and behavior_b == DISCRIMINATOR:
        decision_b = discriminator_decision(G, node_b, node_a)  # Discriminator B decides based on A's standing
        if decision_b == 'cooperate' and behavior_a == COOPERATOR:
            G.nodes[node_a]['wealth'] += cooperation_gain
            G.nodes[node_b]['wealth'] += cooperation_gain
            G.nodes[node_a]['standing'] += standing_increase
            G.nodes[node_b]['standing'] += standing_increase
        if decision_b == 'cooperate' and behavior_a == DEFECTOR:
            G.nodes[node_a]['wealth'] -= coop_defect_loss
            G.nodes[node_b]['wealth'] += coop_defect_gain
            G.nodes[node_a]['standing'] += standing_increase
            G.nodes[node_b]['standing'] += standing_increase
        if decision_b == 'defect' and behavior_a == COOPERATOR:
            G.nodes[node_a]['wealth'] += coop_defect_gain
            G.nodes[node_b]['wealth'] -= coop_defect_loss
            G.nodes[node_a]['standing'] += standing_increase
            G.nodes[node_b]['standing'] += standing_increase
        if decision_b == 'defect' and behavior_a == DEFECTOR:
            G.nodes[node_a]['wealth'] -= defection_loss
            G.nodes[node_b]['wealth'] -= defection_loss
            G.nodes[node_a]['standing'] -= standing_decrease
            G.nodes[node_b]['standing'] -= standing_decrease
    elif behavior_a == DISCRIMINATOR and behavior_b == DISCRIMINATOR:
        decision_a = discriminator_decision(G, node_a, node_b)  # Discriminator A decides based on B's standing
        decision_b = discriminator_decision(G, node_b, node_a)  # Discriminator B decides based on A's standing
        if decision_a == 'cooperate' and decision_b == 'cooperate':
            G.nodes[node_a]['wealth'] += cooperation_gain
            G.nodes[node_b]['wealth'] += cooperation_gain
            G.nodes[node_a]['standing'] += standing_increase
            G.nodes[node_b]['standing'] += standing_increase
        if decision_a == 'cooperate' and decision_b == 'defect':
            G.nodes[node_a]['wealth'] -= coop_defect_loss
            G.nodes[node_b]['wealth'] += coop_defect_gain
            G.nodes[node_a]['standing'] += standing_increase
            G.nodes[node_b]['standing'] += standing_increase
        if decision_a == 'defect' and decision_b == 'cooperate':
            G.nodes[node_a]['wealth'] += coop_defect_gain
            G.nodes[node_b]['wealth'] -= coop_defect_loss
            G.nodes[node_a]['standing'] += standing_increase
            G.nodes[node_b]['standing'] += standing_increase
        if decision_a == 'defect' and decision_b == 'defect':
            G.nodes[node_a]['wealth'] -= defection_loss
            G.nodes[node_b]['wealth'] -= defection_loss
            G.nodes[node_a]['standing'] -= standing_decrease
            G.nodes[node_b]['standing'] -= standing_decrease

def simulate_across_defector_percentages(num_simulations, num_nodes, T):
    total_wealth_per_percentage = []
    percentages = [i / 100 for i in range(101)]  # From 0% to 100% discriminators

    for percentage in percentages:
        total_wealth_list = []
        
        for _ in range(num_simulations):
            G = assign_initial_values(num_nodes, percentage)
            run_simulation(G, T)
            total_wealth_list.append(calculate_total_wealth(G))
        
        avg_total_wealth = sum(total_wealth_list) / num_simulations
        total_wealth_per_percentage.append(avg_total_wealth)

    return percentages, total_wealth_per_percentage




def get_standing_distribution(G):
    standing_values = [G.nodes[node]['standing'] for node in G.nodes]
    
    # Use the maximum standing value to determine the range
    max_standing = max(standing_values)
    standing_distribution = np.bincount(standing_values) / len(G.nodes)
    
    # Plot with a dynamic range
    plt.bar(range(max_standing + 1), standing_distribution[:max_standing + 1])
    plt.xlabel('Standing Value')
    plt.ylabel('Fraction of Nodes')
    plt.title('Standing Distribution')
    plt.show()



def get_wealth_distribution(G):
    # Convert wealth values to integers
    wealth_values = [int(G.nodes[node]['wealth']) for node in G.nodes]
    
    # Use the maximum wealth value to determine the range
    max_wealth = max(wealth_values)
    wealth_distribution = np.bincount(wealth_values) / len(G.nodes)
    
    # Plot with a dynamic range
    plt.bar(range(max_wealth + 1), wealth_distribution[:max_wealth + 1])
    plt.xlabel('Wealth Value')
    plt.ylabel('Fraction of Nodes')
    plt.title('Wealth Distribution')
    plt.show()

        
def plot_results(percentages, total_wealth_per_percentage):
    plt.plot(percentages, total_wealth_per_percentage, marker='o')
    plt.title('Percentage of Defectors vs. Total Wealth')
    plt.xlabel('Percentage of Defectors')
    plt.ylabel('Total Wealth of Population')
    plt.grid(True)
    plt.show()
          
        




# Simulate the network over time and save wealth sums for all defectors and cooperators to be graphed later
# Initialize lists to store total wealth of defectors and cooperators at each time step
cooperator_wealth_sum_over_time = []
defector_wealth_sum_over_time = []
defector_standing_over_time = []
cooperator_standing_over_time = []
discriminator_wealth_sum_over_time = []
discriminator_standing_over_time = []

def run_simulation(G, T):
    # Lists to store wealth and standing values over time
    cooperator_wealth_sum_over_time = []
    defector_wealth_sum_over_time = []
    cooperator_standing_over_time = []
    defector_standing_over_time = []
    discriminator_wealth_sum_over_time = []
    discriminator_standing_over_time = []

    for t in range(T):
        cooperator_wealth_sum = 0  # Initialize total wealth for cooperators at each step
        defector_wealth_sum = 0    # Initialize total wealth for defectors at each step
        cooperator_standing_sum = 0
        defector_standing_sum = 0
        discriminator_wealth_sum = 0
        discriminator_standing_sum = 0

        for node in G.nodes:
            # Only interact if the node has at least 1 unit of wealth
            if G.nodes[node]['wealth'] <= 0:
                continue

            # Try to interact with a neighbor based on probabilities
            neighbor = choose_neighbor(G, node)
            if neighbor is not None:
                interact(G, node, neighbor)

            # Chance to interact with a completely random node
            if random.random() < random_interaction_prob:
                random_node = random.choice(list(G.nodes))
                if random_node != node:
                    interact(G, node, random_node)

            # Track wealth and standing based on behavior type
            if G.nodes[node]['behavior'] == COOPERATOR:
                cooperator_wealth_sum += G.nodes[node]['wealth']
                cooperator_standing_sum += G.nodes[node]['standing']
            elif G.nodes[node]['behavior'] == DEFECTOR:
                defector_wealth_sum += G.nodes[node]['wealth']
                defector_standing_sum += G.nodes[node]['standing']
            elif G.nodes[node]['behavior'] == DISCRIMINATOR:
                discriminator_wealth_sum += G.nodes[node]['wealth']
                discriminator_standing_sum += G.nodes[node]['standing']

        # Store sums at the current time step
        cooperator_wealth_sum_over_time.append(cooperator_wealth_sum)
        defector_wealth_sum_over_time.append(defector_wealth_sum)
        cooperator_standing_over_time.append(cooperator_standing_sum)
        defector_standing_over_time.append(defector_standing_sum)
        discriminator_wealth_sum_over_time.append(discriminator_wealth_sum)
        discriminator_standing_over_time.append(discriminator_standing_sum)


def calculate_total_wealth(G):
    return sum(G.nodes[node]['wealth'] for node in G.nodes)


# Function to display information about the network
def display_network_info(G):
    cooperators = [n for n in G.nodes if G.nodes[n]['behavior'] == COOPERATOR]
    defectors = [n for n in G.nodes if G.nodes[n]['behavior'] == DEFECTOR]
    discriminators = [n for n in G.nodes if G.nodes[n]['behavior'] == DISCRIMINATOR]
    
    print(f"Total Cooperators: {len(cooperators)}")
    print(f"Total Defectors: {len(defectors)}")
    print(f"Total Discriminators: {len(discriminators)}")
    
    # Example: Access specific node's wealth, standing, and behavior
    node = random.choice(list(G.nodes))
    print(f"\nNode {node} Info:")
    print(f"Wealth: {G.nodes[node]['wealth']}")
    print(f"Standing: {G.nodes[node]['standing']}")
    print(f"Behavior: {G.nodes[node]['behavior']}")




percentages, total_wealth_per_percentage = simulate_across_defector_percentages(num_simulations, N, T)
plot_results(percentages, total_wealth_per_percentage)



plt.plot(cooperator_wealth_sum_over_time, label="Cooperators' Total Wealth", color='green')
plt.plot(defector_wealth_sum_over_time, label="Defectors' Total Wealth", color='red')
plt.plot(discriminator_wealth_sum_over_time, label="Discriminators' Total Wealth", color='blue')

plt.xlabel('Time Step')
plt.ylabel('Total Wealth')
plt.title('Wealth of Cooperators, Defectors and discriminators Over Time')
plt.legend()
plt.show()

# Plot the standing of cooperators, defectors and discriminators over time
plt.plot(cooperator_standing_over_time, label="Cooperators' Total Standing", color='green')
plt.plot(defector_standing_over_time, label="Defectors' Total Standing", color='red')
plt.plot(discriminator_standing_over_time, label="Discriminators' Total Standing", color='blue')

plt.xlabel('Time Step')
plt.ylabel('Total Standing')
plt.title('Standing of Cooperators, Defectors and discriminators Over Time')
plt.legend()
plt.show()




def get_node_wealths(G, behavior):
    return [G.nodes[node]['wealth'] for node in G.nodes if G.nodes[node]['behavior'] == behavior]

def plot_wealth_evolution():
    cooperators_wealth = get_node_wealths(COOPERATOR)
    defectors_wealth = get_node_wealths(DEFECTOR)
    
    plt.figure(figsize=(10, 6))
    plt.plot(cooperators_wealth, label='Cooperators', color='blue')
    plt.plot(defectors_wealth, label='Defectors', color='red')
    plt.xlabel('Node Index')
    plt.ylabel('Wealth')
    plt.title('Evolution of Wealth Over Time')
    plt.legend()
    plt.show()
