import numpy as np
import random
import matplotlib.pyplot as plt
from random import uniform
from collections import Counter

# Parameters
N = 50 # Number of nodes
num_simulations = 30 # Number of simulations per setting
T = 50 # Number of time steps
initial_fitness = 100  # Initial fitness for all nodes
info_prop = .61 # 3 degrees of separation 1st degree
num_inter_epoch = N

# Prisioner's dilemma
coop_gain = 0.2  # Gain when both cooperate
def_loss = -0.05 # Loss when both defect
suck_payoff = -1  # Loss when one cooperates and the other defects for the cooperator
temptation = 0.6  # Gain when one cooperates and the other defects for the defector

coop_coop_gain = 2*coop_gain
coop_def_gain = suck_payoff + temptation
def_def_gain = 2*def_loss

# Names
COOPERATOR = 1
DISCRIMINATOR = 0
DEFECTOR = -1

COOPERATE = True
DEFECT = False

GOOD = True
BAD = False

# Aggregate results
res_total_fitness = np.zeros(T, dtype=np.float64)
res_coop_actions = np.zeros(T, dtype=np.float64)
res_cooperators = np.zeros(T, dtype=np.float64)
res_defectors = np.zeros(T, dtype=np.float64)
res_discriminators = np.zeros(T, dtype=np.float64)


def init(f_coop, f_def, f_disc, p_inf):
    
    global nodes, fitness, standing, reputation, epoch_total_fitness, epoch_cooperative_act, \
        epoch_cooperators, epoch_defectors, epoch_discriminators, perfect_inf

    # Initialization of variables
    nodes = np.random.choice([COOPERATOR, DEFECTOR, DISCRIMINATOR], size=N, p=[f_coop, f_def, f_disc])
    fitness = np.full((nodes), initial_fitness, dtype=np.float64)
    standing = np.ones(nodes, dtype=np.bool_)
    reputation = np.ones((nodes, nodes), dtype=np.bool_)

    # Simulation results initialization
    epoch_total_fitness = np.empty(T, dtype=np.float64)
    epoch_cooperative_act = np.empty(T, dtype=np.float64)
    epoch_cooperators = np.empty(T, dtype=np.float64)
    epoch_defectors = np.empty(T, dtype=np.float64)
    epoch_discriminators = np.empty(T, dtype=np.float64)
    perfect_inf = p_inf

def decision(node_id, other_id):
    node_t = nodes[node_id]
    other_t = nodes[other_id]
    
    if node_t == COOPERATOR:
        return COOPERATE
    elif other_t == DEFECTOR:
        return DEFECT
    
    # Discriminator
    if perfect_inf:
        rep_other = standing[other_id]
    else:
        rep_other = reputation[node_id, other_id]

    return rep_other

def update_rep(observer, node1, node2, dec1, dec2, prob_obs):
    # Update with the given probability
    if uniform(0, 1) > prob_obs:
        return
    
    good1 = reputation[observer, node1]
    good2 = reputation[observer, node2]

    # XOR -> Stern judging
    reputation[observer, node1] = dec1 ^ good2
    reputation[observer, node2] = dec2 ^ good1

def interaction(node1, node2):
    dec1 = decision(node1, node2, perfect_inf)
    dec2 = decision(node2, node1, perfect_inf)

    ret = 0
    # Fitness
    ## COOP-COOP
    if dec1 and dec2:
        # Fitness
        fitness[node1] += coop_gain
        fitness[node2] += coop_gain
        ret = 1
    ## COOP-DEF
    elif dec1 and (not dec2):
        fitness[node1] += suck_payoff
        fitness[node2] += temptation
    ## DEF-COOP
    elif dec2:
        fitness[node1] += temptation
        fitness[node2] += suck_payoff
    ## DEF-DEF
    else:
        fitness[node1] += def_loss
        fitness[node2] += def_loss
    
    # Standing
    good1 = standing[node1]
    good2 = standing[node2]
    ## XOR -> stern judging
    standing[node1] = dec1 ^ good2
    standing[node2] = dec2 ^ good1

    # Reputation
    ## (Nodes in the interaction always observe the interaction)
    good1_1 = reputation[node1, node1]
    good1_2 = reputation[node1, node2]
    good2_1 = reputation[node2, node1]
    good2_2 = reputation[node2, node2]
    ## XOR -> stern judging
    new_rep1_1 = dec1 ^ good1_2
    new_rep1_2 = dec2 ^ good1_1
    new_rep2_1 = dec1 ^ good2_2
    new_rep2_2 = dec2 ^ good2_1
    ## Apply update to all nodes with the given probability
    update_observer_rep = lambda observer: update_rep(observer, node1, node2, dec1, dec2, info_prop)
    update_observer_rep(reputation)
    ## Update the interacting nodes' reputation
    reputation[node1, node1] = new_rep1_1
    reputation[node1, node2] = new_rep1_2
    reputation[node2, node1] = new_rep2_1
    reputation[node2, node2] = new_rep2_2
    
    # Returns 1 if it was a cooperation and 0 if it was a defection
    return ret

def run_epoch(id):
    coop_act = 0
    for i in range(num_inter_epoch):
        node1, node2 = np.random.choice(N, 2, False)
        coop_act += interaction(node1, node2)
    
    # Calculate epoch metrics
    epoch_total_fitness[id] = np.sum(fitness)
    epoch_cooperative_act[id] = coop_act
    node_types = Counter(nodes)
    epoch_cooperators[id] = node_types[COOPERATOR]
    epoch_defectors[id] = node_types[DEFECTOR]
    epoch_discriminators[id] = node_types[DISCRIMINATOR]

def run_sim(f_coop, f_def, f_disc, perfect_info):
    global res_total_fitness, res_coop_actions, res_cooperators, res_defectors, res_discriminators

    init(f_coop, f_def, f_disc, perfect_info)
    
    for epoch in range(T):
        run_epoch(epoch)
    
    # Add to result aggregates
    res_total_fitness = np.add(res_total_fitness, epoch_total_fitness)
    res_coop_actions = np.add(res_coop_actions, epoch_cooperative_act)
    res_cooperators = np.add(res_cooperators, epoch_cooperators)
    res_defectors = np.add(res_defectors, epoch_defectors)
    res_discriminators = np.add(res_discriminators, epoch_discriminators)
    