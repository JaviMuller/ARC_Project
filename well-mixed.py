import numpy as np
import random
import matplotlib.pyplot as plt

# Parameters
N = 50 # Number of nodes
num_simulations = 20 # Number of simulationsss per percentage
T = 50 # Number of time steps
initial_fitness = 100  # Initial fitness for all nodes
coop_gain = 0.2  # Gain when both cooperate
def_loss = -0.05 # Loss when both defect
suck_payoff = -1  # Loss when one cooperates and the other defects for the cooperator
temptation = 0.6  # Gain when one cooperates and the other defects for the defector

coop_coop_gain = 2*coop_gain
coop_def_gain = suck_payoff + temptation
def_def_gain = 2*def_loss

COOPERATOR = 1
DISCRIMINATOR = 0
DEFECTOR = -1

GOOD = True
BAD = False

def init(n_nodes, epochs, f_coop, f_def, f_disc, perf_inf):
    # Initialization of variables
    global nodes, fitness, standing, reputation, sum_fitness, epoch_total_fitness, epoch_cooperative_act, n_coop, n_def, n_disc, epoch_cooperators, epoch_defectors, epoch_discriminators
    nodes = np.random.choice([COOPERATOR, DEFECTOR, DISCRIMINATOR], size=n_nodes, p=[f_coop, f_def, f_disc])
    fitness = np.full((nodes), initial_fitness, dtype=np.float64)
    standing = np.ones(nodes, dtype = np.bool_)
    reputation = np.ones((nodes, nodes), dtype=np.bool_)

    # Results initialization
    sum_fitness = initial_fitness*n_nodes
    epoch_total_fitness = np.empty(epochs, dtype=np.float64)
    epoch_cooperative_act = np.empty(epochs, dtype=np.int16)
    n_coop = round(n_nodes * f_coop)
    n_def = round(n_nodes * f_def)
    n_disc = round(n_nodes * f_disc)
    epoch_cooperators = np.empty(epochs, dtype=np.int16)
    epoch_defectors = np.empty(epochs, dtype=np.int16)
    epoch_discriminators = np.empty(epochs, dtype=np.int16)
    perf_inf = perf_inf

def decision(node_id, other_id):
    node_t = nodes[node_id]
    other_t = nodes[other_id]
    
    if node_t == COOPERATOR:
        return True
    elif other_t == DEFECTOR:
        return False
    
    # Discriminator
    if perf_inf:
        rep_other = standing[other_id]
    else:
        rep_other = reputation[node_id, other_id]

    return rep_other

def interaction(node1, node2):
    #TODO: Missing update of both the perfect and imperfect informations
    dec_1 = decision(node1, node2, perf_inf)
    dec_2 = decision(node2, node1, perf_inf)

    # COOP-COOP
    if dec_1 and dec_2:
        fitness[node1] += coop_gain
        fitness[node2] += coop_gain
        sum_fitness += coop_coop_gain
    # COOP-DEF
    elif dec_1 and (not dec_2):
        fitness[node1] += suck_payoff
        fitness[node2] += temptation
        sum_fitness += coop_def_gain
    # DEF-COOP
    elif (not dec_1) and dec_2:
        fitness[node1] += temptation
        fitness[node2] += suck_payoff
        sum_fitness += coop_def_gain
    # DEF-DEF
    elif (not dec_1) and (not dec_2):
        fitness[node1] += def_loss
        fitness[node2] += def_loss
        sum_fitness += def_def_gain

