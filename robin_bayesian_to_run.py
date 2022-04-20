import pickle
import random
from sklearn.linear_model import LinearRegression

import pandas as pd
import seaborn as sns
import networkx as nx
import numpy as np

from scipy.stats import beta

import matplotlib.pyplot as plt

from agents import hateAgent
from misinfo_functions import (
    generate_params_dict,
    step_params_dict,
    calc_energy,
    acceptance_proba,
    make_agent_info_dict,
    update_agent_info,
    choose_orientation,
    getList_ofkeys,
)
from utilities import markov_update_log, make_configuration_model_graph

from sklearn.model_selection import train_test_split



def run_agent_simulation_now_with_added_hate(N_AGENTS, params_dict):
    """
    Given a number of agents & parameters for constructing the simulation,
    run a 100-round simulation of belief updating w/ Bayesian agents.
    
    Returns info on each agent's parameters, sharing for each agent each round, and centrality info for each agent.
    """
    agents = []

    hate_agents = params_dict["hate_agents"]
    counter_hate_agents = params_dict["counter_hate_agents"]

    for i in range(N_AGENTS): # creats the start parameters for each of the agents 
        #each agent gets one of these 
        #if statements are so that hate agents start out hateful and counter hateful agents are not
        if i in hate_agents:
            agent = hateAgent(
                agent_id=i,
                neighbors={},
                forcefulness=np.log(
                    np.random.beta(params_dict["B1_START_FO"], params_dict["B2_START_FO"])
                ),
                share_propensity=np.log(
                    np.random.beta(params_dict["B1_START_SP"], params_dict["B2_START_SP"])
                ),
                misinfo_belief=np.log(
                    np.random.beta(params_dict["B1_START_MB"], params_dict["B2_START_MB"])
                ),
                trust_stability=np.log(
                    np.random.beta(params_dict["B1_START_TS"], params_dict["B2_START_TS"])
                ),
                hatefulness=(
                    np.random.uniform(params_dict["B1_START_Hate_Hatefulnes"], params_dict["B2_START_Hate_Hatefulnes"])
                ),
                hate_orientation= choose_orientation(i, hate_agents,counter_hate_agents)
            )
            # give the hate and counter hate agents a certain amount of hate
            
            agents.append(agent)
        elif i in counter_hate_agents:
            agent = hateAgent(
                agent_id=i,
                neighbors={},
                forcefulness=np.log(
                    np.random.beta(params_dict["B1_START_FO"], params_dict["B2_START_FO"])
                ),
                share_propensity=np.log(
                    np.random.beta(params_dict["B1_START_SP"], params_dict["B2_START_SP"])
                ),
                misinfo_belief=np.log(
                    np.random.beta(params_dict["B1_START_MB"], params_dict["B2_START_MB"])
                ),
                trust_stability=np.log(
                    np.random.beta(params_dict["B1_START_TS"], params_dict["B2_START_TS"])
                ),
                hatefulness=(
                    np.random.uniform(params_dict["B1_START_Counter_Hatefulnes"], params_dict["B2_START_Counter_Hatefulnes"])
                ),
                hate_orientation= choose_orientation(i, hate_agents,counter_hate_agents)
            )
            # give the hate and counter hate agents a certain amount of hate
            
            agents.append(agent)
        else:
            agent = hateAgent(
                agent_id=i,
                neighbors={},
                forcefulness=np.log(
                    np.random.beta(params_dict["B1_START_FO"], params_dict["B2_START_FO"])
                ),
                share_propensity=np.log(
                    np.random.beta(params_dict["B1_START_SP"], params_dict["B2_START_SP"])
                ),
                misinfo_belief=np.log(
                    np.random.beta(params_dict["B1_START_MB"], params_dict["B2_START_MB"])
                ),
                trust_stability=np.log(
                    np.random.beta(params_dict["B1_START_TS"], params_dict["B2_START_TS"])
                ),
                hatefulness=(
                    np.random.uniform(params_dict["B1_START_Hatefulness"], params_dict["B2_START_Hatefulness"])
                ),
                hate_orientation= choose_orientation(i, hate_agents,counter_hate_agents)
            )
            # give the hate and counter hate agents a certain amount of hate
            
            agents.append(agent)

    # G, agents = make_er_graph(0.05, N_AGENTS, agents)
    G, agents = make_configuration_model_graph(N_AGENTS, 2.5, agents, params_dict)

    centrality = sorted(
        [(k, v) for k, v in nx.closeness_centrality(G).items()], key=lambda b: b[0]
    )

    centrality = np.array([c[1] for c in centrality]).reshape(-1, 1) # gets the centrality for each agent 
    agent_records = {a.agent_id: {} for a in agents} #creates the agent records
    shares = {a.agent_id: {} for a in agents} #creates the number of shares in agent records
    shares_hate = {a.agent_id: {} for a in agents} 
    hate_orientation_time = {a.agent_id: {} for a in agents} #creates the orientation time keeper 
    hatefulness_time = {a.agent_id: {} for a in agents} #creates the orientation time keeper 

    # from multiprocessing import Pool
    # pool = Pool(8)

    #records - everything
    #agent info dict - just the most previous timestep 
    for time_step in range(250):
        for agent in agents:
            agent_records[agent.agent_id][time_step] = { #creates a dictionary in agent records where the agent id and time step record the trust, misinfo beleif and share propensity 
                "neighbor_trust": agent.neighbors,
                "misinfo_belief": agent.misinfo_belief,
                "share_propensity": agent.share_propensity,
                "hatefulness": agent.hatefulness,
            }

            

        neighbor_beliefs = [ # gets the neighbor beliefs 
            [(i, agents[i].misinfo_belief) for i in agent.neighbors.keys()]
            for agent in agents
        ]
        neighbor_beliefs_hate = [ # gets the neighbor beliefs 
            [(i, agents[i].hatefulness) for i in agent.neighbors.keys()]
            for agent in agents
        ]
        neighbor_forcefulness = [
            [agents[i].forcefulness for i in agent.neighbors.keys()] for agent in agents
        ]
        neighbor_orientation = [
            [agents[i].hate_orientation for i in agent.neighbors.keys()] for agent in agents
        ]
        
        agent_info_dicts = [  #the most recent agent info, agent records goes all the way back 
            make_agent_info_dict(a, b, f, h, ho, params_dict)
            for a, b, f, h, ho in zip(agents, neighbor_beliefs, neighbor_forcefulness,neighbor_beliefs_hate, neighbor_orientation)
        ]
        
        res = map(update_agent_info, agent_info_dicts) # map means you apply the function to every item in agent info dicts 
        for r, agent in zip(res, agents): #r is the updated agent information 
            # so this updates all of the info for the agents, its a dictionary 
            agent.neighbors = r["neighbor_trust"]
            agent.misinfo_belief = r["misinfo_belief"]
            agent.share_propensity = r["share_propensity"]
            
            shares[agent.agent_id][time_step] = r["shares"]

            #newstuff
            shares_hate[agent.agent_id][time_step] = r["shares_hate"] # to scrap 
            agent.hatefulness = r["hatefulness"]
            agent.hate_orientation = r["hate_orientation"]
            hate_orientation_time[agent.agent_id][time_step] = r["hate_orientation"]
            hatefulness_time[agent.agent_id][time_step] = r["hatefulness"]


    return agents, shares, centrality, hate_orientation_time, hatefulness_time


def p_x_y_hate(agents, hate_orientation_time, centrality, alpha):
    """
    For a set of Bayesian agents at the end of a simulation,
    plus their sharing results & centrality metrics, report 
    the loss of the dataset relative to reality
    
    (note that reality is currently somewhat made up)
    alpha is a scaling parameter.
    
    Returns sum of loss fns, each raised to alpha then divided by alpha

    to do add quanitative info 
    """
    loss = 0.0
    # shared = [np.sum([v for v in shares_hate[a.agent_id].values()]) for a in agents]
    # shared_by_id = [
    #     (a.agent_id, np.sum([v for v in shares[a.agent_id].values()])) for a in agents
    # ]
    # shared_by_id = sorted(shared_by_id, key=lambda b: b[0])
    # shared_by_id = [s[1] for s in shared_by_id]
    # find correlation between sharing and centrality,
    # with the assumption that more central nodes share more.
    # we need to get a quantitative measurement for this.
    
    # in any spreading process we should expect that the spreading process is somewhat correlated with centrality 
    final_hatefulness = []
    for i in range(100):
        final_hatefulness.append(hate_orientation_time[i][249])
    # hatefulness_by_id = [
    #     (([final_hatefulness[a.agent_id]])) for a in agents
    # ]
    #final_hatefulness = np.asarray(hatefulness_by_id)

    
    
    #find out how many agents there were by each type 
    number_counter = final_hatefulness.count(2)
    percentage_counter = number_counter/len(agents)
    percentage_counter_real = 0.3 #made this up for testing
    loss += np.abs(percentage_counter - percentage_counter_real) ** alpha

    number_hateful = final_hatefulness.count(1)
    percentage_hatefulness = number_hateful/len(agents)
    percentage_hatefulness_real =0.3 #made up for testing 
    loss += np.abs(percentage_hatefulness - percentage_hatefulness_real) ** alpha

    # # find out how much misinfo was shared by the top 1% of misinfo sharers
    # # using L1 loss here.
    # shared_by_top_one_percent_model = np.sum(
    #     sorted(shared)[-int(0.01 * len(shared)) :]
    # ) / (1 + np.sum(shared))
    # shared_by_top_one_percent_real = 0.8 # info from lazer lab twitter metrics
    # loss += np.abs(shared_by_top_one_percent_model - shared_by_top_one_percent_real) ** alpha

    # # find out how much sharing per capita occurred
    # # again using L1 loss
    # n_shared_per_capita_model = np.sum(shared) / len(agents)
    # n_shared_per_capita_real = 1.0
    # loss += np.abs(n_shared_per_capita_model - n_shared_per_capita_real) ** alpha
    dataset = pd.DataFrame({'centrality': list(centrality), 'hatefulness': list(final_hatefulness)}, columns=['centrality', 'hatefulness'])
    #print(dataset)
    X = dataset['hatefulness']
    Y = dataset['centrality']
    show_dummy = dataset[['hatefulness']]
    dummy_hate = pd.get_dummies(show_dummy['hatefulness'], prefix="hate")
    
    
    model = LinearRegression()
    model.fit(dummy_hate,Y)
    coeff_parameter = pd.DataFrame(model.coef_,dummy_hate.columns,columns=['Coefficient'])
    
    if len(model.coef_) == 3:
        centrality_to_n_shared_model_hate = model.coef_[1]
        centrality_to_n_shared_model_counter = model.coef_[2]
        centrality_to_n_shared_real_hate = 0.05 # arbitrary guess
        centrality_to_n_shared_real_counter = 0.05 # arbitrary guess
        loss += np.abs(centrality_to_n_shared_model_hate  - centrality_to_n_shared_real_hate) ** alpha
        loss += np.abs(centrality_to_n_shared_model_counter  - centrality_to_n_shared_real_counter) ** alpha
    else:
        loss = 0 
    
    return loss / alpha
    
    
def G_func(my_ensemble_P, x):
    """
    Get smoothed probability of a simulation outcome x
    given my_ensemble_P, which is an ensemble of simulation outcomes
    generated from the overall distribution.
    """
    # make fancier later? right now assumes constant prior (the horror)
    constant_proba = np.log(0.1) * 10 + np.log(0.01) * 2
    candidates = [np.exp(constant_proba) for tup in my_ensemble_P if tup[1] <= x]
    return np.sum(candidates)

    # setting up simulation. 
N_AGENTS = 100
ALPHA = 2.5
EPSILON_INIT = 0.25
rnd_info = []

ensemble_P = []
ensemble_E = []

"""
Robin notes

sp threshold is the threshold an agent needs to have to share misinfo 
"""


# burn-in period (kind of) - getting ensemble_E, 
# which is the fairly likely set of parameter -> outcome items.
while len(ensemble_E) < 100:
    if len(ensemble_E) % 5 == 0 and len(ensemble_E) != 0:
        print(len(ensemble_E))
        
    params_dict = generate_params_dict()
    agents, shares, centrality, hate_orientation_time, hatefulness_time = run_agent_simulation_now_with_added_hate(N_AGENTS, params_dict)
    tup = (params_dict, p_x_y_hate(agents, hate_orientation_time, centrality, ALPHA))

    proba_p = np.exp(-1.0 * tup[1]/ EPSILON_INIT)
    draw = np.random.uniform()
    if draw < proba_p:
        ensemble_E.append(tup)
        # if it's a good draw compared to the overall "background" ensemble,
        # we will likely add it to ensemble_E. 
        print("goodnews")
    ensemble_P.append(tup)
    #print(len(ensemble_E))

    # smoothed probabilities for items in ensemble_P
G_result = [G_func(ensemble_P, tup[1]) for tup in ensemble_P]
# the starting "good" ensemble of predictions
ensemble_E = [(tup[0], G_func(ensemble_P, tup[1])) for tup in ensemble_E]
U = np.mean(G_result)

EPSILON = EPSILON_INIT
t = 1
swap_rate = 0
tries = 0 
while True:
    # we choose a random item in ensemble_E 
    # and generate a new set of parameters & its simulation outcome.
    # if our new item is better than the item in ensemble E,
    # we swap the old item with the new item w/ high probability.
    chosen_one = random.choice([i for i in range(len(ensemble_E))])   
    particle, u = ensemble_E[chosen_one] # u is the goodness of fit, and particle is a dictionary of simulation settings 
    proposal = generate_params_dict()
    agents, shares, centrality, hate_orientation_time, hatefulness_time = run_agent_simulation_now_with_added_hate(N_AGENTS, params_dict)
    proba_star = p_x_y_hate(agents, hate_orientation_time, centrality, ALPHA)
    u_star = G_func(ensemble_P, proba_star)
    
    proba_swap = min(1.0, np.exp(-1.0 * (u_star - u)) / np.exp(EPSILON))
    tries += 1
    if np.random.uniform() < proba_swap:
        swap_rate += 1
        ensemble_E[chosen_one] = (proposal, u_star)

    EPSILON = t ** (0.1)
    # we keep doing this until our continuous swap rate
    # drops below a threshold (0.05) - in that case, 
    # we're rejecting most items so we should give up.
    if t % 10 == 0:
        print(swap_rate / tries)
    if t % 1000 == 0:
        # dumps items to file for future analysis.
        pickle.dump(ensemble_E, open('ensemble_E_{}.pkl'.format(str(t)), 'wb')) ### keep this!!! 
        #change the place that the pickle dumps to!!!! 
    t += 1
    if (swap_rate / tries) < 0.05 and tries > 20:
        break