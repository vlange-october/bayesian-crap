import numpy as np
import random
from sklearn.linear_model import LinearRegression

from utilities import markov_update_log

MARKOV_STEP = 0.01
# step for MCMC variable change; can update as needed for larger/smaller change increments.


def choose_orientation(i, hate_agents,counter_hate_agents):
    """
    assigns agents to be hateful or counterhateful if they are in the hate agents list or counter hate agetns list 
    """
    orientation = 0 
    if i in hate_agents:
        orientation = 1
    elif i in counter_hate_agents:
        orientation = 2
    return orientation

def generate_params_dict():
    """
    Generates global parameters for a misinfo spread simulation. 
    neighbor trust is a beta distribution governed by {B1_NTRUST, B2_NTRUST}.
    an agent's forcefulness is how much it can convince others & hold onto its beliefs.
    forcefulness starts at one of two values, either FORCEFULNESS_START_LOW or its complement, FORCEFULNESS_START_HIGH..
    the prevalence of values in the population is given by FORCEFULNESS_WT
    SP_START_LOW is share propensity. Like forcefulness, it exists in ratio SP_WT: 1 in the population and starts at either SP_START_LOW or its complement.
    MB_START_LOW is misinfo belief (behaves the same as share propensity & forcefulness)
    Trust stability (how stable one's trust in institutions/misinfo is) is drawn from a beta distribution governed by {B1_START_TS, B2_START_TS}.
    NTRUST_THRESHOLD is how much a person's neighbor has to be trusted by that person before they incorporate that person's beliefs into their own.
    SP_THRESHOLD is the threshold over which a user's share propensity has to be in order to share misinfo.
    
    Returns a dict with all these parameters, chosen randomly from sets of appropriate values.
    """
    B1_NTRUST = np.random.choice([i for i in range(1, 11)])
    B2_NTRUST = np.random.choice([i for i in range(1, 11)])
    B1_START_MB = np.random.choice([i for i in range(1, 11)])
    B2_START_MB = np.random.choice([i for i in range(1, 11)])
    B1_START_FO = np.random.choice([i for i in range(1, 11)])
    B2_START_FO = np.random.choice([i for i in range(1, 11)])
    B1_START_SP = np.random.choice([i for i in range(1, 11)])
    B2_START_SP = np.random.choice([i for i in range(1, 11)])
    B1_START_TS = np.random.choice([i for i in range(1, 11)])
    B2_START_TS = np.random.choice([i for i in range(1, 11)])
    NTRUST_THRESHOLD = np.random.uniform()
    SP_THRESHOLD = np.random.uniform()
    B1_START_Hatefulness = np.random.choice([i for i in range(1, 11)])
    B2_START_Hatefulness = np.random.choice([i for i in range(1, 11)]) 
    B1_START_Hate_Hatefulnes = np.random.choice([i for i in range(10, 11)])
    B2_START_Hate_Hatefulnes = np.random.choice([i for i in range(10, 11)])
    B1_START_Counter_Hatefulnes = np.random.choice([i for i in range(1, 2)])
    B2_START_Counter_Hatefulnes = np.random.choice([i for i in range(1, 2)])


    B1_START_Infectiousness_hate = np.random.choice([i for i in range(1, 11)])
    B2_START_Infectiousness_hate = np.random.choice([i for i in range(1, 11)])
    B1_START_Infectiousness_counter = np.random.choice([i for i in range(1, 11)])
    B2_START_Infectiousness_counter = np.random.choice([i for i in range(1, 11)])
    B1_START_Infectiousness_hate_inhibited = np.random.choice([i for i in range(1, 11)]) 
    B2_START_Infectiousness_hate_inhibited = np.random.choice([i for i in range(1, 11)])
    

    #creates hateful agents and counter hateful agents by drawing from a beta distribution that is skewed toward 0 
    num_hate_agents = round(abs(np.log(np.random.beta(1,10))))
    num_counter_agents = round(abs(np.log(np.random.beta(1,10))))

    total_agents = num_counter_agents + num_hate_agents
    hate_nonhate_agents = random.sample(range(0,100),total_agents) #chooses which agents will be hateful and counter hateful 
    hate_agents = hate_nonhate_agents[0:num_hate_agents] 
    counter_hate_agents = hate_nonhate_agents[num_hate_agents:total_agents]
    #print("hate, counter", num_hate_agents, num_counter_agents)
    return {
        "B1_NTRUST": B1_NTRUST,
        "B2_NTRUST": B2_NTRUST,
        "B1_START_MB": B1_START_MB,
        "B2_START_MB": B2_START_MB,
        "B1_START_FO": B1_START_FO,
        "B2_START_FO": B2_START_FO,
        "B1_START_SP": B1_START_SP,
        "B2_START_SP": B2_START_SP,
        "B1_START_TS": B1_START_TS,
        "B2_START_TS": B2_START_TS,
        "NTRUST_THRESHOLD": NTRUST_THRESHOLD,
        "SP_THRESHOLD": SP_THRESHOLD,
        "B1_START_Hatefulness": B1_START_Hatefulness, 
        "B2_START_Hatefulness": B2_START_Hatefulness,
        "B1_START_Hate_Hatefulnes": B1_START_Hate_Hatefulnes,
        "B2_START_Hate_Hatefulnes": B2_START_Hate_Hatefulnes,
        "B1_START_Counter_Hatefulnes": B1_START_Counter_Hatefulnes,
        "B2_START_Counter_Hatefulnes": B2_START_Counter_Hatefulnes,
        "hate_agents":hate_agents,
        "counter_hate_agents":counter_hate_agents,

        "B1_START_Infectiousness_hate" : B1_START_Infectiousness_hate,
        "B2_START_Infectiousness_hate" : B2_START_Infectiousness_hate,

        "B1_START_Infectiousness_counter" : B1_START_Infectiousness_counter,
        "B2_START_Infectiousness_counter" : B2_START_Infectiousness_counter,

        "B1_START_Infectiousness_hate_inhibited" : B1_START_Infectiousness_hate_inhibited,
        "B2_START_Infectiousness_hate_inhibited" : B2_START_Infectiousness_hate_inhibited,
    }


# these spell out legal values for parameters.
# they are not allowed to exceed values in PARAMS_MAX
# or be less than values in PARAMS_MIN.
# we modify a parameter by PARAMS_STEP if it is chosen to be modified
# in the next simulated annealing step.

#changed so that the hate users have to be between 7 and 10 hatefulness and
#counter hate users have to be between 1 and 4

PARAMS_MAX = {
    "B1_NTRUST": 10,
    "B2_NTRUST": 10,
    "B1_START_FO": 10, # forcefulness
    "B2_START_FO": 10,
    "B1_START_SP": 10, #share propensity 
    "B2_START_SP": 10,
    "B1_START_MB": 10, #misinfo belief
    "B2_START_MB": 10,
    "B1_START_TS": 10, #trust stability 
    "B2_START_TS": 10,
    "NTRUST_THRESHOLD": 1.0,
    "SP_THRESHOLD": 1.0,
    "B1_START_Hatefulness": 10,
    "B2_START_Hatefulness": 10,
    "B1_START_Hate_Hatefulnes": 10, # hate users
    "B2_START_Hate_Hatefulnes": 10,
    "B1_START_Counter_Hatefulnes": 1, # counter hate users
    "B2_START_Counter_Hatefulnes": 1,
    
    "B1_START_Infectiousness_hate" : 10,
    "B2_START_Infectiousness_hate" : 10,

    "B1_START_Infectiousness_counter" : 10,
    "B2_START_Infectiousness_counter" : 10,

    "B1_START_Infectiousness_hate_inhibited" : 10,
    "B2_START_Infectiousness_hate_inhibited" : 10,

}

PARAMS_MIN = {
    "B1_NTRUST": 1,
    "B2_NTRUST": 1,
    "B1_START_FO": 1,
    "B2_START_FO": 1,
    "B1_START_SP": 1,
    "B2_START_SP": 1,
    "B1_START_MB": 1,
    "B2_START_MB": 1,
    "B1_START_TS": 1,
    "B2_START_TS": 1,
    "NTRUST_THRESHOLD": 0.00001,
    "SP_THRESHOLD": 0.000001,
    "B1_START_Hatefulnes": 1,
    "B2_START_Hatefulnes": 1,
    "B1_START_Hate_Hatefulnes": 10, # hate users
    "B2_START_Hate_Hatefulnes": 10,
    "B1_START_Counter_Hatefulnes": 1, # counter hate users
    "B2_START_Counter_Hatefulnes": 1,

    "B1_START_Infectiousness_hate" : 0,
    "B2_START_Infectiousness_hate" : 0,

    "B1_START_Infectiousness_counter" : 0,
    "B2_START_Infectiousness_counter" : 0,

    "B1_START_Infectiousness_hate_inhibited" : 0,
    "B2_START_Infectiousness_hate_inhibited" : 0,

}

PARAMS_STEP = {
    "B1_NTRUST": 1,
    "B2_NTRUST": 1,
    "B1_START_FO": 1,
    "B2_START_FO": 1,
    "B1_START_SP": 1,
    "B2_START_SP": 1,
    "B1_START_MB": 1,
    "B2_START_MB": 1,
    "B1_START_TS": 1,
    "B2_START_TS": 1,
    "NTRUST_THRESHOLD": 0.05,
    "SP_THRESHOLD": 0.05,
    "B1_START_Hatefulnes": 1,
    "B2_START_Hatefulnes": 1,

    "B1_START_Infectiousness_hate" : 1,
    "B2_START_Infectiousness_hate" : 1,

    "B1_START_Infectiousness_counter" : 1,
    "B2_START_Infectiousness_counter" : 1,

    "B1_START_Infectiousness_hate_inhibited" : 1,
    "B2_START_Infectiousness_hate_inhibited" : 1,
}


def step_params_dict(params_dict):
    """
    Generate a new params dict one "step" away from the current one.
    The proposed params dict may or may not be kept.
    We choose a random parameter to alter, then step it the appropriate amount from the current value.
    Based on a coin flip, we decide whether to increase or decrease the parameter.
    Note that we limit params to be in the range of [PARAMS_MIN[k], PARAMS_MAX[k]]; 
    this is mostly to avoid invalid values (probabilities should be between 0 and 1, etc).

    Returns a new parameter dict.
    """
    k = np.random.choice([k for k in params_dict.keys()])
    if k in {
        "NTRUST_THRESHOLD",
        "SP_THRESHOLD",
        "B1_NTRUST",
        "B2_NTRUST",
        "B1_START_FO",
        "B2_START_FO",
        "B1_START_SP",
        "B2_START_SP",
        "B1_START_MB",
        "B2_START_MB",
        "B1_START_TS",
        "B2_START_TS",
        "MB_WT",
        "B1_START_TS",
        "B2_START_TS",
        "B1_START_Hatefulnes",
        "B2_START_Hatefulnes",
    }:
        if np.random.uniform() > 0.5:  # the coinflip to increase or decrease the parameter
            params_dict[k] = min(PARAMS_MAX[k], params_dict[k] + PARAMS_STEP[k])
        else:
            params_dict[k] = max(PARAMS_MIN[k], params_dict[k] - PARAMS_STEP[k])
    elif k[-3:] == "LOW":
        if np.random.uniform() > 0.5:
            params_dict[k] = np.log(
                min(0.9999999, np.exp(params_dict[k]) + PARAMS_STEP[k])
            )
        else:
            params_dict[k] = np.log(
                max(0.0000001, np.exp(params_dict[k]) - PARAMS_STEP[k])
            )
    else:
        if np.random.uniform() > 0.5:
            params_dict[k] = np.log(
                min(0.9999999, np.exp(params_dict[k]) + PARAMS_STEP[k])
            )
        else:
            params_dict[k] = np.log(
                max(0.0000001, np.exp(params_dict[k]) - PARAMS_STEP[k])
            )
    return params_dict


def calc_energy(agents, shares_hate, time_step, tot_time_steps, centrality):
    """
    Calculates "energy" for simulated annealing.
    Takes as params:
        agents: list of agents in the simulation
        shares: the sharing records for all the agents over all time steps in the simulation
        time_step: the time step in the simulated annealing algorithm
        tot_time_steps: the number of time steps we will go through in the SA algorithm
        centrality: a list of each agent's eigenvector centrality in the network. ordered by agent_id.

    Right now it returns a weighted combination of how many misinfo articles were shared 
    (to correct for the condition where 0% out of 0 misinfo snippets are shared --> better pareto score)
    Pareto score refers to what fraction of the misinfo was shared by the top 1% of users. 
    Approximating this to 0.8 for now and taking L2 loss.
    According to one paper we read, more central nodes share more misinformation.
    We replicate this by checking for positive linear correlation between centrality & misinfo sharing.

    returns a float value of "energy"; lower "energy" is better.
    
    """
    how_much_shared_by_top_one_percent_gt = 0.8
    shared = [np.sum([v for v in shares_hate[a.agent_id].values()]) for a in agents]
    shared_by_id = [
        (a.agent_id, np.sum([v for v in shares_hate[a.agent_id].values()])) for a in agents
    ]
    shared_by_id = sorted(shared_by_id, key=lambda b: b[0])
    shared_by_id = [s[1] for s in shared_by_id]
    reg = LinearRegression().fit(centrality, shared)
    if reg.coef_[0] > 0.0:
        more_central_more_misinfo = 1.0
    else:
        more_central_more_misinfo = 1000.0
    how_much_shared_by_top_one_percent_model = np.sum(
        sorted(shared)[-int(0.01 * len(shared)) :]
    ) / (1 + np.sum(shared))
    top_misinfo_share_loss = (
        how_much_shared_by_top_one_percent_model - how_much_shared_by_top_one_percent_gt
    ) ** 2.0
    was_anything_shared = np.sum(shared)
    return (
        5000 * top_misinfo_share_loss - was_anything_shared + more_central_more_misinfo
    )


def acceptance_proba(energy_curr, energy_new, time_step, tot_time_steps):
    """
    probability of accepting the new state given its energy.
    energy_curr is the current energy of the system;
    energy_new is the energy of the proposed state.
    time_step is the number of steps we've taken in the SA algorithm
    tot_time_steps is the total number of time steps the SA algo will take.

    If energy_new is an improvement on energy_curr, we almost always decide to go to the new state.
    Otherwise, the decision to go to a new state is a function of time step & how much of a delta there is between the new state & the current state.

    Returns a probability between 0 and 1.
    """
    temp = 1.0 - ((time_step - 1) / tot_time_steps)
    if energy_new + 0.0001 < energy_curr:
        return 0.95
    else:
        return np.exp((energy_new - energy_curr) / (-1 * temp))


def update_agent_info(d):
    """
    External function (to the agent class, at least) to update an agent's info 
    based on MCMC methods & info from neighbors.
    
    We update our beliefs in misinfo based on info from our neighbors - what they believe, how forceful they are, and how much we trust them - and stochastically update our trust in our neighbors as well.
    We also update our share propensity based on how stable our trust is & MCMC methods.
    Finally, we decide whether this agent will share misinfo at this time step or not
    This decision is based on misinfo belief & share propensity.

    Note that in order to run agent info updating in parallel (likely more useful with a computer with more cores & larger simulation size) we will need to make this a top level function in whatever script is running this (due to quirks of python multiprocessing).

    Returns a dict w/ neighbor trust, share propensity, misinfo belief, & whether or not the agent shared misinfo.
    """
    trust_stability = d["trust_stability"]
    misinfo_belief = d["misinfo_belief"]
    agent_forcefulness = d["agent_forcefulness"]
    share_propensity = d["share_propensity"]
    neighbor_trust = d["neighbor_trust"]
    neighbor_beliefs = d["neighbor_beliefs"]
    neighbor_forcefulness = d["neighbor_forcefulness"]
    NTRUST_THRESHOLD = d["NTRUST_THRESHOLD"]
    SP_THRESHOLD = d["SP_THRESHOLD"]
    #new
    hatefulness = d["hatefulness"]
    neighbor_beliefs_hate = d["neighbor_beliefs_hate"]
    hate_orientation = d["hate_orientation"]

    hate_forcefulness = d["hate_forcefulness"]
    counterhate_forcefulness = d["counterhate_forcefulness"]
    hate_inhibited_forcefulness = d["hate_inhibited_forcefulness"]


    

    
    neighbor_orientation = d["neighbor_orientation"] 

    #updates hate_orientation and hatefulness
    if hate_orientation == 0:
        if 1 in neighbor_orientation: # if has hateful neighbors
            if 2 in neighbor_orientation: # if has nonhateful neighbors 
                if hatefulness == 10:
                    if random.random() < hate_inhibited_forcefulness/10:
                        hate_orientation = 1
                elif hatefulness == 1:
                    if random.random() < counterhate_forcefulness/10:
                        hate_orientation = 2
                else: # reduced effect bc of so many neighbors
                    hatefulness = hatefulness + (neighbor_orientation.count(1) * np.random.uniform(0,0.5)) *0.25
                    hatefulness = hatefulness - (neighbor_orientation.count(2) * np.random.uniform(0,0.5)) *0.25
                
                    
            else: # if does not have nonhateful neighbors
                if hatefulness ==10:
                    if random.random() < hate_forcefulness/10:
                        hate_orientation = 1
                else:
                    hatefulness = hatefulness + (neighbor_orientation.count(1) * np.random.uniform(0,0.5))

        if 2 in neighbor_orientation and 1 not in neighbor_orientation:
            if hatefulness ==1:
                if random.random() < counterhate_forcefulness/10:
                    hate_orientation = 2
            else:
                hatefulness = hatefulness - (neighbor_orientation.count(2) * np.random.uniform(0,0.5))
        hatefulness = hate_max_min(hatefulness)
        

    for n in neighbor_trust.keys():
        neighbor_trust[n] = markov_update_log(
            neighbor_trust[n], trust_stability, MARKOV_STEP, NTRUST_THRESHOLD
        )

    trust_belief = []
    for tup, f in zip(neighbor_beliefs, neighbor_forcefulness):
        n, b = tup
        if b > 0.5: # if neighbor forcefulness is greater than 0.5 then we change our trust belief???
            trust_belief.append(np.exp(neighbor_trust[n] + f))
        else:
            trust_belief.append(-1.0 * np.exp(neighbor_trust[n] + f))
    
    trust_belief_hate = []
    for tup, f in zip(neighbor_beliefs_hate, neighbor_forcefulness):
        n, b = tup
        if b > 0.5: # if neighbor forcefulness is greater than 0.5 then we change our trust belief???
            trust_belief_hate.append(np.exp(neighbor_trust[n] + f))
        else:
            trust_belief_hate.append(-1.0 * np.exp(neighbor_trust[n] + f))


    misinfo_belief = np.log(
        max(
            0.00000000001,
            np.exp(misinfo_belief) + np.sum(trust_belief) - np.exp(agent_forcefulness),
        )
    )

    sigma = 0.007 
    #from dutta et al. 2018

    #chance of being hateful from each exposure is sigma, counter hateful is sigma/2 per neighbor
    #but if neighbors are hateful and counter hateful, chance of going either way is sigma/10

    # if the agent is neutral  has hateful neighbors, the hatefulness updates and so does their orientation
    # if not, then nothing changes 
    # if hate_orientation == 0: # if the user is neutral
    #     if 1 in  neighbor_orientation and 2 not in neighbor_orientation: # if they have hateful neighbors  and no nonhateful neighbors
    #         hatefulness = np.log( # updates hatefulness, will be useful for joint model maybe??
    #             max(
    #                 0.00000000001,
    #                 np.exp(hatefulness) + np.sum(trust_belief_hate) - np.exp(agent_forcefulness),
    #             )
    #         )
    #         for neighbor in neighbor_orientation:
    #             if neighbor == 1:
    #                 if np.random.rand() < sigma:
    #                     hate_orientation = 1
    #     if 1 in  neighbor_orientation and 2 in neighbor_orientation: # if they have hateful neighbors  and nonhateful neighbors
    #         hatefulness = np.log( # updates hatefulness, will be useful for joint model maybe?? 
    #             max(
    #                 0.00000000001,
    #                 np.exp(hatefulness) + np.sum(trust_belief_hate) - np.exp(agent_forcefulness),
    #             )
    #         )
    #         for neighbor in neighbor_orientation:
    #             if neighbor == 1:
    #                 if np.random.rand() < sigma/10:
    #                     hate_orientation = 1
    #             if neighbor == 2:
    #                 if np.random.rand() < sigma/10:
    #                     hate_orientation = 2 
    #     if 1 not in  neighbor_orientation and 2 in neighbor_orientation: # if they have no hateful neighbors  and nonhateful neighbors
    #         hatefulness = np.log( # updates hatefulness, will be useful for joint model maybe??
    #             max(
    #                 0.00000000001,
    #                 np.exp(hatefulness) + np.sum(trust_belief_hate) - np.exp(agent_forcefulness),
    #             )
    #         )
    #         for neighbor in neighbor_orientation:
    #             if neighbor == 2:
    #                 if np.random.rand() < sigma/2:
    #                     hate_orientation = 1

        

    # else:
    #     hatefulness = hatefulness 
    #     hate_orientation = hate_orientation 


    share_propensity = markov_update_log(
        share_propensity, trust_stability, MARKOV_STEP, SP_THRESHOLD
    )
    rand0 = np.log(np.random.uniform())
    if rand0 < (0.75 * share_propensity) + (0.25 * misinfo_belief): #whether the agent shares something or not 
        shares = True
    else:
        shares = False

    #shares hate - going to be removed, doesn't actually affect anything 
    rand0 = np.log(np.random.uniform())
    if rand0 < (0.75 * share_propensity) + (0.25 * hatefulness): #whether the agent shares something or not 
        shares_hate = True
    else:
        shares_hate = False



    return {
        "neighbor_trust": neighbor_trust,
        "misinfo_belief": misinfo_belief,
        "share_propensity": share_propensity,
        "shares": shares,
        #new things to return - hate 
        "hatefulness": hatefulness,
        "shares_hate": shares_hate,
        "hate_orientation": hate_orientation,

        "hate_forcefulness": hate_forcefulness,
        "counterhate_forcefulness":counterhate_forcefulness,
        "hate_inhibited_forcefulness": hate_inhibited_forcefulness,


    }
def getList_ofkeys(dict):
    list = []
    for key in dict.keys():
        list.append(key)
          
    return list

def hate_max_min(hatefulness):
    hatefulness = hatefulness
    if hatefulness>10:
        hatefulness = 10
    if hatefulness < 1:
        hatefulness = 1
    return (hatefulness)

def make_agent_info_dict(agent, neighbor_beliefs, neighbor_forcefulness, neighbor_beliefs_hate, neighbor_orientation, params_dict):
    """
    Given an agent & some info about its neighbors & the current parameter state,
    package up the necessary parameters to calculate the agent's info for the next time step.
    Necessary for parallel computation as objects are ... hard to do stuff to in parallel.

    Returns a dict w/ the relevant info.
    """
    return {
        "trust_stability": agent.trust_stability,
        "misinfo_belief": agent.misinfo_belief,
        "hatefulness": agent.hatefulness,
        "share_propensity": agent.share_propensity,
        "neighbor_trust": agent.neighbors,
        "neighbor_beliefs": neighbor_beliefs,
        "neighbor_beliefs_hate": neighbor_beliefs_hate,
        "neighbor_forcefulness": neighbor_forcefulness,
        "neighbor_orientation":neighbor_orientation,
        "agent_forcefulness": agent.forcefulness,
        "NTRUST_THRESHOLD": params_dict["NTRUST_THRESHOLD"],
        "SP_THRESHOLD": params_dict["SP_THRESHOLD"],
        "hate_orientation": agent.hate_orientation,
        "hate_forcefulness": agent.hate_forcefulness,
        "counterhate_forcefulness": agent.counterhate_forcefulness,
        "hate_inhibited_forcefulness": agent.hate_inhibited_forcefulness,

    }
