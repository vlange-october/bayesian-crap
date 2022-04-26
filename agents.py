class misinfoAgent(object):
    """
    Agent in a network that may or may not spread misinfo.
    Has parameters:
        - agent_id: should be an int. unique identifier for the agent.
        - neighbors: the agent's neighbors in the graph
        - forcefulness: the "forcefulness" with which the agent convinces others & holds onto its beliefs
        - share_propensity: the likelihood that this agent will share misinfo
        - misinfo_belief: the likelihood that this agent believes in misinfo
        - trust_stability: how stable this agent's trust in its neighbors is
        - hatefulness: the likelihood this agent believes in hatespeech
    """
    def __init__(
        self,
        agent_id,
        neighbors,
        forcefulness,
        share_propensity,
        misinfo_belief,
        trust_stability,
        hatefulness,
        hate_orientation,
    ):
        super().__init__()
        self.agent_id = agent_id
        self.neighbors = neighbors
        self.forcefulness = forcefulness
        self.share_propensity = share_propensity
        self.misinfo_belief = misinfo_belief
        self.trust_stability = trust_stability
        self.hatefulness = hatefulness
        self.hate_orientation = hate_orientation
    
    def __str__(self) -> str:
        return f'{self.hate_orientation}'

class hateAgent(object):
    """
    angry angry mad mad agent who hates other people  
    jk
    hateful agent 
    Has parameters:
        - agent_id: should be an int. unique identifier for the agent.
        - neighbors: the agent's neighbors in the graph
        - forcefulness: the "forcefulness" with which the agent convinces others & holds onto its beliefs
        - share_propensity: the likelihood that this agent will share misinfo
        - misinfo_belief: the likelihood that this agent believes in misinfo
        - trust_stability: how stable this agent's trust in its neighbors is
        - hatefulness: the likelihood this agent believes in hatespeech
    """
    def __init__(
        self,
        agent_id,
        neighbors,
        forcefulness,
        share_propensity,
        misinfo_belief,
        trust_stability,
        hatefulness,
        hate_orientation,
        hate_forcefulness,
        counterhate_forcefulness,
        hate_inhibited_forcefulness,
    ):
        super().__init__()
        self.agent_id = agent_id
        self.neighbors = neighbors
        self.forcefulness = forcefulness
        self.share_propensity = share_propensity
        self.misinfo_belief = misinfo_belief
        self.trust_stability = trust_stability
        self.hatefulness = hatefulness
        self.hate_orientation = hate_orientation
        self.hate_forcefulness = hate_forcefulness
        self.counterhate_forcefulness = counterhate_forcefulness
        self.hate_inhibited_forcefulness = hate_inhibited_forcefulness
    
    def __str__(self) -> str:
        return f'{self.hate_orientation}'

