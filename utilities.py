import numpy as np

class bayesianAgent(object):
    neighbors = {}

    def __init__(self):
        pass

    def update_info(self):
        pass

    def spreading_process(self):
        pass


class bayesianUpdater(object):
    def __init__(self):
        pass

    def get_all_probas(self):
        pass

    def update_all_probas(self):
        pass


def markov_update(quantity, threshold, step, proba_decrease):
    rand1 = np.random.uniform()
    if rand1 > threshold:
        rand2 = np.random.uniform()
        if rand2 > proba_decrease:
            return min(1.0, quantity + step)
        else:
            return max(0.0, quantity - step)
    else:
        return quantity

class misinfoAgent(bayesianAgent):
    def __init__(
        self,
        agent_id,
        neighbors,
        forcefulness,
        share_propensity,
        misinfo_belief,
    ):
        super().__init__()
        self.agent_id = agent_id
        self.neighbors = neighbors
        self.forcefulness = forcefulness
        self.share_propensity = share_propensity
        self.misinfo_belief = misinfo_belief
        self.neighbor_trust = neighbor_trust
        self.trust_stability = trust_stability

    def update_info(self, neighbor_beliefs, neighbor_forcefulness):
        for n in self.neighbors:
            self.neighbor_trust[n] = markov_update(self.neighbor_trust[n], self.trust_stability, 0.05, 0.5)
        beliefs = []
        trusts = []
        for n, b, f in zip(neighbor_beliefs, neighbor_forcefulness):
            trusts.append(self.neighbor_trust[n])
            beliefs.append(b * f)

        self.misinfo_belief = self.misinfo_belief + np.log(np.multiply(self.neighbor_trust, neighbor_beliefs))

        self.share_propensity = markov_update(self.share_propensity, self.trust_stability, 0.05, 0.9)
        
        
