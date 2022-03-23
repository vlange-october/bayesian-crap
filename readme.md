# approximate bayesian computation struggles
## simulated annealing approach: 
based off of https://royalsocietypublishing.org/doi/10.1098/rspa.2018.0129

uses simulated annealing to infer parameters for the start of the contagion; 
doesn't do super great but it works! it runs!

seen in misinfo_simulation.py

## approximate bayesian computation approach: 
https://arxiv.org/pdf/1208.2157.pdf

based off the above paper. uses ABC based somewhat on simulated annealing to infer a distribution over parameters for the start of the contagion. found in misinfo_abc.py. 

a more commented copy of the same code is found in bayesian_things.ipynb along with some plots of the parameter distributions
##
Hate speech related parameters are mostly drawn from here: https://dl.acm.org/doi/pdf/10.1145/3487351.3488324 

## utilities etc
misinfo_functions.py contains functions for computing relevant quantities and simulations

utilities.py contains utility functions for log computation etc

agents.py instantiates the agent classes, which are hopefully flexible enough for our purposes.

# hate idea
so basically, everyone has a different propensity for believing hate, but actually emitting hate speech is a line that people cross 
so everyone has a different hate propensity, but only a random number of people believe and emit hate speech 
hate users always emitt hate and counter hate users always emit counter hate

