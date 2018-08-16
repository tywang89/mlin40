import numpy as np
import math
from hmmlearn.hmm import MultinomialHMM

model_man_derby = MultinomialHMM(n_components=2)
states = ["Home", "Away"]
observations = ["Win", "Lose", "Draw"]

initial_vector = np.array([0.5, 0.5])
model_man_derby.startprob_ = initial_vector
transition_matrix = np.array([[0.2, 0.8],[0.8, 0.2]])
model_man_derby.transmat_ = transition_matrix
emission_matrix = np.array([[0.4, 0.467, 0.133],[0.4, 0.4, 0.2]])
model_man_derby.emissionprob_ = emission_matrix

result = np.array([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]).T
titles = ["WW", "WL", "WD", "LW", "LL", "LD", "DW", "DL", "DD"]
i = 0

for title in titles:
    logprob = model_man_derby.score(result[:,i].reshape(1, -1))
    print(title, ':', math.exp(logprob))
    i += 1

