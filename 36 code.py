import pymc3 as pm
import theano
floatX = theano.config.floatX
import theano.tensor as T
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

stats = pd.read_table('/Users/wangtianyi/Documents/python_work/regression.csv')
point = stats.iloc[:,4] / 38
point = point.astype(floatX)
point = point.reshape(point.shape[0], -1)
rating = stats.iloc[:,5]
rating = rating.astype(floatX)
rating = rating.reshape(point.shape[0], -1)

with pm.Model() as linear:
    #normal prior of the parameters
    weights = pm.Normal('weights', mu=0, sd=1, shape=(rating.shape[1], point.shape[1]))
    bias = pm.Normal('bias', mu=0, sd=1, shape=(point.shape[1]))   
    output = pm.Normal('Y', T.dot(rating, weights) + bias, observed=point)

with linear:
    # Automatic Differentiation Variational Inference
    # loss as the evidence lower bound
    inference = pm.ADVI()
    #number of the samples should be as great as possible
    approx = pm.fit(n=1000, method=inference)
    trace = approx.sample(draws=5000)

pm.traceplot(trace)
plt.show()
