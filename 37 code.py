import numpy as np
import pymc3 as pm
import scipy.stats as stats
import matplotlib.pyplot as plt

# n as number of toss and z as number of heads
n = 50
z = 20

# Parameter values for Beta prior and Beta posterior
alpha_prior = 12
beta_prior = 12
alpha_post = 32 # 12 + 20
beta_post = 42 # 12 + 30 

# iterations of the Metropolis algorithm，as many as possible
iter_num = 100000

basic_model = pm.Model()
with basic_model:
    theta = pm.Beta("theta", alpha=alpha_prior, beta=beta_prior)
    y = pm.Binomial("y", n=n, p=theta, observed=z)
    # MAP estimation as initial value for MCMC
    initial = pm.find_MAP() 
    transition = pm.Metropolis()
    trace = pm.sample(iter_num, transition, initial, random_seed=1, progressbar=True)

plt.hist(trace["theta"], bins=50, histtype="step",
         normed=True, label="Posterior (MCMC)", color="red")
x = np.linspace(0, 1, 100)
plt.plot(x, stats.beta.pdf(x, alpha_post, beta_post), 
    label='Posterior (Analytic)', color="blue")

plt.xlabel("$\\theta$, Fairness")
plt.ylabel("Density")
plt.legend()
plt.show()
