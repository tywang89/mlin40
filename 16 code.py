import pandas as pd
import numpy as np
import statsmodels.api as sm

rawstat = pd.read_table('/Users/wangtianyi/Documents/python_work/poisson regression.csv')
offensive = rawstat.iloc[:,[0,1,2,3,4,5]]
goal = rawstat.iloc[:,6]

offensive_add = sm.add_constant(offensive)
poisson_result = sm.GLM(goal, offensive_add, family=sm.families.Poisson()).fit()
print(poisson_result.summary())

offensive_reduced = offensive.iloc[:,[1,2,5]]
offensive_add = sm.add_constant(offensive_reduced)
poisson_result = sm.GLM(goal, offensive_add, family=sm.families.Poisson()).fit()
print(poisson_result.summary())
