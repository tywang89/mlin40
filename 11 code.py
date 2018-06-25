import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm

os.getcwd()
stats = pd.read_table('/Users/wangtianyi/Documents/python_work/regression.csv')
point = stats.iloc[:,4] / 38
rating = stats.iloc[:,5]
positional_rating = stats.iloc[:,[0,1,2,3]]

rating_add = sm.add_constant(rating)
est_simple = sm.OLS(point,rating_add).fit()
print(est_simple.summary())

positional_rating_add = sm.add_constant(positional_rating)
est_multi = sm.OLS(point,positional_rating_add).fit()
print(est_multi.summary())

fig = plt.figure()

ax1 = fig.add_subplot(121)  
plt.plot(rating, est_simple.fittedvalues, c="r", linewidth=4)
plt.scatter(rating, point, c="b", s=5)
plt.xlabel("average rating")
plt.ylabel("average point per game")
plt.title("Simple Linear Regression")
  
ax2 = fig.add_subplot(122)  
plt.scatter(range(len(point)), est_multi.model.endog, c="b", s=5)
plt.scatter(range(len(point)), est_multi.fittedvalues, c="r", s=5)
plt.title("Multivariate Linear Regression")

plt.show()
