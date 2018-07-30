import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn import linear_model, tree

stats = pd.read_table('/Users/wangtianyi/Documents/python_work/regression.csv')
point = stats.iloc[:,4] / 38
rating = stats.iloc[:,5]
positional_rating = stats.iloc[:,[1,3]].values
rating = rating.values.reshape(-1, 1)

stat = pd.DataFrame(stats.iloc[:,4] / 38)
stat['DF'] = stats.iloc[:,1]
stat['FW'] = stats.iloc[:,3]

# performing a tiny pertubation and rearranging data in the ascending order
x = np.arange(6.5, 7.2, 1e-3)[:, np.newaxis]

linear = linear_model.LinearRegression()
linear_pred = linear.fit(rating, point).predict(x)

# one dimensional tree
regression_tree = tree.DecisionTreeRegressor(max_depth=3)
tree_pred = regression_tree.fit(rating, point).predict(x)

# two dimensional tree
multi_tree = tree.DecisionTreeRegressor(max_depth=3)
multi_pred = multi_tree.fit(positional_rating, point).predict(np.c_[xx.ravel(), yy.ravel()])
multi_result = multi_pred.reshape(xx.shape)

x_min, x_max = positional_rating[:,0].min() - 0.05, positional_rating[:,0].max() + 0.05
y_min, y_max = positional_rating[:,1].min() - 0.05, positional_rating[:,1].max() + 0.05
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.001), np.arange(y_min, y_max, 0.001))

plt.figure(1)
plt.scatter(rating, point, c="r", s=10, label = "data")
plt.plot(x, linear_pred, c="b", linewidth=2, label = "linear regression")
plt.plot(x, tree_pred, c="g", linewidth=2, label = "regression tree")
plt.title("Univariate Regression Tree Result")
plt.xlabel('Average Rating')
plt.ylabel('Average Points')
plt.legend()

plt.figure(2)
plt.pcolormesh(xx, yy, multi_result, cmap=plt.cm.Paired)
plt.scatter(positional_rating[:,0], positional_rating[:,1], c='r', marker = 'o')
for a,b,c in zip(stat['Points'],stat['DF'], stat['FW']):  
    plt.text(b+0.005, c+0.005, "%.2f" % a, ha='center', va= 'bottom',fontsize=9)
plt.title("Bivariate Regression Tree Result")
plt.xlabel('Average Defender Rating')
plt.ylabel('Average Attacker Rating')

plt.show()
