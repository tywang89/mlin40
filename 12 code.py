import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn import linear_model,metrics

stats = pd.read_table('/Users/wangtianyi/Documents/python_work/regression.csv')
point = stats.iloc[:,4] / 38
positional_rating = stats.iloc[:,[0,1,2,3]]

linear = linear_model.LinearRegression()
linear.fit(positional_rating, point)
print("The coefficients of linear regression is: \n", linear.coef_, linear.intercept_)
linear_pred = linear.predict(positional_rating)
linear_error = metrics.mean_squared_error(point, linear_pred)

lasso = linear_model.Lasso(alpha = 0.05)
lasso.fit(positional_rating, point)
print("The coefficients of LASSO is: \n", lasso.coef_, lasso.intercept_)
lasso_pred = lasso.predict(positional_rating)
lasso_error = metrics.mean_squared_error(point, lasso_pred)

ridge = linear_model.Ridge(alpha = 0.05)
ridge.fit(positional_rating, point)
print("The coefficients of ridge regression is: \n", ridge.coef_, ridge.intercept_)
ridge_pred = ridge.predict(positional_rating)
ridge_error = metrics.mean_squared_error(point, ridge_pred)

plt.scatter(range(len(point)), linear_pred, c="b", s=5, label = "RMSE = {}".format(linear_error))
plt.scatter(range(len(point)), lasso_pred, c="g", s=5, label = "RMSE = {}".format(lasso_error))
plt.scatter(range(len(point)), ridge_pred, c="r", s=5, label = "RMSE = {}".format(ridge_error))
plt.legend()
plt.title("Multivariate Linear Regression with Regularization")
plt.show()
