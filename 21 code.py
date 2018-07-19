import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import statsmodels.api as sm
import patsy

# data acquisition
stats = pd.read_table('/Users/wangtianyi/Documents/python_work/regression.csv')
point = stats.iloc[:,4] / 38
rating = stats.iloc[:,5]

# linear regression with statsmodels
rating_add = sm.add_constant(rating)
est_simple = sm.OLS(point,rating_add).fit()
linear_error = metrics.mean_squared_error(point, est_simple.fittedvalues)

# polynomial fitting with numpy
poly_model = np.polyfit(rating, point, 10)
xx = np.linspace(rating.min(), rating.max(), 100)
poly_value = np.polyval(poly_model, xx)
poly_error = metrics.mean_squared_error(point, np.polyval(poly_model, rating))
 
spline_data = stats.iloc[:,[4,5]]
# a small pertubation is applied to distinguish identical ratings
spline_data['Ratings'] = spline_data['Ratings'].map(lambda x: x + np.random.normal(0, 1e-5))
spline_data['Points'] = spline_data['Points'].map(lambda x: x / 38)
# ascending sorted data for spline regression
spline_data = spline_data.sort_values(by=['Ratings']).values

# generating cubic spline with 3 knots at quantiles
transformed_x = patsy.dmatrix("cr(spline_data[:,1], knots=(6.74,6.88,7.02))")
# fitting generalised linear model on transformed dataset
cubicspline = sm.GLM(spline_data[:,0], transformed_x).fit()
# predicted value for error calculation
pred = cubicspline.predict(patsy.dmatrix("cr(spline_data[:,1], knots=(6.74,6.88,7.02))"))
spline_error = metrics.mean_squared_error(point, pred)
# data for plot
pred_smooth = cubicspline.predict(patsy.dmatrix("cr(xx, knots=(6.74,6.88,7.02))"))

plt.figure(1)
plt.plot(rating, est_simple.fittedvalues, c="r", linewidth=4, label = "RMSE = {}".format(linear_error))
plt.plot(xx, poly_value, c="g", linewidth=4, label = "RMSE = {}".format(poly_error))
plt.scatter(rating, point, c="b", s=4)
plt.ylim(0.5, 3)
plt.xlabel("average rating")
plt.ylabel("average point per game")
plt.title("Simple Linear Regression")
plt.legend()
  
plt.figure(2)
plt.plot(rating, est_simple.fittedvalues, c="r", linewidth=4, label = 'Linear')
plt.plot(xx, pred_smooth, c="g", linewidth=4, label = 'CubicSpline')
plt.scatter(rating, point, c="b", s=4)
plt.ylim(0.5, 3)
plt.xlabel("average rating")
plt.ylabel("average point per game")
plt.title("Spline Regression")
plt.legend()

plt.show()
