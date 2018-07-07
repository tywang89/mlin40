import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn import svm

ratio = []

rawstat = pd.read_table('/Users/wangtianyi/Documents/python_work/linear inseparable.csv')
category = rawstat.iloc[:,0]
pass_ratio = rawstat.iloc[:,1] / rawstat.iloc[:,2]
shot_ratio = rawstat.iloc[:,3] / rawstat.iloc[:,4]

ratio.append(pass_ratio)
ratio.append(shot_ratio)
ratio = np.array(ratio)
ratio = ratio.astype('float')

linear_svm = svm.SVC(kernel='linear', C=1e10).fit(ratio.T, category)

x_min, x_max = ratio[0].min() - 0.05, ratio[0].max() + 0.05
y_min, y_max = ratio[1].min() - 0.05, ratio[1].max() + 0.05
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
xy = np.vstack([xx.ravel(), yy.ravel()]).T
Z = linear_svm.decision_function(xy).reshape(xx.shape)
 
plt.scatter(pass_ratio[category == 0], shot_ratio[category == 0], c='r', marker = 'o')
plt.scatter(pass_ratio[category == 1], shot_ratio[category == 1], c='b', marker = '^')
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
plt.title("playing style discrimination with SVM")
plt.xlabel('pass ratio')
plt.ylabel('shot ratio')
plt.show()
