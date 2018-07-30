import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn import ensemble

ratio = []

rawstat = pd.read_table('/Users/wangtianyi/Documents/python_work/linear inseparable.csv')
category = rawstat.iloc[:,0]
pass_ratio = rawstat.iloc[:,1] / rawstat.iloc[:,2]
shot_ratio = rawstat.iloc[:,3] / rawstat.iloc[:,4]

ratio.append(pass_ratio)
ratio.append(shot_ratio)
ratio = np.array(ratio)
ratio = ratio.astype('float')

x_min, x_max = ratio[0].min() - 0.05, ratio[0].max() + 0.05
y_min, y_max = ratio[1].min() - 0.05, ratio[1].max() + 0.05
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
xy = np.vstack([xx.ravel(), yy.ravel()]).T

gbdt = ensemble.GradientBoostingClassifier(learning_rate=1e-2,max_depth=3).fit(ratio.T, category)
rf = ensemble.RandomForestClassifier(max_depth=3,max_features=1).fit(ratio.T, category)

fig, axarr = plt.subplots(1, 2)
for j, model, subtitle in zip([0,1],[gbdt,rf],['GBDT result','RF result']):
    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    result = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    axarr[j].pcolormesh(xx, yy, result, cmap=plt.cm.Paired)
    axarr[j].scatter(pass_ratio[category == 0], shot_ratio[category == 0], c='r', marker = 'o')
    axarr[j].scatter(pass_ratio[category == 1], shot_ratio[category == 1], c='b', marker = '^')
    axarr[j].set_title(subtitle)
    axarr[j].set_xlabel('pass ratio')
    axarr[j].set_ylabel('shot ratio')

plt.show()
