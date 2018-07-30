import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn import tree, ensemble

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

boost_tree = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=3)).fit(ratio.T, category)
bag_tree = ensemble.BaggingClassifier(tree.DecisionTreeClassifier(max_depth=3)).fit(ratio.T, category)
 
plt.figure(1)
fig, axarr = plt.subplots(1, 2)
for i in [0,1]:
    decision_tree = tree.DecisionTreeClassifier(max_depth=i+4).fit(ratio.T, category)
    tree_result = decision_tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    axarr[i].pcolormesh(xx, yy, tree_result, cmap=plt.cm.Paired)
    axarr[i].scatter(pass_ratio[category == 0], shot_ratio[category == 0], c='r', marker = 'o')
    axarr[i].scatter(pass_ratio[category == 1], shot_ratio[category == 1], c='b', marker = '^')
    axarr[i].set_title("Decision Tree with Depth = {}".format(i+4))
    axarr[i].set_xlabel('pass ratio')
    axarr[i].set_ylabel('shot ratio')

plt.figure(2)
fig, axarr = plt.subplots(1, 2)
for j, model, subtitle in zip([0,1],[boost_tree, bag_tree],['Decision Tree with Boosting','Decision Tree with Bagging']):
    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    result = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    axarr[j].pcolormesh(xx, yy, result, cmap=plt.cm.Paired)
    axarr[j].scatter(pass_ratio[category == 0], shot_ratio[category == 0], c='r', marker = 'o')
    axarr[j].scatter(pass_ratio[category == 1], shot_ratio[category == 1], c='b', marker = '^')
    axarr[j].set_title(subtitle)
    axarr[j].set_xlabel('pass ratio')
    axarr[j].set_ylabel('shot ratio')

plt.show()
