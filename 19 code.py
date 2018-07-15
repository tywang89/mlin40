import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn import neighbors

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
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.001), np.arange(y_min, y_max, 0.001))

fig, axarr = plt.subplots(1, 3, sharex='col', sharey='row')

for idx, hyper_k, subtitle in zip([0,1,2],[1,7,15],['kNN (k = 1)','kNN (k = 7)','kNN (k = 15)']):

    knn = neighbors.KNeighborsClassifier(n_neighbors=hyper_k).fit(ratio.T, category)
    knn_result = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    knn_result = knn_result.reshape(xx.shape)
 
    axarr[idx].pcolormesh(xx, yy, knn_result, cmap=plt.cm.Paired)
    axarr[idx].scatter(pass_ratio[category == 0], shot_ratio[category == 0], c='r', marker = 'o')
    axarr[idx].scatter(pass_ratio[category == 1], shot_ratio[category == 1], c='b', marker = '^')
    axarr[idx].set_title(subtitle)
    
plt.show()
