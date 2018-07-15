import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn import cluster

rawstat = pd.read_table('/Users/wangtianyi/Documents/python_work/clustering.csv')
stat = pd.DataFrame(rawstat.iloc[:,0])
stat['pass_ratio'] = rawstat.iloc[:,1] / rawstat.iloc[:,2]
stat['dribble_ratio'] = rawstat.iloc[:,3] / rawstat.iloc[:,4]

seed = stat[rawstat['Team'].isin(['ARS', 'EVE', 'STK'])]
seed = seed.iloc[:,[1,2]]
init = seed.values

cluster_stat = stat.iloc[:,[1,2]].values
kmeans_set = cluster.KMeans(n_clusters=3, init=init).fit(cluster_stat)
kmeans_rand = cluster.KMeans(n_clusters=3).fit(cluster_stat)

x_min, x_max = stat['pass_ratio'].min() - 0.05, stat['pass_ratio'].max() + 0.05
y_min, y_max = stat['dribble_ratio'].min() - 0.05, stat['dribble_ratio'].max() + 0.05
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.001), np.arange(y_min, y_max, 0.001))

fig, axarr = plt.subplots(1, 2, sharex='col', sharey='row')

for idx, kmeans, subtitle in zip([0,1],[kmeans_set, kmeans_rand],['Kmeans','auto Kmeans']):

    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    axarr[idx].pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
    axarr[idx].scatter(stat['pass_ratio'], stat['dribble_ratio'], c='r', marker = 'o')
    for a,b,c in zip(stat['Team'],stat['pass_ratio'], stat['dribble_ratio']):  
        axarr[idx].text(b, c+0.005, a, ha='center', va= 'bottom',fontsize=9)   
    axarr[idx].set_title(subtitle)

plt.show()
