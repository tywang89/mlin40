import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn import manifold

stats = pd.read_table('/Users/wangtianyi/Documents/python_work/regression.csv')
point = stats.iloc[:,4] / 38
positional_rating = stats.iloc[:,[0,1,2,3]]

iso = manifold.Isomap(n_neighbors=2, n_components=2).fit_transform(positional_rating)
lle = manifold.LocallyLinearEmbedding(n_neighbors=2, n_components=2).fit_transform(positional_rating)
tsne = manifold.TSNE(n_components=2).fit_transform(positional_rating)

fig = plt.figure()  
  
ax1 = fig.add_subplot(131)  
ax1.scatter(iso[:, 0], iso[:, 1], c='r')
plt.title("Isomap result")
plt.xticks([]), plt.yticks([])
plt.axis('tight')
  
ax2 = fig.add_subplot(132)  
ax2.scatter(lle[:, 0], lle[:, 1], c='b')
plt.title("LLE result")
plt.xticks([]), plt.yticks([])
plt.axis('tight')
  
ax3 = fig.add_subplot(133)
ax3.scatter(tsne[:, 0], tsne[:, 1], c='g')
plt.title("tSNE result") 
plt.xticks([]), plt.yticks([])
plt.axis('tight')

plt.show()
