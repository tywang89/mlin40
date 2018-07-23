import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn import linear_model

rawstat = pd.read_table('/Users/wangtianyi/Documents/python_work/linear separable.csv')
stat = rawstat.iloc[:,[0,1]]
shots = rawstat.iloc[:,0]
tackles = rawstat.iloc[:,1]
category = rawstat.iloc[:,2]

x_min, x_max = shots.min() - 0.2, shots.max() + 0.2
y_min, y_max = tackles.min() - 0.2, tackles.max() + 0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

fig, axarr = plt.subplots(1, 3)
for i in range(3):
    # eta0 is the update rate
    per_model = linear_model.Perceptron(max_iter=2*i+1, eta0=1e-3).fit(stat, category)
    per_result = per_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    axarr[i].pcolormesh(xx, yy, per_result, cmap=plt.cm.Paired)
    axarr[i].scatter(shots[category == 0], tackles[category == 0], c='r', marker = 'o')
    axarr[i].scatter(shots[category == 1], tackles[category == 1], c='b', marker = '^')
    axarr[i].set_title("Perceptron with i = {}".format(2*i+1))
    axarr[i].set_xlabel('shots per game')
    axarr[i].set_ylabel('tackles per game')

plt.show()
