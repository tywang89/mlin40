import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn import discriminant_analysis, linear_model

rawstat = pd.read_table('/Users/wangtianyi/Documents/python_work/linear separable.csv')
stat = rawstat.iloc[:,[0,1]]
shots = rawstat.iloc[:,0]
tackles = rawstat.iloc[:,1]
category = rawstat.iloc[:,2]

x_min, x_max = shots.min() - 0.2, shots.max() + 0.2
y_min, y_max = tackles.min() - 0.2, tackles.max() + 0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

lda_model = discriminant_analysis.LinearDiscriminantAnalysis().fit(stat, category)
lda_result = lda_model.predict(np.c_[xx.ravel(), yy.ravel()])
lda_result = lda_result.reshape(xx.shape)

# an infinite C indicates no regularization
logreg_model = linear_model.LogisticRegression(C=1e9).fit(stat, category)
lr_result = logreg_model.predict(np.c_[xx.ravel(), yy.ravel()])
lr_result = lr_result.reshape(xx.shape)
  
fig = plt.figure()

ax1 = fig.add_subplot(121)  
plt.pcolormesh(xx, yy, lda_result, cmap=plt.cm.Paired)
plt.scatter(shots[category == 0], tackles[category == 0], c='r', marker = 'o')
plt.scatter(shots[category == 1], tackles[category == 1], c='b', marker = '^')
plt.title("linear discrmininant analysis result")
plt.xlabel('shots per game')
plt.ylabel('tackles per game')
  
ax2 = fig.add_subplot(122)  
plt.pcolormesh(xx, yy, lr_result, cmap=plt.cm.Paired)
plt.scatter(shots[category == 0], tackles[category == 0], c='r', marker = 'o')
plt.scatter(shots[category == 1], tackles[category == 1], c='b', marker = '^')
plt.title("logistic regression result")
plt.xlabel('shots per game')
plt.ylabel('tackles per game')

plt.show()
