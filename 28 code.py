import numpy as np
from sklearn import naive_bayes

data = np.array([[0,0,1,1,1],[1,0,1,1,0],[1,1,0,0,1],[1,1,0,0,0],
                [0,1,0,0,1],[0,0,0,1,0],[1,0,0,1,1],[1,1,0,0,1],
                [1,1,1,1,0],[1,1,0,1,0],[1,1,0,1,1],[1,0,1,1,0],
                [1,0,1,0,0]])
label = np.array([0,0,0,0,0,0,1,1,1,1,1,1,1])
test_data = np.array([1,0,1,1,0])

#No smoothing equivalent to frequentist naive Bayes
nb_model = naive_bayes.BernoulliNB(alpha=0.0)
nb_result = nb_model.fit(data, label).predict_proba(test_data.reshape(1,-1))
print('The probability that the new man is Scottish by naive Bayes is: ', nb_result[:,1])

#Laplacian smoothing equivalent to Bayesian naive Bayes
bnb_model = naive_bayes.BernoulliNB()
bnb_result = bnb_model.fit(data, label).predict_proba(test_data.reshape(1,-1))
print('The probability that the new man is Scottish by Bayesian naive Bayes is: ', bnb_result[:,1])

