import pandas as pd
import numpy as np
from scipy import linalg
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import mixture

rawstat = pd.read_table('/Users/wangtianyi/Documents/python_work/clustering.csv')
stat = pd.DataFrame(rawstat.iloc[:,0])
stat['pass_ratio'] = rawstat.iloc[:,1] / rawstat.iloc[:,2]
stat['dribble_ratio'] = rawstat.iloc[:,3] / rawstat.iloc[:,4]

mixture_stat = stat.iloc[:,[1,2]].values
gmm_model = mixture.GaussianMixture(n_components=3).fit(mixture_stat)
gmm_result = gmm_model.predict(mixture_stat)

fig = plt.figure()
fig = fig.gca()
for i, (mean, covar, color) in enumerate(zip(
            gmm_model.means_, gmm_model.covariances_, (['r','g','b']))):

        plt.scatter(mixture_stat[gmm_result == i, 0], mixture_stat[gmm_result == i, 1], 2, color=color)

        # Plot an ellipse for every Gaussian component
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(fig.bbox)
        ell.set_alpha(0.5)
        fig.add_artist(ell)
        for a,b,c in zip(stat['Team'],stat['pass_ratio'], stat['dribble_ratio']):  
            fig.text(b, c+0.005, a, ha='center', va= 'bottom',fontsize=9) 
plt.title("Gaussian Mixture Result")
plt.show()
