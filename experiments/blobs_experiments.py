import numpy as np
import pandas as pd
from src.ecm_ts import ECM_Clustering
from src.ecm_tools import ecm_plot, _ecm_plot, extractMass
from evclust.ecm import ecm
from sklearn.datasets import make_blobs
import logging
logging.basicConfig(level=logging.INFO)

n_clusters=3
np.random.seed(1)
X, y = make_blobs(n_samples=500, centers=[(0, 0), (3, 2), (3, -1)], cluster_std=0.75, random_state=42)
X = np.expand_dims(X,2)


# Soft-ecm
clus=ECM_Clustering(n_clusters, focal_max_size=None)
clus.nb_outer_it = 20
clus.nb_inner_it = 10
clus.delta_ = 10
clus.lbda_= 1.0
clus.inner_lr = 1e-3
clus.inner_convergence_criteria = 1e-2
clus.outter_convergence_criteria = 1e-4
clus.fit( X )

ecm_plot(clus, X)



# ------------------------ Example 1 -----------------------------
# Data
n_clusters=3
centers = [(0, 0), (80, 80), (160, 160), (-130, 310)]
std_devs = [20, 20, 30, 50]
X, Y = make_blobs(n_samples=[100, 100, 100, 10], centers=centers, cluster_std=std_devs, random_state=0)

# ecm
res = ecm(x=X, c=n_clusters,beta = 2,  alpha=1/6, delta=300)
V = pd.DataFrame(res['g'])
_ecm_plot(res, X, res['crit'], V)


# Soft-ecm
clus=ECM_Clustering(n_clusters, focal_max_size =2)
clus.lbda_ = 3.0
clus.alpha_ = 1/6
clus.beta_ = 2
clus.delta_ = 300
clus.nb_inner_it = 10 
clus.nb_outer_it = 20 
clus.inner_lr = 1e-3
clus.inner_convergence_criteria = 1e-2
clus.outter_convergence_criteria = 1e-4

clus.fit(X)


res = extractMass(clus.M, clus.F, J=clus.J, 
            param={clus.C, clus.alpha_, clus.beta_, 
                    clus.delta_, clus.lbda_})
V = pd.DataFrame(clus.V.squeeze().numpy()).loc[lambda df: ~(df == 0).all(axis=1)]


_ecm_plot(res, X, clus.Jall, V)


ecm_plot(clus, X)



