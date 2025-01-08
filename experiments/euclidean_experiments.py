import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from src.ecm_ts import ECM_Clustering
from src.ecm_tools import ecm_plot, extractMass, _ecm_plot, ani_ecm_plot
from evclust.ecm import ecm

import logging
logging.basicConfig(level=logging.INFO)



# ------------------------ Simulation Example 1-----------------------------
# Data
n_clusters=2
np.random.seed(0)
X = np.vstack((np.random.multivariate_normal([5, 5], [[3, 1], [1, 1]], 80), 
               np.random.multivariate_normal([12, 12], [[1, 0], [0, 1]], 20)))
X_ = pd.DataFrame(X)
X = np.expand_dims(X, 2)

# ecm
res = ecm(x=X_, c=n_clusters, beta = 2,  alpha=1, delta=10)
V = pd.DataFrame(res['g'])
_ecm_plot(res, X_, res['crit'], V)

# Soft-ecm
clus=ECM_Clustering(n_clusters)
clus.lbda_= 3
clus.alpha_ = 1; clus.beta_ = 2; clus.delta_ = 10
clus.fit(X)
res = extractMass(clus.M, clus.F, J=clus.J, 
            param={clus.C, clus.alpha_, clus.beta_, 
                    clus.delta_, clus.lbda_})
V = pd.DataFrame(clus.V.squeeze().numpy()).loc[lambda df: ~(df == 0).all(axis=1)]
_ecm_plot(res, X_, clus.Jall, V)




# ------------------------ Simulation Example 2-----------------------------
# Data
from sklearn.datasets import make_blobs
n_clusters=3
np.random.seed(1)
X, y = make_blobs(n_samples=500, centers=[(0, 0), (1.5, 1), (1.5, 0)], cluster_std=0.5, random_state=42)
X = pd.DataFrame(X)

# ecm
res = ecm(x=X, c=n_clusters,beta = 2,  alpha=1, delta=10)
V = pd.DataFrame(res['g'])
ecm_plot(res, X, res['crit'], V)

# Soft-ecm
clus=ECM_Clustering(n_clusters)
clus.fit(X)
res = extractMass(clus.M, clus.F, J=clus.J, 
            param={clus.C, clus.alpha_, clus.beta_, 
                    clus.delta_, clus.lbda_})
V = pd.DataFrame(clus.V.squeeze().numpy()).loc[lambda df: ~(df == 0).all(axis=1)]
ecm_plot(res, X, clus.Jall, V)



# ------------------------ Simulation Example 3-----------------------------
# Data
from sklearn.datasets import make_blobs
n_clusters=3
np.random.seed(2)
X, y = make_blobs(n_samples=500, centers=[(0, 0), (2, 2), (4, 0)], cluster_std=0.5, random_state=42)
X = pd.DataFrame(X)

# ecm
res = ecm(x=X, c=n_clusters,beta = 2,  alpha=1, delta=10)
V = pd.DataFrame(res['g'])
ecm_plot(res, X, res['crit'], V)

# Soft-ecm
clus=ECM_Clustering(n_clusters)
clus.fit(X)
res = extractMass(clus.M, clus.F, J=clus.J, 
            param={clus.C, clus.alpha_, clus.beta_, 
                    clus.delta_, clus.lbda_})
V = pd.DataFrame(clus.V.squeeze().numpy()).loc[lambda df: ~(df == 0).all(axis=1)]
ecm_plot(res, X, clus.Jall, V)



# ------------------------ Real dataset Example 4-----------------------------
# Data
n_clusters=3
np.random.seed(4)
def generate_cluster(center, angle, width, height, num_points):
    t = np.linspace(0, 2 * np.pi, num_points)
    x = center[0] + width * np.cos(t)
    y = center[1] + height * np.sin(t)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], 
                                [np.sin(angle), np.cos(angle)]])
    data = np.column_stack((x, y)) @ rotation_matrix
    return data
clusters = [
    {"center": (-7, 5), "angle": 0.3, "width": 5, "height": 10},
    {"center": (0, 5), "angle": -0.3, "width": 5, "height": 10},
    {"center": (7, 5), "angle": 0.3, "width": 5, "height": 10},
]
X = np.vstack([generate_cluster(**cluster, num_points=100) for cluster in clusters])
X = pd.DataFrame(X)

# ecm
res = ecm(x=X, c=n_clusters,beta = 2,  alpha=1, delta=10)
V = pd.DataFrame(res['g'])
ecm_plot(res, X, res['crit'], V)

# Soft-ecm
clus=ECM_Clustering(n_clusters)
clus.fit(X)
res = extractMass(clus.M, clus.F, J=clus.J, 
            param={clus.C, clus.alpha_, clus.beta_, 
                    clus.delta_, clus.lbda_})
V = pd.DataFrame(clus.V.squeeze().numpy()).loc[lambda df: ~(df == 0).all(axis=1)]
ecm_plot(res, X, clus.Jall, V)




# ------------------------ Real dataset Example 5-----------------------------
# Data
n_clusters=2
X = pd.read_csv('../data/butterfly.csv')

# ecm
res = ecm(x=X, c=n_clusters,beta = 2,  alpha=1, delta=10)
V = pd.DataFrame(res['g'])
_ecm_plot(res, X, res['crit'], V)

# Soft-ecm
clus=ECM_Clustering(n_clusters, focal_max_size =2)
clus.lbda_ = 1.2
clus.fit(X)
res = extractMass(clus.M, clus.F, J=clus.J, 
            param={clus.C, clus.alpha_, clus.beta_, 
                    clus.delta_, clus.lbda_})
V = pd.DataFrame(clus.V.squeeze().numpy()).loc[lambda df: ~(df == 0).all(axis=1)]
_ecm_plot(res, X, clus.Jall, V)



# ------------------------ Real dataset Example 6-----------------------------
# Data
n_clusters=4
X_ = pd.read_csv('../data/fourclass.csv')
X = X_.iloc[:, :2]

# ecm
res = ecm(x=X, c=n_clusters,beta = 2,  alpha=1, delta=10)
V = pd.DataFrame(res['g'])
ecm_plot(res, X, res['crit'], V)

# Soft-ecm
clus=ECM_Clustering(n_clusters)
clus.fit(X)
res = extractMass(clus.M, clus.F, J=clus.J, 
            param={clus.C, clus.alpha_, clus.beta_, 
                    clus.delta_, clus.lbda_})
V = pd.DataFrame(clus.V.squeeze().numpy()).loc[lambda df: ~(df == 0).all(axis=1)]
ecm_plot(res, X, clus.Jall, V)









# ------------------------ Real dataset Example 7-----------------------------
# Data
from sklearn.datasets import make_blobs
n_clusters=3
np.random.seed(3)
X, y = make_blobs(n_samples=2000, centers=[(-4, 0), (0, 2), (4, 0)], cluster_std=1.3)
X = pd.DataFrame(X)

# ecm
res = ecm(x=X, c=n_clusters,beta = 2,  alpha=1, delta=10)
V = pd.DataFrame(res['g'])
ecm_plot(res, X, res['crit'], V)

# Soft-ecm
clus=ECM_Clustering(n_clusters)
clus.fit(X)
res = extractMass(clus.M, clus.F, J=clus.J, 
            param={clus.C, clus.alpha_, clus.beta_, 
                    clus.delta_, clus.lbda_})
V = pd.DataFrame(clus.V.squeeze().numpy()).loc[lambda df: ~(df == 0).all(axis=1)]
ecm_plot(res, X, clus.Jall, V)





# ------------------------ iteraction on lambda -----------------------------
import matplotlib.animation as animation
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

n_clusters=3
np.random.seed(1)
X, y = make_blobs(n_samples=500, centers=[(0, 0), (3, 2), (3, -1)], cluster_std=0.75, random_state=42)
X = np.expand_dims(X,2)

# ------
fig, ax = plt.subplots()
def animate(i):
    ax.clear()
    clus=ECM_Clustering(n_clusters, focal_max_size=None)
    clus.lbda_= i+1
    clus.fit(X)
    ani_ecm_plot(clus, X, ax)

ani = animation.FuncAnimation(fig, animate, frames=5, interval=5000, repeat=False)
#plt.show()
ani.save('experiments/outputs/focalmax_5lbda_.gif', writer='pillow')  
ani.save('experiments/outputs/focalmax_5lbda_.mp4', writer='ffmpeg')

# ------
fig, ax = plt.subplots()
def animate(i):
    ax.clear()
    clus=ECM_Clustering(n_clusters, focal_max_size=2)
    clus.lbda_= i+1
    clus.fit(X)
    ani_ecm_plot(clus, X, ax)

ani = animation.FuncAnimation(fig, animate, frames=3, interval=5000, repeat=False)
#plt.show()
ani.save('experiments/outputs/focal2_5lbda_.gif', writer='pillow') 
ani.save('experiments/outputs/focal2_5lbda_.mp4', writer='ffmpeg')


# ------
fig, ax = plt.subplots()
def animate(i):
    ax.clear()
    clus=ECM_Clustering(n_clusters, focal_max_size=2)
    clus.lbda_= i+1
    clus.fit(X)
    ani_ecm_plot(clus, X, ax)

ani = animation.FuncAnimation(fig, animate, frames=20, interval=5000, repeat=False)
#plt.show()
ani.save('experiments/outputs/focal2_20lbda_.gif', writer='pillow') 
ani.save('experiments/outputs/focal2_20lbda_.mp4', writer='ffmpeg')





