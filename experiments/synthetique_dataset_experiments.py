import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.ecm_ts import ECM_Clustering
from src.ecm_tools import ecm_plot, _ecm_plot, extractMass, get_ensembles
from evclust.ecm import ecm
from sklearn.datasets import make_blobs
import logging
logging.basicConfig(level=logging.INFO)


# Function
def transform_labels(labels, num_singletons):
    def label_to_latex(label):
        if label == 'Cl_atypique':
            return r'$\emptyset$'
        parts = label.split('_')[1:] 
        if len(parts) == num_singletons:  
            return r'$\Omega$'
        latex_parts = [f'\\omega_{part}' for part in parts]
        return r'$' + ', '.join(latex_parts) + '$'
    return [label_to_latex(label) for label in labels]


# ---- Synthetique dataset Exemple 1 --------------
# Raw Data
np.random.seed(1234)
n_clusters=2
X = pd.read_csv('experiments/data/butterfly.csv')

plt.figure(figsize=(8, 4))
plt.scatter(X['V1'], X['V2'], c='black', edgecolor='k', s=25, alpha=0.7)
for i in range(len(X)):
    plt.text(X['V1'][i], X['V2'][i], str(i+1), fontsize=10)
#plt.title('a. Original data', fontsize=12, y=-0.2)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()


# ecm
clus1 = ecm(x=X, c=n_clusters, beta = 2,  alpha=1/6, delta=11)
M = pd.DataFrame(clus1['mass'], columns=get_ensembles(clus1['F']))
M.index = M.index + 1

plt.figure(figsize=(8, 4))
plt.plot(M.index, M['Cl_atypique'], marker='^', label=r'$m(\emptyset)$', color='blue')
plt.plot(M.index, M['Cl_1'], marker='o', label=r'$m(\omega_1)$', color='green')
plt.plot(M.index, M['Cl_2'], marker='s', label=r'$m(\omega_2)$', color='red')
plt.plot(M.index, M['Cl_1_2'], marker='d', label=r'$m(\Omega)$', color='purple')
plt.ylabel('Belief mass')
#plt.title('b. Mass of belief of each cluster by ECM', fontsize=12, y=-0.22)
plt.legend()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()




# Soft-ecm (lambda=1.5)
clus2=ECM_Clustering(n_clusters, focal_max_size=None)
clus2.lbda_= 1.5
clus2.nb_outer_it = 20
clus2.nb_inner_it = 10
clus2.delta_ = 11
clus2.inner_lr = 1e-1
clus2.inner_convergence_criteria = 1e-1
clus2.outter_convergence_criteria = 1e-3
clus2.alpha_ = 1/6
clus2.fit( X )

M = pd.DataFrame(clus2.M, columns=get_ensembles(clus2.F))
M.index = M.index + 1

plt.figure(figsize=(8, 4))
plt.plot(M.index, M['Cl_atypique'], marker='^', label=r'$m(\emptyset)$', color='blue')
plt.plot(M.index, M['Cl_1'], marker='o', label=r'$m(\omega_1)$', color='green')
plt.plot(M.index, M['Cl_2'], marker='s', label=r'$m(\omega_2)$', color='red')
plt.plot(M.index, M['Cl_1_2'], marker='d', label=r'$m(\Omega)$', color='purple')
plt.ylabel('Belief mass')
#plt.title('c. Mass of belief of each cluster by Soft-ECM', fontsize=12, y=-0.22)
plt.legend()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()




# Soft-ecm (lambda=3.5)
clus3=ECM_Clustering(n_clusters, focal_max_size=None)
clus3.lbda_= 3.5
clus3.nb_outer_it = 20
clus3.nb_inner_it = 10
clus3.delta_ = 11
clus3.inner_lr = 1e-1
clus3.inner_convergence_criteria = 1e-1
clus3.outter_convergence_criteria = 1e-3
clus3.alpha_ = 1/6

clus3.fit( X )

M = pd.DataFrame(clus3.M, columns=get_ensembles(clus3.F))
M.index = M.index + 1

plt.figure(figsize=(8, 4))
plt.plot(M.index, M['Cl_atypique'], marker='^', label=r'$m(\emptyset)$', color='blue')
plt.plot(M.index, M['Cl_1'], marker='o', label=r'$m(\omega_1)$', color='green')
plt.plot(M.index, M['Cl_2'], marker='s', label=r'$m(\omega_2)$', color='red')
plt.plot(M.index, M['Cl_1_2'], marker='d', label=r'$m(\Omega)$', color='purple')
plt.ylabel('Belief mass')
#plt.title('c. Mass of belief of each cluster by Soft-ECM', fontsize=12, y=-0.22)
plt.legend()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()








# ---- Synthetique dataset Exemple 2 ---------
# Data
np.random.seed(5678)
n_clusters=3
centers = [(0, 0), (80, 80), (160, 160), (-130, 310)]
std_devs = [20, 20, 30, 50]
X, Y = make_blobs(n_samples=[100, 100, 100, 10], centers=centers, cluster_std=std_devs, random_state=0)

X_ = pd.DataFrame(X, columns=['V1', 'V2'])
plt.figure(figsize=(10, 6))
plt.scatter(X_['V1'], X_['V2'], c=Y, cmap='Dark2')
plt.title('Original data')
plt.show()


# ecm
clus = ecm(x=X, c=n_clusters,beta = 2,  alpha=1/6, delta=300)
V = pd.DataFrame(clus['g'], columns=['V1', 'V2'])
M = pd.DataFrame(clus['mass'])
cluster = pd.Categorical(M.apply(lambda row: row.idxmax(), axis=1))
latex_labels = transform_labels(get_ensembles(clus['F']), 3)

plt.figure(figsize=(8, 4))
scatter=plt.scatter(X_['V1'], X_['V2'], c=cluster, cmap='Dark2')
plt.scatter(V['V1'], V['V2'], color='black', marker='s', s=25)
plt.title('Clustering result by ECM')
plt.legend()
handles, _ = scatter.legend_elements()
plt.legend(handles, latex_labels, title="Clusters", fontsize='small')
plt.show()



# Soft-ecm (lambda=1.5)
clus=ECM_Clustering(n_clusters, focal_max_size =2)
clus.lbda_ = 1.5
clus.alpha_ = 1/6
clus.beta_ = 2
clus.delta_ = 300
clus.nb_inner_it = 10 
clus.nb_outer_it = 20 
clus.inner_lr = 1e-1
clus.inner_convergence_criteria = 1e-1
clus.outter_convergence_criteria = 1e-3
clus.fit(X)

V = pd.DataFrame(clus.V.squeeze().numpy(), columns=['V1', 'V2']).loc[lambda df: ~(df == 0).all(axis=1)]
M = pd.DataFrame(clus.M)
cluster = pd.Categorical(M.apply(lambda row: row.idxmax(), axis=1))
latex_labels = transform_labels(get_ensembles(clus.F), 3)

plt.figure(figsize=(8, 4))
scatter=plt.scatter(X_['V1'], X_['V2'], c=cluster, cmap='Dark2')
plt.scatter(V['V1'], V['V2'], color='black', marker='s', s=25)
plt.title('Clustering result by Soft-ECM')
plt.legend()
handles, _ = scatter.legend_elements()
plt.legend(handles, latex_labels, title="Clusters", fontsize='small')
plt.show()


