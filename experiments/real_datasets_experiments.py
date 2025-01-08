# Basic packages
import numpy as np
import pandas as pd
import time
import torch
import logging
logging.basicConfig(level=logging.INFO)

# Soft-ECM
from src.ecm_ts import ECM_Clustering, euclidean
from src.ecm_tools import  extractMass, nonspecificity, Dmetric, TSeuclidean

# ECM package
from evclust.ecmdd import ecmdd
from evclust.ecm import ecm
from evclust.catecm import catecm

# Clustering package
from tslearn.clustering import TimeSeriesKMeans
from kmodes.kmodes import KModes
from sklearn.cluster import KMeans

# Metrics package
from tslearn.metrics import soft_dtw, SoftDTWLossPyTorch
from sklearn.metrics import accuracy_score, adjusted_rand_score, rand_score, silhouette_score, confusion_matrix
from sklearn.metrics import pairwise_distances

from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import pdist, squareform

# Data package
from aeon.datasets import load_classification


def true_lab(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    cm_argmax = cm.argmax(axis=0)
    y_pred_ = np.array([cm_argmax[i] for i in y_pred])
    return y_pred_

def Hamming(X):
    """Hamming
    """
    D = squareform(pdist(X, metric='hamming'))
    return D

def LossHamming(x, y, eps=1e-6):
    """Loss Hamming
    """
    x_sigmoid = torch.sigmoid(x)
    y_sigmoid = torch.sigmoid(y)
    diff = torch.abs(x_sigmoid - y_sigmoid) + eps
    return torch.sum(diff, dim=(1, 2))


# ------------------------ Data  ------------------------------------------------
# abalone
abalone = pd.read_csv('experiments/data/abalone.data', header=None) 
Xabalone = abalone.iloc[:, 1:9] 
Yabalone = pd.factorize(abalone.iloc[:, 0])[0]
Cablone = len(np.unique(Yabalone))

# ecoli
ecoli = pd.read_csv("experiments/data/ecoli.data", header=None, delim_whitespace=True)
Xecoli = ecoli.iloc[:, 1:8]
Yecoli = pd.factorize(ecoli.iloc[:, 8])[0]
Cecoli = len(np.unique(Yecoli))

# Glass
glass = pd.read_csv("experiments/data/glass.data", header=None)
Xglass = glass.iloc[:, 1:10]
Yglass = pd.factorize(glass.iloc[:, 10])[0]
Cglass = len(np.unique(Yglass))




#===========================> Categorical
# Breast Cancer (BC)
bc = pd.read_csv('experiments/data/breast-cancer.data', header=None) 
Xbc = bc.iloc[:, 1:10] 
Ybc = pd.factorize(bc.iloc[:, 0])[0]
Cbc = len(np.unique(Ybc))

# Soybean (Soybean)
so = pd.read_csv('experiments/data/soybean-small.data', delimiter=",", dtype="O", header=None) 
Xso = np.delete(so,  so.shape[1] - 1, axis=1)
Yso = pd.factorize(so.iloc[:, so.shape[1]-1])[0]
Cso = len(np.unique(Yso))

# Lung (Lung)
lu = pd.read_csv('experiments/data/lung-cancer.data', delimiter=",", dtype="O", header=None) 
Xlu = np.delete(lu,  0, axis=1)
Ylu = pd.factorize(lu.iloc[:, 0])[0]
Clu = len(np.unique(Ylu))



#===========================> TS
# BasicMotions (BM)
Xbm, Ybm = load_classification('BasicMotions', 'train', return_metadata=False)
Ybm = pd.factorize(Ybm)[0]
Cbm = len(np.unique(Ybm))

# ERing (ERing)
Xer, Yer = load_classification('ERing', 'train', return_metadata=False)
Yer= pd.factorize(Yer)[0]
Cer = len(np.unique(Yer))

# FingerMovements (FM)
Xfm, Yfm = load_classification('FingerMovements', 'train', return_metadata=False)
Yfm= pd.factorize(Yfm)[0]
Cfm = len(np.unique(Yfm))


# AtrialFibrillation (AF) 
Xaf, Yaf = load_classification('AtrialFibrillation', 'train', return_metadata=False)
Yaf= pd.factorize(Yaf)[0]
Caf = len(np.unique(Yaf))



# ------------------------ Hyperparametres  ------------------------------------------------
alpha_ = [2.0]
beta_ = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
delta_ = 10.0
lbda_ = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

maxiter = 11

results = pd.DataFrame(columns=['iter','Algorithmes','Dataset', 'Type', 'Metric',
                        'Alpha', 'Beta', 'Lambda', 'ACC', 'ARI', 'RI', 'NS', 'ASW', 'J', 'Time'])
Recm = Recmdd = Rcatecm = Rkmeans = Rkmodes = Rtskmeans = Rsoftecm = results





# ------------------------ ECM Clustering  ------------------------------------------------
datasets = {
    'Abalone': ('Abalone', Xabalone, Yabalone, Cablone, 'Euclidean', 'Numerical'),
    'Ecoli': ('Ecoli', Xecoli, Yecoli, Cecoli, 'Euclidean', 'Numerical'),
    'Glass': ('Glass', Xglass, Yglass, Cglass, 'Euclidean', 'Numerical')
}

clus_ = []
for key, (dataset_name, X, Y, C, Metric, Type) in datasets.items():

    for iter in range(1, maxiter):
        print(f"ECM with {dataset_name} for inter {iter} \n")
        for alpha in alpha_:
            for beta in beta_:
                try:
                    start_time = time.time()
                    clus = ecm(x=X, c=C, beta=beta, alpha=alpha, delta=delta_, type='pairs')
                    end_time = time.time()

                    # Mtrics
                    betp = pd.DataFrame(clus['betp'])
                    cluster = pd.factorize(betp.apply(lambda row: row.idxmax(), axis=1))[0]
                    cluster = true_lab(Y,cluster)
                    acc = round(accuracy_score(Y, cluster), 3)
                    ari = round(adjusted_rand_score(Y, cluster), 3)
                    ri = round(rand_score(Y, cluster), 3)
                    ns = round(nonspecificity(clus['mass'], clus['F']), 3)
                    J = round(clus['crit'], 3)
                    time_ = round(end_time - start_time, 3)
                    try:
                        asw = round(silhouette_score(X, cluster), 3)
                    except Exception:
                        asw = 'None'
                except Exception as e:
                    acc = 'None'
                    ari = 'None'
                    ri = 'None'
                    ns = 'None'
                    asw = 'None'
                    J = 'None'
                    time_ = 'None'
                    print(f"Erreur for data={dataset_name}, alpha={alpha}, beta={beta} : {e}")
                
                # DataFrame
                new_row = pd.DataFrame([{
                    'iter': iter,
                    'Algorithmes': 'ecm',
                    'Dataset': dataset_name,
                    'Type': Type,
                    'Metric': Metric,
                    'Alpha': alpha,
                    'Beta': beta,
                    'Lambda': 'None',
                    'ACC': acc,
                    'ARI': ari,
                    'RI': ri,
                    'NS': ns,
                    'ASW': asw,
                    'J': J,
                    'Time': time_
                }])
                clus_.append({'data': key, 'iter':iter, 'alpha':alpha, 'beta':beta, 'lambda':'None', 'clus': clus})
                Recm = pd.concat([Recm, new_row], ignore_index=True)

Recm.to_csv('experiments/outputs/ieee25/Recm.csv', index=False)
np.save('experiments/outputs/ieee25/Recm.npy', clus_)





# ------------------------ ECMdd Clustering  ------------------------------------------------
datasets = {
    'Abalone': ('Abalone', Xabalone, Yabalone, Cablone, 'Euclidean', 'Numerical'),
    'Ecoli': ('Ecoli', Xecoli, Yecoli, Cecoli, 'Euclidean', 'Numerical'),
    'Glass': ('Glass', Xglass, Yglass, Cglass, 'Euclidean', 'Numerical'),

    'BC': ('BC', Xbc, Ybc, Cbc, 'Hamming', 'Categorical'),
    'Soybean': ('Soybean', Xso, Yso, Cso, 'Hamming', 'Categorical'),
    'Lung': ('Lung', Xlu, Ylu, Clu, 'Hamming', 'Categorical'),

    'BM_SoftDTW': ('BM', Xbm, Ybm, Cbm, 'SoftDTW', 'TimeSeries'),
    'BM_Euclidean': ('BM', Xbm, Ybm, Cbm, 'Euclidean', 'TimeSeries'),

    'ER_SoftDTW': ('ERing', Xer, Yer, Cer, 'SoftDTW', 'TimeSeries'),
    'ER_Euclidean': ('ERing', Xer, Yer, Cer, 'Euclidean', 'TimeSeries'),

    'FM_SoftDTW': ('FM', Xfm, Yfm, Cfm, 'SoftDTW', 'TimeSeries'),
    'FM_Euclidean': ('FM', Xfm, Yfm, Cfm, 'Euclidean', 'TimeSeries'),

    'AF_SoftDTW': ('AF', Xaf, Yaf, Caf, 'SoftDTW', 'TimeSeries'),
    'AF_Euclidean': ('AF', Xaf, Yaf, Caf, 'Euclidean', 'TimeSeries')

}

clus_ = []
for key, (dataset_name, X, Y, C, Metric, Type) in datasets.items():

    if Metric == 'SoftDTW' and Type == 'TimeSeries':
        D = Dmetric(X, soft_dtw)
    elif Metric == 'Euclidean' and Type == 'TimeSeries':
        D = Dmetric(torch.Tensor(X), TSeuclidean)
    elif Metric == 'Euclidean' and Type == 'Numerical':
        D = pairwise_distances(X, metric='euclidean')
    elif Metric == 'Hamming' and Type == 'Categorical':
        X = X.apply(LabelEncoder().fit_transform).to_numpy()
        D = Hamming(X)

    for iter in range(1, maxiter):
        print(f"ECM with {dataset_name} for inter {iter} \n")
        for alpha in alpha_:
            for beta in beta_:
                try:
                    start_time = time.time()
                    clus = ecmdd(x=D, c=C, beta=beta, alpha=alpha, delta=delta_,epsi=0.001, type='pairs')
                    end_time = time.time()

                    # Mtrics
                    betp = pd.DataFrame(clus['betp'])
                    cluster = pd.factorize(betp.apply(lambda row: row.idxmax(), axis=1))[0]
                    cluster = true_lab(Y,cluster)
                    acc = round(accuracy_score(Y, cluster), 3)
                    ari = round(adjusted_rand_score(Y, cluster), 3)
                    ri = round(rand_score(Y, cluster), 3)
                    ns = round(nonspecificity(clus['mass'], clus['F']), 3)
                    J = round(clus['crit'], 3)
                    time_ = round(end_time - start_time, 3)
                    try:
                        asw = round(silhouette_score(X, cluster), 3)
                    except Exception:
                        try:
                            asw = round(silhouette_score(D, cluster), 3)
                        except Exception:
                            asw = 'None'
                except Exception as e:
                    acc = 'None'
                    ari = 'None'
                    ri = 'None'
                    ns = 'None'
                    asw = 'None'
                    J = 'None'
                    time_ = 'None'
                    print(f"Erreur for data={dataset_name}, alpha={alpha}, beta={beta} : {e}")

                # DataFrame
                new_row = pd.DataFrame([{
                    'iter': iter,
                    'Algorithmes': 'ecmdd',
                    'Dataset': dataset_name,
                    'Type': Type,
                    'Metric': Metric,
                    'Alpha': alpha,
                    'Beta': beta,
                    'Lambda': 'None',
                    'ACC': acc,
                    'ARI': ari,
                    'RI': ri,
                    'NS': ns,
                    'ASW': asw,
                    'J': J,
                    'Time': time_
                }])
                clus_.append({'data': key, 'iter': iter, 'alpha':alpha, 'beta':beta, 'lambda':'None', 'clus': clus})
                Recmdd = pd.concat([Recmdd, new_row], ignore_index=True)

Recmdd.to_csv('experiments/outputs/ieee25/Recmdd.csv', index=False)
np.save('experiments/outputs/ieee25/Recmdd.npy', clus_)





# ------------------------ CatECM Clustering  ------------------------------------------------
datasets = {
    'BC': ('BC', Xbc, Ybc, Cbc, 'Hamming', 'Categorical'),
    'Soybean': ('Soybean', Xso, Yso, Cso, 'Hamming', 'Categorical'),
    'Lung': ('Lung', Xlu, Ylu, Clu, 'Hamming', 'Categorical')
}

clus_ = []
for key, (dataset_name, X, Y, C, Metric, Type) in datasets.items():

    X_ = X.apply(LabelEncoder().fit_transform).to_numpy()
    D = Hamming(X_)

    for iter in range(1, maxiter):
        print(f"ECM with {dataset_name} for inter {iter} \n")
        for alpha in alpha_:
            for beta in beta_:
                try:
                    start_time = time.time()
                    clus = catecm(X=X.values, c=C, beta=beta, alpha=alpha, delta=delta_, type='pairs')
                    end_time = time.time()

                    # Mtrics
                    betp = pd.DataFrame(clus['betp'])
                    cluster = pd.factorize(betp.apply(lambda row: row.idxmax(), axis=1))[0]
                    cluster = true_lab(Y,cluster)
                    acc = round(accuracy_score(Y, cluster), 3)
                    ari = round(adjusted_rand_score(Y, cluster), 3)
                    ri = round(rand_score(Y, cluster), 3)
                    ns = round(nonspecificity(clus['mass'], clus['F']), 3)
                    J = round(clus['crit'], 3)
                    time_ = round(end_time - start_time, 3)
                    try:
                        asw = round(silhouette_score(X, cluster), 3)
                    except Exception:
                        try:
                            asw = round(silhouette_score(D, cluster), 3)
                        except Exception:
                            asw = 'None'
                except Exception as e:
                    acc = 'None'
                    ari = 'None'
                    ri = 'None'
                    ns = 'None'
                    asw = 'None'
                    J = 'None'
                    time_ = 'None'
                    print(f"Erreur for data={dataset_name}, alpha={alpha}, beta={beta} : {e}")

                # DataFrame
                new_row = pd.DataFrame([{
                    'iter': iter,
                    'Algorithmes': 'catecm',
                    'Dataset': dataset_name,
                    'Type': Type,
                    'Metric': Metric,
                    'Alpha': alpha,
                    'Beta': beta,
                    'Lambda': 'None',
                    'ACC': acc,
                    'ARI': ari,
                    'RI': ri,
                    'NS': ns,
                    'ASW': asw,
                    'J': J,
                    'Time': time_
                }])
                clus_.append({'data': key, 'iter': iter, 'alpha':alpha, 'beta':beta, 'lambda':'None', 'clus': clus})
                Rcatecm = pd.concat([Rcatecm, new_row], ignore_index=True)

Rcatecm.to_csv('experiments/outputs/ieee25/Rcatecm.csv', index=False)
np.save('experiments/outputs/ieee25/Rcatecm.npy', clus_)




# ------------------------ KMeans Clustering  ------------------------------------------------
datasets = {
    'Abalone': ('Abalone', Xabalone, Yabalone, Cablone, 'Euclidean', 'Numerical'),
    'Ecoli': ('Ecoli', Xecoli, Yecoli, Cecoli, 'Euclidean', 'Numerical'),
    'Glass': ('Glass', Xglass, Yglass, Cglass, 'Euclidean', 'Numerical')
}

clus_ = []
for key, (dataset_name, X, Y, C, Metric, Type) in datasets.items():

    for iter in range(1, maxiter):
        print(f"ECM with {dataset_name} for inter {iter} \n")
        try:
            start_time = time.time()
            clus = KMeans(n_clusters=C, max_iter=20, tol=0.001).fit(X)
            end_time = time.time()

            # Mtrics
            cluster = clus.labels_
            cluster = true_lab(Y,cluster)
            acc = round(accuracy_score(Y, cluster), 3)
            ari = round(adjusted_rand_score(Y, cluster), 3)
            ri = round(rand_score(Y, cluster), 3)
            J = round(clus.inertia_, 3)
            time_ = round(end_time - start_time, 3)
            try:
                asw = round(silhouette_score(X, cluster), 3)
            except Exception:
                try:
                    D = pairwise_distances(X, metric='euclidean')
                    asw = round(silhouette_score(D, cluster), 3)
                except Exception:
                    asw = 'None'
        except Exception as e:
            acc = 'None'
            ari = 'None'
            ri = 'None'
            ns = 'None'
            asw = 'None'
            J = 'None'
            time_ = 'None'
            print(f"Erreur for data={dataset_name}: {e}")
        
        # DataFrame
        new_row = pd.DataFrame([{
            'iter': iter,
            'Algorithmes': 'kmeans',
            'Dataset': dataset_name,
            'Type': Type,
            'Metric': Metric,
            'Alpha': 'None',
            'Beta': 'None',
            'Lambda': 'None',
            'ACC': acc,
            'ARI': ari,
            'RI': ri,
            'NS': 'None',
            'ASW': asw,
            'J': J,
            'Time': time_
        }])
        clus_.append({'data': key, 'iter': iter, 'alpha':'None', 'beta':'None', 'lambda':'None', 'clus': clus})
        Rkmeans = pd.concat([Rkmeans, new_row], ignore_index=True)

Rkmeans.to_csv('experiments/outputs/ieee25/Rkmeans.csv', index=False)
np.save('experiments/outputs/ieee25/Rkmeans.npy', clus_)






# ------------------------ KModes Clustering  ------------------------------------------------
datasets = {
    'BC': ('BC', Xbc, Ybc, Cbc, 'Hamming', 'Categorical'),
    'Soybean': ('Soybean', Xso, Yso, Cso, 'Hamming', 'Categorical'),
    'Lung': ('Lung', Xlu, Ylu, Clu, 'Hamming', 'Categorical')
}

clus_ = []
for key, (dataset_name, X, Y, C, Metric, Type) in datasets.items():

    for iter in range(1, maxiter):
        print(f"ECM with {dataset_name} for inter {iter} \n")
        try:
            start_time = time.time()
            clus = KModes(n_clusters=C, max_iter=20).fit(X)
            end_time = time.time()

            # Mtrics
            cluster = clus.labels_
            cluster = true_lab(Y,cluster)
            acc = round(accuracy_score(Y, cluster), 3)
            ari = round(adjusted_rand_score(Y, cluster), 3)
            ri = round(rand_score(Y, cluster), 3)
            J = round(clus.cost_, 3)
            time_ = round(end_time - start_time, 3)
            try:
                asw = round(silhouette_score(X, cluster), 3)
            except Exception:
                try:
                    X_ = X.apply(LabelEncoder().fit_transform).to_numpy()
                    D = Hamming(X_)
                    asw = round(silhouette_score(D, cluster), 3)
                except Exception:
                    asw = 'None'
        except Exception as e:
            acc = 'None'
            ari = 'None'
            ri = 'None'
            ns = 'None'
            asw = 'None'
            J = 'None'
            time_ = 'None'
            print(f"Erreur for data={dataset_name} : {e}")

        # DataFrame
        new_row = pd.DataFrame([{
            'iter': iter,
            'Algorithmes': 'kmodes',
            'Dataset': dataset_name,
            'Type': Type,
            'Metric': Metric,
            'Alpha': 'None',
            'Beta': 'None',
            'Lambda': 'None',
            'ACC': acc,
            'ARI': ari,
            'RI': ri,
            'NS': 'None',
            'ASW': asw,
            'J': J,
            'Time': time_
        }])
        clus_.append({'data': key, 'iter': iter, 'alpha':'None', 'beta':'None', 'lambda':'None', 'clus': clus})
        Rkmodes = pd.concat([Rkmodes, new_row], ignore_index=True)

Rkmodes.to_csv('experiments/outputs/ieee25/Rkmodes.csv', index=False)
np.save('experiments/outputs/ieee25/Rkmodes.npy', clus_)





# ------------------------ TSKmeans Clustering  ------------------------------------------------
datasets = {
    'BM_SoftDTW': ('BM', Xbm, Ybm, Cbm, 'SoftDTW', 'TimeSeries'),
    'BM_Euclidean': ('BM', Xbm, Ybm, Cbm, 'Euclidean', 'TimeSeries'),

    'ER_SoftDTW': ('ERing', Xer, Yer, Cer, 'SoftDTW', 'TimeSeries'),
    'ER_Euclidean': ('ERing', Xer, Yer, Cer, 'Euclidean', 'TimeSeries'),

    'FM_SoftDTW': ('FM', Xfm, Yfm, Cfm, 'SoftDTW', 'TimeSeries'),
    'FM_Euclidean': ('FM', Xfm, Yfm, Cfm, 'Euclidean', 'TimeSeries'),

    'AF_SoftDTW': ('AF', Xaf, Yaf, Caf, 'SoftDTW', 'TimeSeries'),
    'AF_Euclidean': ('AF', Xaf, Yaf, Caf, 'Euclidean', 'TimeSeries')
}

clus_ = []
for key, (dataset_name, X, Y, C, Metric, Type) in datasets.items():

    if Metric == 'SoftDTW' and Type == 'TimeSeries':
        D = Dmetric(X, soft_dtw)
        Metric_ = 'softdtw'
    elif Metric == 'Euclidean' and Type == 'TimeSeries':
        D = Dmetric(torch.Tensor(X), TSeuclidean)
        Metric_ = 'euclidean'

    for iter in range(1, maxiter):
        print(f"ECM with {dataset_name} for inter {iter} \n")
        try:
            start_time = time.time()
            clus = TimeSeriesKMeans(n_clusters=C, metric=Metric_, tol=0.001, max_iter=20).fit(X)
            end_time = time.time()
            # Mtrics
            cluster = clus.labels_
            cluster = true_lab(Y,cluster)
            acc = round(accuracy_score(Y, cluster), 3)
            ari = round(adjusted_rand_score(Y, cluster), 3)
            ri = round(rand_score(Y, cluster), 3)
            J = round(clus.inertia_, 3)
            time_ = round(end_time - start_time, 3)
            try:
                asw = round(silhouette_score(X, cluster), 3)
            except Exception:
                try:
                    asw = round(silhouette_score(D, cluster), 3)
                except Exception:
                    asw = 'None'
        except Exception as e:
            acc = 'None'
            ari = 'None'
            ri = 'None'
            ns = 'None'
            asw = 'None'
            J = 'None'
            time_ = 'None'
            print(f"Erreur for data={dataset_name} : {e}")

        # DataFrame
        new_row = pd.DataFrame([{
            'iter': iter,
            'Algorithmes': 'tskmeans',
            'Dataset': dataset_name,
            'Type': Type,
            'Metric': Metric,
            'Alpha': 'None',
            'Beta': 'None',
            'Lambda': 'None',
            'ACC': acc,
            'ARI': ari,
            'RI': ri,
            'NS': 'None',
            'ASW': asw,
            'J': J,
            'Time': time_
        }])
        clus_.append({'data': key, 'iter': iter, 'alpha':'None', 'beta':'None', 'lambda':'None', 'clus': clus})
        Rtskmeans = pd.concat([Rtskmeans, new_row], ignore_index=True)

Rtskmeans.to_csv('experiments/outputs/ieee25/Rtskmeans.csv', index=False)
np.save('experiments/outputs/ieee25/Rtskmeans.npy', clus_)







# ------------------------ Soft-ECM Clustering  ------------------------------------------------
datasets = {
    'Abalone': ('Abalone', Xabalone, Yabalone, Cablone, 'Euclidean', 'Numerical'),
    'Ecoli': ('Ecoli', Xecoli, Yecoli, Cecoli, 'Euclidean', 'Numerical'),
    'Glass': ('Glass', Xglass, Yglass, Cglass, 'Euclidean', 'Numerical'),

    'BC': ('BC', Xbc, Ybc, Cbc, 'Hamming', 'Categorical'),
    'Soybean': ('Soybean', Xso, Yso, Cso, 'Hamming', 'Categorical'),
    'Lung': ('Lung', Xlu, Ylu, Clu, 'Hamming', 'Categorical'),

    'BM_SoftDTW': ('BM', Xbm, Ybm, Cbm, 'SoftDTW', 'TimeSeries'),
    'BM_Euclidean': ('BM', Xbm, Ybm, Cbm, 'Euclidean', 'TimeSeries'),

    'ER_SoftDTW': ('ERing', Xer, Yer, Cer, 'SoftDTW', 'TimeSeries'),
    'ER_Euclidean': ('ERing', Xer, Yer, Cer, 'Euclidean', 'TimeSeries'),

    'FM_SoftDTW': ('FM', Xfm, Yfm, Cfm, 'SoftDTW', 'TimeSeries'),
    'FM_Euclidean': ('FM', Xfm, Yfm, Cfm, 'Euclidean', 'TimeSeries'),

    'AF_SoftDTW': ('AF', Xaf, Yaf, Caf, 'SoftDTW', 'TimeSeries'),
    'AF_Euclidean': ('AF', Xaf, Yaf, Caf, 'Euclidean', 'TimeSeries')
}

clus_ = []
for key, (dataset_name, X, Y, C, Metric, Type) in datasets.items():

    if Metric == 'SoftDTW' and Type == 'TimeSeries':
        D = Dmetric(X, soft_dtw)
        Metric_ = SoftDTWLossPyTorch(gamma=0.1)
    elif Metric == 'Euclidean' and Type == 'TimeSeries':
        D = Dmetric(torch.Tensor(X), TSeuclidean)
        Metric_ = euclidean
    elif Metric == 'Euclidean' and Type == 'Numerical':
        D = pairwise_distances(X, metric='euclidean')
        Metric_ = euclidean
    elif Metric == 'Hamming' and Type == 'Categorical':
        X = X.apply(LabelEncoder().fit_transform).to_numpy()
        D = Hamming(X)
        Metric_ = LossHamming

    for iter in range(1, maxiter):
        print(f"ECM with {dataset_name} for inter {iter} \n")
        for alpha in alpha_:
            for beta in beta_:
                for lbda in lbda_:
                    try:
                        start_time = time.time()
                        clus = ECM_Clustering(n_clusters = C, focal_max_size=2)
                        clus.alpha_ = alpha
                        clus.beta_ = beta
                        clus.delta_ = delta_
                        clus.lbda_ = lbda
                        clus.metric = Metric_
                        clus.nb_outer_it = 20
                        clus.nb_inner_it = 10
                        clus.inner_lr = 1e-1
                        clus.inner_convergence_criteria = 1e-1
                        clus.outter_convergence_criteria = 1e-3

                        clus.fit(X)
                        end_time = time.time()
                        cluss = extractMass(clus.M, clus.F)

                        # Mtrics
                        betp = pd.DataFrame(cluss['betp'])
                        cluster = pd.factorize(betp.apply(lambda row: row.idxmax(), axis=1))[0]
                        cluster = true_lab(Y,cluster)
                        acc = round(accuracy_score(Y, cluster), 3)
                        ari = round(adjusted_rand_score(Y, cluster), 3)
                        ri = round(rand_score(Y, cluster), 3)
                        ns = round(nonspecificity(cluss['mass'], cluss['F']), 3)
                        J = round(clus.J.item(), 3)
                        time_ = round(end_time - start_time, 3)
                        try:
                            asw = round(silhouette_score(X, cluster), 3)
                        except Exception:
                            try:
                                asw = round(silhouette_score(D, cluster), 3)
                            except Exception:
                                asw = 'None'
                    except Exception as e:
                        acc = 'None'
                        ari = 'None'
                        ri = 'None'
                        ns = 'None'
                        asw = 'None'
                        J = 'None'
                        time_ = 'None'
                        print(f"Erreur for data={dataset_name}, alpha={alpha}, beta={beta} : {e}")

                    # DataFrame
                    new_row = pd.DataFrame([{
                        'iter': iter,
                        'Algorithmes': 'softecm',
                        'Dataset': dataset_name,
                        'Type': Type,
                        'Metric': Metric,
                        'Alpha': alpha,
                        'Beta': beta,
                        'Lambda': lbda,
                        'ACC': acc,
                        'ARI': ari,
                        'RI': ri,
                        'NS': ns,
                        'ASW': asw,
                        'J': J,
                        'Time': time_
                    }])
                    clus_.append({'data': key, 'iter': iter, 'alpha':alpha, 'beta':beta, 'lambda':lbda, 'clus': clus})
                    Rsoftecm = pd.concat([Rsoftecm, new_row], ignore_index=True)

Rsoftecm.to_csv('experiments/outputs/ieee25/Rsoftecm.csv', index=False)
np.save('experiments/outputs/ieee25/Rsoftecm.npy', clus_)