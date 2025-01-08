import numpy as np
import random
import math
import pandas as pd
from matplotlib import pyplot as plt
from ecm_ts import ECM_Clustering
from ecm_tools import get_ensembles
from tslearn.metrics import SoftDTWLossPyTorch
import pickle
import os

if os.path.exists('test1_series.pkl'):
    print("load time series")
    with open('test1_series.pkl','rb') as f:
        series = pickle.load(f)

    """plt.figure(figsize=(8, 4))
    plt.plot(series[0], 'r')
    plt.plot(series[1], 'r')
    plt.plot(series[2], 'r')
    plt.plot(series[50], 'g')
    plt.plot(series[51], 'g')
    plt.plot(series[52], 'g')
    plt.plot(series[100], 'b')
    plt.plot(series[101], 'b')
    plt.plot(series[102], 'b')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title("Cylinder-Bell-Funnel")
    plt.show()
    """
else:

    # cylinder bell funnel based on "Learning comprehensible descriptions of multivariate time series"

    def generate_bell(length, amplitude, default_variance):
        bell = np.random.normal(0, default_variance, length) + amplitude * np.arange(length)/length
        return bell


    def generate_funnel(length, amplitude, default_variance):
        funnel = np.random.normal(0, default_variance, length) + amplitude * np.arange(length)[::-1]/length
        return funnel


    def generate_cylinder(length, amplitude, default_variance):
        cylinder = np.random.normal(0, default_variance, length) + amplitude
        return cylinder


    std_generators = [generate_bell, generate_funnel, generate_cylinder]


    def generate_pattern_data(length, avg_pattern_length, avg_amplitude, default_variance = 1, variance_pattern_length = 10, variance_amplitude = 2, generators = std_generators):
        data = np.random.normal(0, default_variance, length)
        current_start = 4+random.randint(0, int(avg_pattern_length/4))
        current_length = current_length = max(1, math.ceil(random.gauss(avg_pattern_length, variance_pattern_length)))
        
        generator = random.choice(generators)
        current_amplitude = random.gauss(avg_amplitude, variance_amplitude)
        
        pattern = generator(current_length, current_amplitude, default_variance)
            
        data[current_start : current_start + current_length] = pattern
        
        return data

    train_size = 100
    length = 40

    avg_amplitude = 10
    avg_pattern_length = 20
    variance_pattern_length = 5 
    variance_amplitude = 0.4
    clean = 0.2

    series = [generate_pattern_data(length, avg_pattern_length, 
                                    avg_amplitude, clean, 
                                    variance_pattern_length,
                                    variance_amplitude,
                                    [generate_bell]) for i in range(int(train_size/2))]

    series += [generate_pattern_data(length, avg_pattern_length, 
                                    avg_amplitude, clean, 
                                    variance_pattern_length,
                                    variance_amplitude,
                                    [generate_funnel]) for i in range(int(train_size/2))]

    series += [generate_pattern_data(length, avg_pattern_length, 
                                    avg_amplitude, clean, 
                                    variance_pattern_length,
                                    variance_amplitude,
                                    [generate_cylinder]) for i in range(int(train_size/2))]

    series = np.stack(series)
    series = np.stack(series)
    with open('test1_series.pkl','wb') as f:
        pickle.dump(series, f)

    """print(np.sum((series[10,:]-series[12,:])**2), np.sum((series[10,:]-series[52,:])**2))

    plt.plot(series[:50,:].T)
    plt.show()
    plt.plot(series[50:,:].T)
    plt.show()"""


n_clusters = 3

"""clus=ECM_Clustering(n_clusters, focal_max_size=2)
clus.lbda_= 1.5
clus.nb_outer_it = 100
clus.nb_inner_it = 40
clus.delta_ = 600
clus.inner_lr = 1e-3
clus.inner_convergence_criteria = 1e-3
clus.outter_convergence_criteria = 1e-3

clus.fit(series)

M = pd.DataFrame(clus.M, columns=get_ensembles(clus.F))

plt.figure(figsize=(8, 4))
plt.plot(M.index, M['Cl_atypique'], marker='^', label=r'$m(\emptyset)$', color='blue')
plt.plot(M.index, M['Cl_1'], marker='o', label=r'$m(\omega_1)$', color='green')
plt.plot(M.index, M['Cl_2'], marker='s', label=r'$m(\omega_2)$', color='red')
plt.plot(M.index, M['Cl_3'], marker='v', label=r'$m(\omega_3)$', color='orange')
plt.plot(M.index, M['Cl_1_2'], marker='d', label=r'$m(\omega_1, \omega_2)$', color='purple')
plt.plot(M.index, M['Cl_1_3'], marker='1', label=r'$m(\omega_1, \omega_3)$', color='brown')
plt.plot(M.index, M['Cl_2_3'], marker='*', label=r'$m(\omega_2, \omega_3)$', color='gold')
plt.xlabel('Index')
plt.ylabel('Mass of belief')
plt.title('Mass of belief of each cluster by Soft-ECM (Euclidean metric)')
plt.legend()
plt.show()"""

# start again with soft-DTW


clus=ECM_Clustering(n_clusters, focal_max_size=2)
clus.lbda_= 1
clus.nb_outer_it = 80
clus.nb_inner_it = 50
clus.inner_lr = 1e-3
clus.inner_convergence_criteria = 1e-3
clus.outter_convergence_criteria = 1e-3
clus.delta_ = 300
clus.metric = SoftDTWLossPyTorch(gamma=0.1)

clus.fit(series)

M = pd.DataFrame(clus.M, columns=get_ensembles(clus.F))

plt.figure(figsize=(8, 4))
plt.plot(M.index, M['Cl_atypique'], marker='^', label=r'$m(\emptyset)$', color='blue')
plt.plot(M.index, M['Cl_1'], marker='o', label=r'$m(\omega_1)$', color='green')
plt.plot(M.index, M['Cl_2'], marker='s', label=r'$m(\omega_2)$', color='red')
plt.plot(M.index, M['Cl_3'], marker='v', label=r'$m(\omega_3)$', color='orange')
plt.plot(M.index, M['Cl_1_2'], marker='d', label=r'$m(\omega_1, \omega_2)$', color='purple')
plt.plot(M.index, M['Cl_1_3'], marker='1', label=r'$m(\omega_1, \omega_3)$', color='brown')
plt.plot(M.index, M['Cl_2_3'], marker='*', label=r'$m(\omega_2, \omega_3)$', color='gold')
plt.xlabel('Index')
plt.ylabel('Mass of belief')
plt.title('Mass of belief of each cluster by Soft-ECM (SoftDTW metric)')
plt.legend()
plt.show()
