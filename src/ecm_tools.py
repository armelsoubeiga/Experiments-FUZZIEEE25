import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import silhouette_score

# Creates an object of class "credpart"
def extractMass(mass, F, J=None, param=None):
    mass = mass.numpy()
    F = F.numpy()
    if J is not None:
        J = J.numpy()
    if param is not None:
        param = [p.numpy() if torch.is_tensor(p) else p for p in param]

    n = mass.shape[0]
    c = F.shape[1]
    if any(F[0, :] == 1):
        F = np.vstack((np.zeros(c), F))  # add the empty set
        mass = np.hstack((np.zeros((n, 1)), mass))
    
    f = F.shape[0]
    card = np.sum(F, axis=1)
    conf = mass[:, 0]  # degree of conflict
    C = 1 / (1 - conf)
    mass_n = C[:, np.newaxis] * mass[:, 1:f]  # normalized mass function
    pl = np.matmul(mass, F) # unnormalized plausibility
    pl_n = C[:, np.newaxis] * pl # normalized plausibility
    p = pl / np.sum(pl, axis=1, keepdims=True) # plausibility-derived probability
    bel = mass[:, card == 1] # unnormalized belief
    bel_n = C[:, np.newaxis] * bel # normalized belief
    y_pl = np.argmax(pl, axis=1) # maximum plausibility cluster
    y_bel = np.argmax(bel, axis=1) # maximum belief cluster
    Y = F[np.argmax(mass, axis=1), :] # maximum mass set of clusters

    # non dominated elements
    Ynd = np.zeros((n, c))
    for i in range(n):
        ii = np.where(pl[i, :] >= bel[i, y_bel[i]])[0]
        Ynd[i, ii] = 1
    nonzero_card = np.where(card != 0)  
    P = np.zeros_like(F)
    P[nonzero_card] = F[nonzero_card] / card[nonzero_card, np.newaxis]
    P[0, :] = 0
    betp = np.matmul(mass, P) # unnormalized pignistic probability
    betp_n = C[:, np.newaxis] * betp # normalized pignistic probability
    lower_approx, upper_approx = [], []
    lower_approx_nd, upper_approx_nd = [], []
    nclus = np.sum(Y, axis=1)
    outlier = np.where(nclus == 0)[0]
    nclus_nd = np.sum(Ynd, axis=1)
    for i in range(c):
        upper_approx.append(np.where(Y[:, i] == 1)[0])  # upper approximation
        lower_approx.append(np.where((Y[:, i] == 1) & (nclus == 1))[0])  # upper approximation
        upper_approx_nd.append(np.where(Ynd[:, i] == 1)[0])  # upper approximation
        lower_approx_nd.append(np.where((Ynd[:, i] == 1) & (nclus_nd == 1))[0])  # upper approximation
    
    # Nonspecificity
    card = np.concatenate(([c], card[1:f]))
    Card = np.tile(card, (n, 1))
    N = np.sum(np.log(Card) * mass) / np.log(c) / n

    clus = {'conf': conf, 'F': F, 'mass': mass, 'mass_n': mass_n, 'pl': pl, 
            'pl_n': pl_n, 'bel': bel, 'bel_n': bel_n, 'y_pl': y_pl, 'y_bel': y_bel, 
            'Y': Y, 'betp': betp, 'betp_n': betp_n, 'p': p, 'upper_approx': upper_approx,
            'lower_approx': lower_approx, 'Ynd': Ynd, 'upper_approx_nd': upper_approx_nd,
            'lower_approx_nd': lower_approx_nd, 'N': N, 'outlier': outlier, 'J': J, 
            'param': param}
    return clus


# get cluster name using mass
def get_ensembles(table):
    result = []
    for row in table:
        row_str = 'Cl_' + '_'.join([str(i + 1) if elem == 1 else str(int(elem)) for i, elem in enumerate(row) if elem != 0])
        result.append(row_str)
    result[0] = 'Cl_atypique'
    #result[-1] = 'Cl_incertains'
    cleaned_result = [''.join(ch for i, ch in enumerate(row_str) if ch != '_' or (i > 0 and row_str[i-1] != '_')) for row_str in result]
    return cleaned_result


# Plot function
def _ecm_plot(clus, X, J_all, V):
    X = pd.DataFrame(X)
    mas = pd.DataFrame(clus['mass'])
    
    cols = get_ensembles(clus['F'])
    mas.columns = cols
    cluster = pd.Categorical(mas.apply(lambda row: row.idxmax(), axis=1))
    pcolor = sns.color_palette("Dark2", n_colors=len(cluster.unique()))
    
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 2], height_ratios=[1, 1])
    
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[0, 0])
    ax4 = fig.add_subplot(gs[1, 0])

    # First zone
    sns.scatterplot(data=X, x=X.columns[0], y=X.columns[1], ax=ax1)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set(xlabel='', ylabel='')
    
    # Second zone
    scatter = sns.scatterplot(data=X, x=X.columns[0], y=X.columns[1], 
                              hue=cluster, palette=pcolor, ax=ax2)
    sns.scatterplot(data=V, x=V.columns[0], y=V.columns[1], 
                    color='black', s=150, marker='X', ax=ax2)
    scatter.legend(fontsize='7') 
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set(xlabel='', ylabel='')

    # Third window
    ax3.plot(J_all, color='black', linewidth=1)
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    
    # Last window
    silhouette_index = silhouette_score(X, cluster.codes)
    ax4.text(0.5, 0.5, 'Silhouette Index: {:.2f}'.format(silhouette_index), 
             horizontalalignment='center', verticalalignment='center', 
             transform=ax4.transAxes)
    ax4.set_xticklabels([])
    ax4.set_yticklabels([])

    plt.tight_layout()
    plt.show()

def ecm_plot(clus, X):
    res = extractMass(clus.M, 
                      torch.where(clus.S, torch.tensor(1), torch.tensor(0)), 
                      J=None, 
                      param={clus.C, clus.alpha_, 
                             clus.beta_, clus.delta_, clus.lbda_})
    V = pd.DataFrame(clus.V.squeeze().numpy()).loc[lambda df: ~(df == 0).all(axis=1)]
    X = pd.DataFrame(X.squeeze())
    mas = pd.DataFrame(res['mass'])
    
    cols = get_ensembles(res['F'])
    mas.columns = cols
    cluster = pd.Categorical(mas.apply(lambda row: row.idxmax(), axis=1))
    pcolor = sns.color_palette("Dark2", n_colors=len(cluster.unique()))
    
    plt.figure(figsize=(10, 8))

    scatter = sns.scatterplot(data=X, x=X.columns[0], y=X.columns[1], hue=cluster, palette=pcolor)
    sns.scatterplot(data=V, x=V.columns[0], y=V.columns[1], color='black', s=150, marker='X')
    scatter.legend(fontsize='7') 

    plt.tight_layout()
    plt.show()



def ani_ecm_plot(clus, X, ax):
    res = extractMass(clus.M, 
                      torch.where(clus.S, torch.tensor(1), torch.tensor(0)), 
                      J=None, 
                      param={clus.C, clus.alpha_, 
                             clus.beta_, clus.delta_, clus.lbda_})
    V = pd.DataFrame(clus.V.squeeze().numpy()).loc[lambda df: ~(df == 0).all(axis=1)]
    X = pd.DataFrame(X.squeeze())
    mas = pd.DataFrame(res['mass'])
    
    cols = get_ensembles(res['F'])
    mas.columns = cols
    cluster = pd.Categorical(mas.apply(lambda row: row.idxmax(), axis=1))
    pcolor = sns.color_palette("Dark2", n_colors=len(cluster.unique()))
    
    scatter = sns.scatterplot(data=X, x=X.columns[0], y=X.columns[1], hue=cluster, palette=pcolor, ax=ax)
    sns.scatterplot(data=V, x=V.columns[0], y=V.columns[1], color='black', s=150, marker='X', ax=ax)
    scatter.legend(fontsize='7') 

    ax.set_title(f'lbda_ = {clus.lbda_}, delta_ = {clus.delta_}, beta_ = {clus.beta_}, alpha_ = {clus.alpha_}', fontsize=10, fontname='Arial')


def nonspecificity(mass, F):
    """Compute the nonspecificity of a credal partition.

    Parameters
    ----------
    mass : ndarray (n, F size)
            The credal partition.

    F : array of length of focalsets
        The folcalsets.

    Returns
    -------
    NS : float
        The nonspecificity of the credal partition.

    """
    focalsets = [tuple(index + 1 for index in row.nonzero()[0]) for row in F]
    n_samples = mass.shape[0]
    len_fs = [len(fs) for fs in focalsets if fs != tuple()]
    len_fs = np.array(len_fs)
    NS = np.sum(len_fs * mass[:, 1:]) + np.sum(
        mass[:, 0] * np.log2(max(len_fs)))
    NS /= n_samples * np.log2(max(len_fs))
    return NS

def Dmetric(X, metric):
    """Compute the matrix of dissimilarity of a ts metric.
    """
    n_series = X.shape[0]
    D = np.zeros((n_series, n_series))

    for i in range(n_series):
        for j in range(i, n_series):  # Fill only upper triangle and diagonal
            dist = metric(X[i], X[j])
            D[i, j] = dist
            D[j, i] = dist  # Ensure symmetry

    return D


def TSeuclidean(x, y):
    """Euclidean metric between batches of matrices (e.g. multidimensional time series).
    """
    return torch.sum((x - y) ** 2, dim=(0, 1)).item()




def ts_ecm_plot(X, V, clus, plot_centers=True):
    """
    Plot the results of ecm clustering algorithm for ts.

    Parameters
    ----------
    X : array-like
        The time series data.
    V : array-like
        The medoid (center) time series for each cluster.
    clus : dict
        The clustering results, with 'mass' and 'F' keys.
    plot_centers : bool, optional
        If True, plot the cluster centers in color. If False, plot the individual series and the cluster centers in black. 
        Default is True.
    """
    # Get the cluster labels from 'clus'
    mas = pd.DataFrame(clus['mass'])
    mas.columns = get_ensembles(clus['F'])
    cluster = pd.Categorical(mas.apply(lambda row: row.idxmax(), axis=1))

    # Number of clusters
    unique_clusters = np.unique(cluster)
    k = len(unique_clusters)

    # Number grid
    grid_cols = int(np.ceil(np.sqrt(k)))
    grid_rows = int(np.ceil(k / grid_cols))

    fig, axes = plt.subplots(nrows=grid_rows, ncols=grid_cols, figsize=(10, 6))
    plt.rcParams["figure.dpi"] = 100

    colors = plt.cm.viridis(np.linspace(0, 1, k))  

    for i in range(grid_rows):
        for j in range(grid_cols):
            idx = i * grid_cols + j
            if idx < k:
                ax = axes[i, j]
                if plot_centers:
                    ax.plot(V[idx], color=colors[idx], linewidth=2)
                else:
                    cluster_series = X[cluster == unique_clusters[idx]]
                    for series in cluster_series:
                        ax.plot(series, color=colors[idx], alpha=0.5)
                        ax.plot(V[idx], color='black', linewidth=2)
                ax.set_title(f'Cluster {unique_clusters[idx]}')

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            else:
                axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()
    return fig, axes


def diffHammingLoss_(y_true, y_pred, epsilon=1e-12):
    """
    Computes the differentiable Hamming Loss between true and predicted labels.
    
    Parameters:
    y_true (numpy.ndarray): True labels.
    y_pred (numpy.ndarray): Predicted labels.
    epsilon (float): Small value to avoid division by zero in the sigmoid function.
    
    Returns:
    float: Differentiable Hamming Loss.
    """
    # Sigmoid function to approximate the indicator function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    diff = np.abs(y_true - y_pred)
    diff_sigmoid = sigmoid(diff)
    loss = np.mean(diff_sigmoid)
    
    return loss


def diffHammingLoss(y_true, y_pred, epsilon=1e-12):
    """
    Computes the differentiable Hamming Loss between true and predicted labels.
    
    Parameters:
    y_true (numpy.ndarray): True labels.
    y_pred (numpy.ndarray): Predicted labels.
    epsilon (float): Small value to avoid division by zero in the sigmoid function.
    
    Returns:
    float: Differentiable Hamming Loss.
    """
    # Sigmoid function to approximate the indicator function
    def sigmoid(x):
        return 1 / (1 + torch.exp(-x))
    diff = torch.abs(y_true - y_pred)
    diff_sigmoid = sigmoid(diff)
    diff_sigmoid_tensor = torch.tensor(diff_sigmoid, dtype=torch.float32, requires_grad=True)
    loss = torch.mean(diff_sigmoid_tensor, (1,2) )
    
    return loss