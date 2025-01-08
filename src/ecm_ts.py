import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import torch
import torch.optim as optim
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
import logging


def euclidean(x, y):
    """Euclidean metric between batches of matrices (e.g. multidimensional time series).
    This function make a one-to-one comparison of time series.

    Args:
        x (torch.tensor): dimension (bs, sz, d), where bs is the batch size, sz, the length of time series and d its dimension.
        y (torch.tensor): dimension (bs, sz, d), same as x

    Returns:
        torch.tensor: dimension (bs), return the one-to-one euclidean distances between matrices
    """
    return torch.sum((x - y) ** 2, (1, 2))


class ECM_Clustering:
    """
    Evidential C-Means algorithm with flexible (non-distance) metric.
    This algorithm adapts the classical ECM algorithm [Masson et Denoeux, 2008] without
    constraining the use of a metric (more specifically, it does not require to have
    a distance with the trianglular inegality's property).

    It assumes the objects to classify are vectors of vectors (of dimension `d`).

    A possible application context is the clustering of multidimensional time series.
    Each time series is made of several multidimensional points of dimension `d`. Thus,
    different metrics can be used to cluster examples (Euclidean, DTW, divergence, etc.)


    Parameters
    ----------
    n_clusters: int (default: 3)
        Number of singleton clusters (in evidential framework, there are also the mixed-clustered).

    metric: lambda function (default: eulidean metric)
        Comparison between two objects.
        Such metric function must take two parameters X, Y of size (batch_size, sz, d) and
            it returns a list of distances of length batch_size (one 2 one distance)
    focal_max_size: int (default: None)
        Maximum size of a focal element to consider. This parameters allows to discard focal
        elements with large size, and that are difficult to interpret. By defaut, the method
        computes the clustering for all focal elements (power set of classes).

    Attributes
    ----------
    M : numpy.ndarray (nb_examples, dim)
        Mass matrix. It assigns each example to the :math:`2^C` clusters.

    V : numpy.ndarray (nb_clusters, sz, d)
        Centroids description

    alpha_: float (default 1.0)
        Loss parameter. :math:`\alpha` controls the penalization of the subsets in :math:`\Omega` of high cardinality.

    beta_: float (default 2.0)
        Loss parameter from the Davé's methods. Must be strictly greater than 1.
        :math:`\beta` is a weighting exponent that controls the fuzziness of the partition.

    delta_: float (default 1.0)
        Loss parameter from the Davé's methods. \delta controls the amount of data considered as outliers.

    lbda_: float (default 1.0)
        Hyper parameters parameter to control the importance of the faithfullness of
        focal elements centroids wrt each others (according to the metric)

    metric:
        Differentiable function having two collections of multidimensional time series
        as inputs and return floats (see function euclidean for more details).
        This parameter can typically be an Euclidean metric or a softDTW metric

    nb_inner_it: int (default 10)
        Number of maximum iterations for V optimization

    nb_outer_it: int (default 10)
        Number of maximum iterations for the overall optimization

    inner_lr: float (default 1e-3)
        Learning rate for the inner loop.

    inner_convergence_criteria: float (default None)
        If defined, the inner loop is stopped when the criteria is satisfyied.
        The criteria is the relative difference between successive errors that has to be
        lower than the parameter value. A typical value depends on the `inner_lr` value.

    outter_convergence_criteria: float (default None)
        If defined, the outter loop is stopped when the criteria is satisfyied.
        The criteria is the squared error between successive centroids (`self.V`).
    """

    def __init__(self, n_clusters=3, metric=euclidean, focal_max_size=None):

        self.C = n_clusters
        self.alpha_ = 1
        self.beta_ = 2
        self.delta_ = 10
        self.lbda_ = 1.0

        self.metric = metric
        self.nb_inner_it = 10  # number of iterations for V optimization
        self.nb_outer_it = 20  # number of iterations for the overall optimization
        self.inner_lr = 1e-3
        self.focal_max_size = focal_max_size  # constraints on the size of focal elements (2 would be a good value)
        self.inner_convergence_criteria = None
        self.outter_convergence_criteria = 1e-3

        # Description of the focal elements
        #   - self.dim is the number of focal elements
        #   - self.S describes the focal elements wrt clusters (binary matrix of shape (dim, nb_clusters))

        #       self.S[c,:] describes to which clusters is associated the c-th focal element
        #       self.S[0,:] is the empty focal element
        #       self.S[-1,:] is the Omega focal element
        #   - self.focal_sizes

        self.S = torch.tensor(
            [vals for vals in product([False, True], repeat=n_clusters)],
            dtype=torch.bool,
            requires_grad=False,
        )
        if self.focal_max_size:
            self.S = self.S[torch.sum(self.S, 1) <= self.focal_max_size, :]
        self.F = torch.where(self.S, torch.tensor(1), torch.tensor(0))
        self.dim_ = self.S.shape[0]

        # pre-compute the size of the focal elements (as the number of singleton cluster it is associated to)
        self.focal_sizes = torch.sum(self.S, axis=1)
        # pre-compute the indices if the singleton focal elements
        self.id_singletons = torch.where(self.focal_sizes == 1)[0].flip(0)

        # model parameters
        self.M = None  # shape (nb_examples, dim): mass assignment for each example to the 2^C clusters
        self.V = None  # shape (nb_clusters, sz, d): centroids description

    def _update_assignment(self, X):
        """Assign the mass values (`self.M`) for each example in `X` according
        to the current centroids. It is assigned according the proximity with the
        centroids.

        This function modifies the assignement matrix `self.M`.

        Args:
            `X` ( numpy.ndarray ): shape (nb_examples, sz, d) where nb_examples
             is the number of examples, sz is the length of time series and `d`
             is the dimension
        """
        D = np.zeros((X.shape[0], self.dim_ - 1))
        # shape (nb_examples, nb_focal_elems-1) : metric value between the current centroids and examples
        # d(x_i, v_A) for all i, A\neq \emptyset
        # We have `self.dim_ - 1` because the emptyset has no centroid to compute.

        for j in range(self.dim_ - 1):
            # compute all distances wrt to the j-th focal element
            V = self.V[j + 1, :, :].repeat(X.shape[0], 1, 1)
            D[:, j] = self.metric(X, V).numpy()

        # compute |A|^\frac{-\alpha}{β−1} d (xi , v A )^\frac{-1}{β−1} for all i, A\neq \emptyset
        Q = np.power(self.focal_sizes[1:], -self.alpha_ / (self.beta_ - 1)) * np.power(
            (D), -1 / (self.beta_ - 1)
        )
        # replace the 0 distance by a 1 in Q (complete assignment)
        W = np.where(D == 0)
        Q[W[0], :] = 0
        Q[W[0], W[1]] = 1

        R = torch.sum(Q, axis=1) + np.power(self.delta_, -2 / (self.beta_ - 1))

        # compute the new masses for the focal element except the empty set
        self.M[:, 1:] = torch.div(Q.T, R).T
        # adjust the masses of the total ignorance
        self.M[:, 0] = 0
        self.M[:, 0] = 1 - torch.sum(self.M, axis=1)

    def _update_centroids(self, X):
        """Function to update the centroids of the clusters, ie. `self.V` by finding the
        good "average positions" of clusters.
        The centroid position of a focal element is a compromise between the barycenter
        of the elements of the clusters and its position wrt to the centroids of the singleton
        clusters.

        Args:
            X ( numpy.ndarray ): shape (nb_examples, sz, d)
        """

        self.V.requires_grad_()  # enable gradient computing

        # optimizer for the second inner-optimization
        optimizer = optim.Adam([self.V], lr=self.inner_lr)

        error = np.inf

        for i in range(self.nb_inner_it):
            optimizer.zero_grad()

            previous_error = error
            error = 0
            for iA in range(1, self.dim_):
                # inertia wrt the data
                V = self.V[iA, :, :].repeat(X.shape[0], 1, 1)
                D = self.metric(X, V)
                D = (
                    self.focal_sizes[iA] ** self.alpha_
                    * self.M[:, iA] ** self.beta_
                    * D
                )
                error += torch.sum(D)

                # inertia wrt to the singleton barycenters
                error += self.lbda_ * torch.sum(
                    torch.stack(
                        [
                            self.metric(
                                self.V[self.id_singletons[i], :, :].unsqueeze(0),
                                self.V[iA, :, :].unsqueeze(0),
                            )
                            for i in range(self.C)
                            if self.S[iA, i]
                        ]
                    )
                )

            if self.inner_convergence_criteria and i > 0:
                with torch.no_grad():
                    reldiff = torch.abs(previous_error - error) / previous_error
                    if reldiff < self.inner_convergence_criteria:
                        logging.info(
                            f"Inner optim converged after {i}-th loops: {reldiff}."
                        )
                        break

            error.backward()
            optimizer.step()

        self.V = self.V.detach()  # disable gradient computing

    def _jaccard(self, X):
        """
        Compute the Jaccard metric

        TODO
        -----
        To be optimized
        """
        self.J = 0

        for iA in range(1, self.dim_):
            for i in range(X.shape[0]):
                self.J += (
                    self.focal_sizes[iA] ** self.alpha_
                    * self.M[i, iA] ** self.beta_
                    * self.metric(
                        X[i, :, :].unsqueeze(2), self.V[iA, :, :].unsqueeze(2)
                    ).mean()
                    + self.lbda_**2 * self.M[i, 0] ** self.beta_
                )
        return self.J

    def fit(self, X):
        if isinstance(X, np.ndarray):
            X = torch.Tensor(X)
        else:
            X = torch.Tensor(X.values)
        # Add a third dimension to 1 for numeric data
        if len(X.shape) == 2:
            X = X.unsqueeze(2)

        self.M = torch.zeros((X.shape[0], self.dim_))

        # init V: choose randomly self.C time series
        idx = np.random.randint(X.shape[0], size=self.C)

        # then, we initialise all the barycenters with the euclidean mean (for the sake of simplicity)
        self.V = torch.zeros((self.dim_, X.shape[1], X.shape[2]), dtype=torch.float32)
        for i in range(1, self.dim_):
            self.V[i] = torch.mean(X[idx][self.S[i]], axis=0)

        # main loop : repeat the main two optimization stages
        self.Jall = []
        i = 0
        J_true = True
        self.J_old = float("inf")
        while J_true and i < self.nb_outer_it:
            self._update_assignment(X)
            self._update_centroids(X)
            self._jaccard(X)

            logging.info(f"{i}/{self.nb_outer_it} - {self.J}")
            J_true = (
                abs(self.J - self.J_old) >= self.outter_convergence_criteria
                if self.outter_convergence_criteria
                else True
            )
            self.J_old = self.J
            self.Jall.append(self.J)
            i += 1



if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from tslearn.datasets import CachedDatasets
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
    from ecm_ts import ECM_Clustering

    from tslearn.metrics import soft_dtw, SoftDTWLossPyTorch

    dtw_metric = SoftDTWLossPyTorch(gamma=0.1)

    seed = 0
    np.random.seed(seed)
    X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
    X_train = X_train[y_train < 4]  # Keep first 3 classes
    np.random.shuffle(X_train)
    # Keep only 50 time series
    X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train[:50])
    # Make time series shorter
    X_train = TimeSeriesResampler(sz=40).fit_transform(X_train)
    sz = X_train.shape[1]

    nb_clusters = 3
    clusterer = ECM_Clustering(nb_clusters)
    clusterer.nb_inner_it = 10
    clusterer.metric = dtw_metric

    clusterer.fit(X_train)

    print(clusterer.V)
    print(clusterer.V[clusterer.id_singletons])
    plt.figure()
    for yi in range(nb_clusters):
        plt.subplot(nb_clusters, 1, yi + 1)
        plt.plot(clusterer.V[clusterer.id_singletons[yi]].ravel(), "r-")
