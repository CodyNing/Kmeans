import numpy as np

DISTS_CACHE = None
np.random.seed(0)

class KMeans():

    def __init__(self, n_clusters: int, init: str='random', max_iter = 300):
        """

        :param n_clusters: number of clusters
        :param init: centroid initialization method. Should be either 'random' or 'kmeans++'
        :param max_iter: maximum number of iterations
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None # Initialized in initialize_centroids()

    def fit(self, X: np.ndarray):
        self.initialize_centroids(X)
        # print(self.centroids.shape)
        iteration = 0
        clustering = np.zeros(X.shape[0])
        # while iteration < self.max_iter:
        for i in range(self.max_iter):
            # calculate pairwise dist and get the argmin index as cluster label
            clustering = self.euclidean_distance(self.centroids, X).argmin(axis=0)
            self.update_centroids(clustering, X)
            # iteration += 1

        # define the final cluster    
        clustering = self.euclidean_distance(self.centroids, X).argmin(axis=0)
        return clustering

    def update_centroids(self, clustering: np.ndarray, X: np.ndarray):
        #your code
        for i in range(self.n_clusters):
            # get rows for cluster i
            rows = X[clustering == i]
            # update centroid with mean of the cluster (if cluster not empty)
            if rows.shape[0] != 0:
                self.centroids[i] = rows.mean(axis=0)

    def initialize_centroids(self, X: np.ndarray):
        """
        Initialize centroids either randomly or using kmeans++ method of initialization.
        :param X:
        :return:
        """
        if self.init == 'random':
            # your code
            self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        elif self.init == 'kmeans++':
            # your code
            self.centroids = np.zeros((self.n_clusters, X.shape[1]))

            # Randomly select the first centroid from the data points
            self.centroids[0] = X[np.random.choice(X.shape[0], 1)]

            for i in range(1, self.n_clusters):
                # shape(X.shape[0], ) row number = number of nearest distance
                nearest_dist = self.euclidean_distance(self.centroids[0:i], X).min(axis=0)
                # normalize into probability
                p = nearest_dist / nearest_dist.sum()
                # sample against distribution
                row = np.random.choice(np.arange(p.shape[0]), p=p)
                # use the sampled row as the next centroid
                self.centroids[i] = X[row]
        else:
            raise ValueError('Centroid initialization method should either be "random" or "k-means++"')

    def euclidean_distance(self, X1:np.ndarray, X2:np.ndarray):
        """
        Computes the euclidean distance between all pairs (x,y) where x is a row in X1 and y is a row in X2.
        Tip: Using vectorized operations can hugely improve the efficiency here.
        :param X1:
        :param X2:
        :return: Returns a matrix `dist` where `dist_ij` is the distance between row i in X1 and row j in X2.
        """
        # your code
        # pairwise broadcast minus
        # https://stackoverflow.com/questions/51091560/numpy-indexing-using-none-for-pairwise-operations
        return np.sqrt(((X1[:, :, np.newaxis] - X2[:, :, np.newaxis].T) ** 2).sum(axis=1))

    def silhouette(self, clustering: np.ndarray, X: np.ndarray, cache: bool = False):
        # your code

        global DISTS_CACHE
        # attempt to cache the distance matrix since it take super long to calc
        if cache:
            if DISTS_CACHE is None:
                dists = self.euclidean_distance(X, X)
                DISTS_CACHE = dists
            else:
                dists = DISTS_CACHE
        else:
            dists = self.euclidean_distance(X, X)

        # Count the size of each cluster
        cluster_freq = np.bincount(clustering)
        # distances of each sample to each cluster
        cluster_dists = np.zeros((X.shape[0], self.n_clusters))
        for i in range(X.shape[0]):
            # group by each cluster, sum the distance of sameple i to other samples in that cluster
            # https://stackoverflow.com/questions/4373631/sum-array-by-number-in-numpy
            cluster_dists[i] = np.bincount(clustering, weights=dists[i], minlength=self.n_clusters)
        
        # get the intra clsuter distance sum of each sample 
        intra_dists = cluster_dists[np.arange(X.shape[0]), clustering]
        # set the intra cluster distance sum of each sample to inf so we can ignore them later
        cluster_dists[np.arange(X.shape[0]), clustering] = np.inf
        # take the minimum inter cluster distance mean of all other cluster
        inter_dists = (cluster_dists / cluster_freq).min(axis=1)

        # subtract the cluster size by one (exclude self for intra class distance)
        intra_sizes = cluster_freq - 1
        # repeat the intra cluster size to each cluster member
        # https://stackoverflow.com/questions/46868056/how-to-gather-elements-of-specific-indices-in-numpy
        intra_sizes = intra_sizes.take(clustering, mode='clip')

        # divide to get the intra class distance mean for each sample
        intra_dists /= intra_sizes

        # silhouette formula
        numerator = inter_dists - intra_dists
        denominator = np.maximum(inter_dists, intra_dists)

        # take silhouette mean of individual silhouette score
        return np.nan_to_num(numerator / denominator).mean()

    def silhouette_1(self, clustering: np.ndarray, X: np.ndarray):
        # your code
        from tqdm import tqdm

        # Count the size of each cluster
        cluster_freq = np.bincount(clustering)
        # distances of each sample to each cluster
        scores = np.zeros(X.shape[0])
        for i in tqdm(range(X.shape[0])):
            # group by each cluster, sum the distance of sameple i to other samples in that cluster
            # https://stackoverflow.com/questions/4373631/sum-array-by-number-in-numpy
            dist = self.euclidean_distance(X[i].reshape(1, -1), X)[0]
            cluster_dist = np.bincount(clustering, weights=dist, minlength=self.n_clusters)
            cluster = clustering[i]
            intra_dist = cluster_dist[cluster] / (cluster_freq[cluster] - 1)
            cluster_dist[cluster] = np.inf
            inter_dist = (cluster_dist / cluster_freq).min()
            numerator = inter_dist - intra_dist
            denominator = np.maximum(inter_dist, intra_dist)
            score = np.nan_to_num(numerator / denominator)
            scores[i] = score
        
        return scores.mean()
