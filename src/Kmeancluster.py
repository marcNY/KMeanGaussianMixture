import numpy as np




class kmeanclusterer():
    def __init__(self, X, ncluster):
        self.data = X
        self.n_cluster = ncluster
        self.mu = self._init_centroids()
        self.cass = self._assign()

    @property
    def ndim(self):
        return self.data.shape[1]

    @property
    def npoints(self):
        return self.data.shape[0]

    def _init_centroids(self):
        centroids=self.data.copy()
        np.random.shuffle(centroids)
        return centroids[:self.n_cluster]

    def _assign(self):
        f = lambda x: np.argmin(np.linalg.norm(self.mu - x, axis=1))
        B = np.apply_along_axis(f, 1, self.data)
        return B

    def _compute_mean(self):
        f = lambda x: np.mean(self.data[self.cass == x], 0)
        mu = self.mu
        for i in range(self.n_cluster):
            if np.sum(self.cass == i) > 0:
                mu[i] = f(i)
            else:
                print('The centroid has no pints linked to it')
        return mu

    def runiteration(self):
        self.mu = self._compute_mean()
        self.cass = self._assign()
        return self.cass, self.mu

