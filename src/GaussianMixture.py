import numpy as np
import math


class GMclusterer():
    def __init__(self, X, ncluster):
        self.data = X
        self.n_cluster = ncluster
        self.pi = np.ones(ncluster) / ncluster
        self.mu = self._init_mean()
        self.sig = self.init_sig(self.n_cluster, self.ndim)

    @property
    def ndim(self):
        return self.data.shape[1]

    @property
    def npoints(self):
        return self.data.shape[0]

    ## Initialize functions
    def _init_mean(self):
        min = np.amin(self.data, axis=0)
        max = np.amax(self.data, axis=0)
        rand = np.random.rand(self.n_cluster, self.ndim)
        mu = min + rand * (max - min)
        return mu

    def init_sig(self, ncluster, ndim):
        min = np.amin(self.data, axis=0)
        max = np.amax(self.data, axis=0)
        lbda = np.linalg.norm(max - min) / float(ncluster)
        S = np.identity(ndim) * lbda ** 2
        b = np.repeat(S[:, :, np.newaxis], ncluster, axis=2)
        return b

    ## Estep assignement step
    # compute phi for a point x and a class j
    def computephi(self):
        def computePhiforapoint(self, x, j):
            return self.pi[j] * GMclusterer.density(x, self.mu[j], self.sig[:, :, j])

        Phi = np.zeros([self.npoints, self.n_cluster])
        for i in range(self.npoints):  # looping in all points
            for j in range(self.n_cluster):  # looping for all classes
                Phi[i, j] = computePhiforapoint(self, self.data[i], j)
        Den = np.sum(Phi, axis=1)
        Den = np.repeat(Den[:, np.newaxis], self.n_cluster, axis=1)
        Phi = np.divide(Phi, Den)
        return Phi

    ## Mstep maximization step

    def _compute_pi(self, phi):
        self.pi = np.sum(phi, axis=0) / self.npoints
        return self.pi

    def _compute_mu(self, phi):
        nk = np.sum(phi, axis=0)

        for i in range(self.n_cluster):
            iphi = phi[:, i]
            mult= np.repeat(iphi[:,np.newaxis], self.ndim, axis=1)
            self.mu[i] = 1 / float(nk[i]) * np.sum(mult* self.data,axis=0)
        return self.mu

    def _compute_sig(self, phi):
        nk = np.sum(phi, axis=0)

        for i in range(self.n_cluster):
            iphi = phi[:, i]
            mult= np.repeat(iphi[:,np.newaxis], self.ndim, axis=1)
            m = np.dot(np.transpose(self.data - self.mu[i,:]), mult* (self.data - self.mu[i,:]))
            self.sig[:, :, i] = 1 / float(nk[i]) * m
            assert(np.allclose(self.sig[:,:,i], self.sig[:,:,i].T, atol=1e-8))
        return self.sig

    def update_param(self, phi):
        self._compute_pi(phi)
        self._compute_mu(phi)
        self._compute_sig(phi)
        return self.pi, self.mu, self.sig
    ## EM Steps

    def runiteration(self):
        phi=self.computephi()
        pi,mu,sig=self.update_param(phi)
        return phi,pi,mu,sig
    ## Tools

    @staticmethod
    def density(x, mu, sig):
        try:
            siginv = np.linalg.inv(sig)
        except:
            raise IOError('The matrix %s is non singular'%sig)
        num = np.exp(-0.5 * np.dot(np.dot(np.transpose(x - mu), siginv), x - mu))
        den = np.sqrt(np.linalg.det(2 * math.pi * sig))
        return num / den

    @staticmethod
    def get_assign(phi):
        return np.argmax(phi, axis=1)
