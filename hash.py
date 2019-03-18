from __future__ import division
from __future__ import print_function
import numpy as np


class RandomProjection(object):
    def __init__(self, bit, r=2.5, verbose=True):
        # bit: codelength
        self.projector = None
        self.L = bit
        self.r = r

    def class_message(self):
        return "RandomProjection , bit length: {}, r: {}".format(self.L, self.r)

    def fit(self, vecs, niter=0, seed=0):
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        # N vectors of D dimension
        N, D = vecs.shape

        self.projector = np.random.normal(size=(D, self.L))
        for i in range(self.L):
            self.projector[:, i] = np.random.normal(size=D)
        self.b = np.random.uniform(low=0, high=self.r, size=(self.L))

        return self

    def encode(self, vecs):
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape
        # print(N, D)
        b = np.empty((N, self.L))
        for i in range(N):
            b[i, :] = self.b

        codes = np.floor((vecs @ self.projector + b) /
                         self.r)   # (N, D) (D, L) -> (N, L)

        assert codes.shape == (N, self.L)

        return codes

    def decode(self, codes):
        pass

    def compress(self, vecs):
        return self.encode(vecs)
