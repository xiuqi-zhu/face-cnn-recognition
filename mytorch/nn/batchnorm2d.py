# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class BatchNorm2d:

    def __init__(self, num_features, alpha=0.9):
        # num features: number of channels
        self.alpha = alpha
        self.eps = 1e-8

        self.Z = None
        self.NZ = None
        self.BZ = None

        self.BW = np.ones((1, num_features, 1, 1))
        self.Bb = np.zeros((1, num_features, 1, 1))
        self.dLdBW = np.zeros((1, num_features, 1, 1))
        self.dLdBb = np.zeros((1, num_features, 1, 1))

        self.M = np.zeros((1, num_features, 1, 1))
        self.V = np.ones((1, num_features, 1, 1))

        # inference parameters
        self.running_M = np.zeros((1, num_features, 1, 1))
        self.running_V = np.ones((1, num_features, 1, 1))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the
        training phase of the problem or are we in the inference phase.
        So see what values you need to recompute when eval is True.
        """

        if eval:
            M = self.running_M
            V = self.running_V
            NZ = (Z - M) / np.sqrt(V + self.eps)
            BZ = self.BW * NZ + self.Bb
            return BZ

        self.Z = Z
        N, C, H, W = Z.shape
        self.N = N * H * W

        self.M = np.mean(Z, axis=(0, 2, 3), keepdims=True)
        self.V = np.var(Z, axis=(0, 2, 3), keepdims=True)

        self.NZ = (Z - self.M) / (np.sqrt(self.V + self.eps))
        self.BZ = self.BW * self.NZ + self.Bb

        self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M
        self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V

        return self.BZ

    def backward(self, dLdBZ):
        self.dLdBW = np.sum(dLdBZ * self.NZ, axis=(0, 2, 3), keepdims=True)
        self.dLdBb = np.sum(dLdBZ, axis=(0, 2, 3), keepdims=True)

        dLdNZ = dLdBZ * self.BW

        dLdV = np.sum(dLdNZ * (self.Z - self.M) * (-0.5) * (self.V + self.eps) ** (-3/2),
                      axis=(0, 2, 3), keepdims=True)

        dLdM = np.sum(dLdNZ * (-1 / np.sqrt(self.V + self.eps)),
                      axis=(0, 2, 3), keepdims=True) + dLdV * np.mean(-2 * (self.Z - self.M),
                      axis=(0, 2, 3), keepdims=True)

        dLdZ = (dLdNZ / np.sqrt(self.V + self.eps)) + \
               (dLdV * 2 * (self.Z - self.M) / self.N) + \
               (dLdM / self.N)

        return dLdZ
