import numpy as np

class Flatten():
    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """
        self.A_shape = A.shape  # Store input shape for backward pass
        Z = np.reshape(A, (A.shape[0], -1))  # Flatten to (batch_size, in_channels * in_width)
        return Z


    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """
        dLdA = np.reshape(dLdZ, self.A_shape)  # Reshape to original input shape
        return dLdA

