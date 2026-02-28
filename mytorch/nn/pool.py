import numpy as np
from .resampling import *


class MaxPool2d_stride1():
    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        self.A = A
        output_width = A.shape[2] - self.kernel + 1
        output_height = A.shape[3] - self.kernel + 1
        Z = np.zeros((A.shape[0], A.shape[1], output_width, output_height))
        self.max_indices = np.zeros((A.shape[0], A.shape[1], output_width, output_height, 2), dtype=int)  # 存储 (m, n)

        for b in range(A.shape[0]):
            for c in range(A.shape[1]):
                for i in range(output_width):
                    for j in range(output_height):
                        patch = A[b, c, i:i + self.kernel, j:j + self.kernel]
                        max_val = np.max(patch)
                        Z[b, c, i, j] = max_val
                        max_idx = np.unravel_index(np.argmax(patch), patch.shape)
                        self.max_indices[b, c, i, j] = max_idx
        return Z

    def backward(self, dLdZ):
        dLdA = np.zeros(self.A.shape)
        for b in range(dLdZ.shape[0]):
            for c in range(dLdZ.shape[1]):
                for i in range(dLdZ.shape[2]):
                    for j in range(dLdZ.shape[3]):
                        m, n = self.max_indices[b, c, i, j]
                        dLdA[b, c, i + m, j + n] += dLdZ[b, c, i, j]
        return dLdA

class MeanPool2d_stride1():
    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        batch_size, in_channels, input_width, input_height = A.shape
        output_width = input_width - self.kernel + 1
        output_height = input_height - self.kernel + 1
        Z = np.zeros((batch_size, in_channels, output_width, output_height))

        for b in range(batch_size):
            for c in range(in_channels):
                for i in range(output_width):
                    for j in range(output_height):
                        patch = A[b, c, i:i + self.kernel, j:j + self.kernel]
                        Z[b, c, i, j] = np.mean(patch)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        batch_size, in_channels, input_width, input_height = self.A.shape
        dLdA = np.zeros(self.A.shape)
        kernel_area = self.kernel * self.kernel

        for b in range(batch_size):
            for c in range(in_channels):
                for i in range(dLdZ.shape[2]):
                    for j in range(dLdZ.shape[3]):
                        dLdA[b, c, i:i + self.kernel, j:j + self.kernel] += dLdZ[b, c, i, j] / kernel_area

        return dLdA


class MaxPool2d():
    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(self.kernel)  # TODO
        self.downsample2d = Downsample2d(self.stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z=self.maxpool2d_stride1.forward(A)
        Z=self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA=self.downsample2d.backward(dLdZ)
        dLdA=self.maxpool2d_stride1.backward(dLdA)

        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(self.kernel)  # TODO
        self.downsample2d = Downsample2d(self.stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z=self.meanpool2d_stride1.forward(A)
        Z=self.downsample2d.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA=self.downsample2d.backward(dLdZ)
        dLdA=self.meanpool2d_stride1.backward(dLdA)
        return dLdA
